"""
V2 Declarative Benchmark Manifest Validator (Phase 2.6.2.2 - Step 3).
Implements the Three-Layer Validation Architecture:
- Layer A: JSON Schema Validation (jsonschema Draft 2020-12).
- Layer B: Python Cross-Field Business Logic Validation.
- Layer C: Document Integrity & Text Grounding Validation.
"""
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Union

import jsonschema

from src.models.benchmark import (
    BenchmarkManifest,
    BenchmarkEvidencePresence,
    BenchmarkVerificationStatus,
    BenchmarkParentComplianceStatus,
    BenchmarkDocumentComplianceVerdict,
)
from src.parsers.pdf_parser import TargetPDFParser


# ---------------------------------------------------------------------------
# Custom Benchmark Validation Exceptions
# ---------------------------------------------------------------------------

class BenchmarkValidationError(Exception):
    """Base exception for all benchmark manifest validation failures."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class BenchmarkSchemaError(BenchmarkValidationError):
    """Raised when JSON Schema Draft 2020-12 validation fails (Layer A)."""
    def __init__(self, message: str, json_path: str, schema_path: str, details: Optional[Dict[str, Any]] = None):
        formatted_msg = f"[Layer A: JSON Schema Error] At '{json_path}': {message}"
        super().__init__(formatted_msg, details)
        self.json_path = json_path
        self.schema_path = schema_path


class BenchmarkCrossFieldError(BenchmarkValidationError):
    """Raised when Python cross-field logic or invariant validation fails (Layer B)."""
    def __init__(self, message: str, field_or_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        prefix = f"[Layer B: Cross-Field Error on '{field_or_id}'] " if field_or_id else "[Layer B: Cross-Field Error] "
        super().__init__(f"{prefix}{message}", details)
        self.field_or_id = field_or_id


class BenchmarkIntegrityError(BenchmarkValidationError):
    """Raised when physical document integrity or evidence grounding fails (Layer C)."""
    def __init__(self, message: str, document_or_block: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        prefix = f"[Layer C: Document Integrity Error on '{document_or_block}'] " if document_or_block else "[Layer C: Document Integrity Error] "
        super().__init__(f"{prefix}{message}", details)
        self.document_or_block = document_or_block


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_schema_path() -> Path:
    """Returns the absolute filesystem path to the benchmark JSON Schema."""
    return Path(__file__).resolve().parent / "schemas" / "benchmark_manifest.schema.json"


def _compute_sha256(file_path: Path) -> str:
    """Computes the 64-character lowercase SHA-256 hexadecimal digest of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def _normalize_text(text: str) -> str:
    """Normalizes whitespace and converts text to lowercase for tolerant excerpt matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


# ---------------------------------------------------------------------------
# Layer A: JSON Schema Validation
# ---------------------------------------------------------------------------

def validate_schema(data: Dict[str, Any]) -> None:
    """
    Validates raw dictionary data against the benchmark JSON Schema Draft 2020-12.

    Args:
        data: The deserialized JSON dictionary to validate.

    Raises:
        BenchmarkSchemaError: If data does not conform to the schema.
    """
    schema_path = get_schema_path()
    if not schema_path.is_file():
        raise FileNotFoundError(f"Benchmark schema file not found at: {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)

    if errors:
        first_error = errors[0]
        json_path = "/" + "/".join(str(p) for p in first_error.path) if first_error.path else "/"
        schema_path_str = "/" + "/".join(str(p) for p in first_error.schema_path) if first_error.schema_path else "/"
        raise BenchmarkSchemaError(
            message=first_error.message,
            json_path=json_path,
            schema_path=schema_path_str,
            details={"all_errors": [e.message for e in errors]}
        )


# ---------------------------------------------------------------------------
# Layer B: Python Cross-Field Validation
# ---------------------------------------------------------------------------

def validate_cross_field_rules(manifest: BenchmarkManifest) -> None:
    """
    Validates logical invariants, ID uniqueness, parent-child hierarchies,
    count conservation, coverage calculations, parent status aggregation,
    and document verdict consistency.

    Args:
        manifest: Strongly typed BenchmarkManifest instance.

    Raises:
        BenchmarkCrossFieldError: If any business rule or invariant is violated.
    """
    # -----------------------------------------------------------------------
    # 1. ID Uniqueness & Hierarchy
    # -----------------------------------------------------------------------
    seen_parent_ids: Set[str] = set()
    seen_sub_ids: Set[str] = set()

    all_subreqs = []

    for parent in manifest.parent_requirements:
        pid = parent.parent_id.strip()
        if not pid:
            raise BenchmarkCrossFieldError("Parent ID cannot be empty or whitespace.", field_or_id="parent_id")
        if pid in seen_parent_ids:
            raise BenchmarkCrossFieldError(f"Duplicate parent ID '{pid}' encountered.", field_or_id=pid)
        seen_parent_ids.add(pid)

        # -------------------------------------------------------------------
        # 2. Parent Non-Empty
        # -------------------------------------------------------------------
        if not parent.sub_requirements:
            raise BenchmarkCrossFieldError(
                f"Parent requirement '{pid}' contains zero sub-requirements. Parents must contain at least one sub-requirement.",
                field_or_id=pid
            )

        # -------------------------------------------------------------------
        # 3. Parent Mandatory Consistency
        # -------------------------------------------------------------------
        actual_parent_mandatory = any(sr.is_mandatory for sr in parent.sub_requirements)
        if parent.is_mandatory != actual_parent_mandatory:
            raise BenchmarkCrossFieldError(
                f"Parent '{pid}' has is_mandatory={parent.is_mandatory}, but child requirements have is_mandatory={actual_parent_mandatory}. "
                f"Parent is_mandatory must equal whether at least one child sub-requirement is mandatory.",
                field_or_id=pid
            )

        for sr in parent.sub_requirements:
            sid = sr.sub_id.strip()
            if not sid:
                raise BenchmarkCrossFieldError("Sub-requirement ID cannot be empty.", field_or_id=f"{pid}.?")
            if sid in seen_sub_ids:
                raise BenchmarkCrossFieldError(f"Duplicate sub-requirement ID '{sid}' encountered across manifest.", field_or_id=sid)
            seen_sub_ids.add(sid)

            expected_prefix = f"{pid}."
            if not sid.startswith(expected_prefix):
                raise BenchmarkCrossFieldError(
                    f"Sub-requirement '{sid}' does not follow parent ID prefix '{expected_prefix}'.",
                    field_or_id=sid
                )

            # ---------------------------------------------------------------
            # 4. Evidence Presence & Status Compatibility
            # ---------------------------------------------------------------
            presence = sr.ground_truth.retrieval.evidence_presence
            status = sr.ground_truth.verification.expected_status
            evidence_list = sr.ground_truth.retrieval.expected_evidence

            if presence == BenchmarkEvidencePresence.EXISTS:
                permitted = {
                    BenchmarkVerificationStatus.FULFILLED,
                    BenchmarkVerificationStatus.CONTRADICTED,
                    BenchmarkVerificationStatus.NEEDS_REVIEW,
                }
                if status not in permitted:
                    raise BenchmarkCrossFieldError(
                        f"Sub-requirement '{sid}' has evidence_presence='EXISTS' with incompatible expected_status='{status.value}'. "
                        f"Permitted statuses for EXISTS: {[p.value for p in permitted]}.",
                        field_or_id=sid
                    )
                if len(evidence_list) == 0:
                    raise BenchmarkCrossFieldError(
                        f"Sub-requirement '{sid}' has evidence_presence='EXISTS' but expected_evidence list is empty. "
                        f"EXISTS requires at least one evidence annotation.",
                        field_or_id=sid
                    )

            elif presence == BenchmarkEvidencePresence.EXPLICIT_OMISSION:
                permitted = {BenchmarkVerificationStatus.EXPLICIT_OMISSION}
                if status not in permitted:
                    raise BenchmarkCrossFieldError(
                        f"Sub-requirement '{sid}' has evidence_presence='EXPLICIT_OMISSION' with incompatible expected_status='{status.value}'. "
                        f"Permitted status: 'EXPLICIT_OMISSION'.",
                        field_or_id=sid
                    )
                if len(evidence_list) == 0:
                    raise BenchmarkCrossFieldError(
                        f"Sub-requirement '{sid}' has evidence_presence='EXPLICIT_OMISSION' but expected_evidence list is empty. "
                        f"EXPLICIT_OMISSION requires at least one disclaimer evidence annotation.",
                        field_or_id=sid
                    )

            elif presence == BenchmarkEvidencePresence.ABSENT:
                permitted = {
                    BenchmarkVerificationStatus.MISSING,
                    BenchmarkVerificationStatus.NEEDS_REVIEW,
                }
                if status not in permitted:
                    raise BenchmarkCrossFieldError(
                        f"Sub-requirement '{sid}' has evidence_presence='ABSENT' with incompatible expected_status='{status.value}'. "
                        f"Permitted statuses for ABSENT: {[p.value for p in permitted]}.",
                        field_or_id=sid
                    )
                if len(evidence_list) > 0:
                    raise BenchmarkCrossFieldError(
                        f"Sub-requirement '{sid}' has evidence_presence='ABSENT' but defines {len(evidence_list)} evidence annotation(s). "
                        f"ABSENT requires expected_evidence to be strictly empty.",
                        field_or_id=sid
                    )

            all_subreqs.append(sr)

        # -------------------------------------------------------------------
        # 5. Parent Status Aggregation Rules
        # -------------------------------------------------------------------
        m_items = [sr for sr in parent.sub_requirements if sr.is_mandatory]
        o_items = [sr for sr in parent.sub_requirements if not sr.is_mandatory]
        m_total = len(m_items)
        o_total = len(o_items)

        if m_total > 0:
            m_review = sum(1 for sr in m_items if sr.ground_truth.verification.expected_status == BenchmarkVerificationStatus.NEEDS_REVIEW)
            m_fulfilled = sum(1 for sr in m_items if sr.ground_truth.verification.expected_status == BenchmarkVerificationStatus.FULFILLED)
            m_contra = sum(1 for sr in m_items if sr.ground_truth.verification.expected_status == BenchmarkVerificationStatus.CONTRADICTED)

            if m_review > 0:
                calc_status = BenchmarkParentComplianceStatus.NEEDS_REVIEW
            elif m_fulfilled == m_total and m_contra == 0:
                calc_status = BenchmarkParentComplianceStatus.FULLY_COMPLIANT
            elif m_fulfilled > 0:
                calc_status = BenchmarkParentComplianceStatus.PARTIALLY_COMPLIANT
            else:
                calc_status = BenchmarkParentComplianceStatus.NON_COMPLIANT
        else:
            # Optional-only parent
            o_review = sum(1 for sr in o_items if sr.ground_truth.verification.expected_status == BenchmarkVerificationStatus.NEEDS_REVIEW)
            o_fulfilled = sum(1 for sr in o_items if sr.ground_truth.verification.expected_status == BenchmarkVerificationStatus.FULFILLED)

            if o_review > 0:
                calc_status = BenchmarkParentComplianceStatus.NEEDS_REVIEW
            elif o_fulfilled == o_total:
                calc_status = BenchmarkParentComplianceStatus.FULLY_COMPLIANT
            elif o_fulfilled > 0:
                calc_status = BenchmarkParentComplianceStatus.PARTIALLY_COMPLIANT
            else:
                calc_status = BenchmarkParentComplianceStatus.NON_COMPLIANT

        if parent.expected_compliance_status != calc_status:
            raise BenchmarkCrossFieldError(
                f"Parent '{pid}' declared expected_compliance_status='{parent.expected_compliance_status.value}', "
                f"but calculated aggregation rule yielded '{calc_status.value}'.",
                field_or_id=pid
            )

    # -----------------------------------------------------------------------
    # 6. Mandatory Requirement Invariant
    # -----------------------------------------------------------------------
    actual_m_total = sum(1 for sr in all_subreqs if sr.is_mandatory)
    if actual_m_total < 1:
        raise BenchmarkCrossFieldError(
            "Benchmark pair must contain at least one mandatory atomic sub-requirement (found 0).",
            field_or_id="mandatory_total"
        )

    # -----------------------------------------------------------------------
    # 7. Document-Level Counts and Coverage Consistency
    # -----------------------------------------------------------------------
    actual_m_fulfilled = sum(1 for sr in all_subreqs if sr.is_mandatory and sr.ground_truth.verification.expected_status == BenchmarkVerificationStatus.FULFILLED)
    actual_o_total = sum(1 for sr in all_subreqs if not sr.is_mandatory)
    actual_o_fulfilled = sum(1 for sr in all_subreqs if not sr.is_mandatory and sr.ground_truth.verification.expected_status == BenchmarkVerificationStatus.FULFILLED)

    c = manifest.expected_document_compliance
    if c.mandatory_total != actual_m_total:
        raise BenchmarkCrossFieldError(
            f"expected_document_compliance.mandatory_total={c.mandatory_total} does not match actual count of mandatory sub-requirements ({actual_m_total}).",
            field_or_id="mandatory_total"
        )
    if c.mandatory_fulfilled != actual_m_fulfilled:
        raise BenchmarkCrossFieldError(
            f"expected_document_compliance.mandatory_fulfilled={c.mandatory_fulfilled} does not match actual count of fulfilled mandatory sub-requirements ({actual_m_fulfilled}).",
            field_or_id="mandatory_fulfilled"
        )
    if c.optional_total != actual_o_total:
        raise BenchmarkCrossFieldError(
            f"expected_document_compliance.optional_total={c.optional_total} does not match actual count of optional sub-requirements ({actual_o_total}).",
            field_or_id="optional_total"
        )
    if c.optional_fulfilled != actual_o_fulfilled:
        raise BenchmarkCrossFieldError(
            f"expected_document_compliance.optional_fulfilled={c.optional_fulfilled} does not match actual count of fulfilled optional sub-requirements ({actual_o_fulfilled}).",
            field_or_id="optional_fulfilled"
        )

    # Status counts conservation
    status_counts = c.expected_status_counts
    actual_counts = {
        "FULFILLED": sum(1 for sr in all_subreqs if sr.ground_truth.verification.expected_status == BenchmarkVerificationStatus.FULFILLED),
        "EXPLICIT_OMISSION": sum(1 for sr in all_subreqs if sr.ground_truth.verification.expected_status == BenchmarkVerificationStatus.EXPLICIT_OMISSION),
        "CONTRADICTED": sum(1 for sr in all_subreqs if sr.ground_truth.verification.expected_status == BenchmarkVerificationStatus.CONTRADICTED),
        "MISSING": sum(1 for sr in all_subreqs if sr.ground_truth.verification.expected_status == BenchmarkVerificationStatus.MISSING),
        "NEEDS_REVIEW": sum(1 for sr in all_subreqs if sr.ground_truth.verification.expected_status == BenchmarkVerificationStatus.NEEDS_REVIEW),
    }

    if status_counts.fulfilled != actual_counts["FULFILLED"]:
        raise BenchmarkCrossFieldError(f"Status count FULFILLED={status_counts.fulfilled} != actual {actual_counts['FULFILLED']}.", field_or_id="expected_status_counts.FULFILLED")
    if status_counts.explicit_omission != actual_counts["EXPLICIT_OMISSION"]:
        raise BenchmarkCrossFieldError(f"Status count EXPLICIT_OMISSION={status_counts.explicit_omission} != actual {actual_counts['EXPLICIT_OMISSION']}.", field_or_id="expected_status_counts.EXPLICIT_OMISSION")
    if status_counts.contradicted != actual_counts["CONTRADICTED"]:
        raise BenchmarkCrossFieldError(f"Status count CONTRADICTED={status_counts.contradicted} != actual {actual_counts['CONTRADICTED']}.", field_or_id="expected_status_counts.CONTRADICTED")
    if status_counts.missing != actual_counts["MISSING"]:
        raise BenchmarkCrossFieldError(f"Status count MISSING={status_counts.missing} != actual {actual_counts['MISSING']}.", field_or_id="expected_status_counts.MISSING")
    if status_counts.needs_review != actual_counts["NEEDS_REVIEW"]:
        raise BenchmarkCrossFieldError(f"Status count NEEDS_REVIEW={status_counts.needs_review} != actual {actual_counts['NEEDS_REVIEW']}.", field_or_id="expected_status_counts.NEEDS_REVIEW")

    # Coverage calculations
    expected_m_cov = round((actual_m_fulfilled / actual_m_total) * 100.0, 1)
    if c.mandatory_coverage_pct != expected_m_cov:
        raise BenchmarkCrossFieldError(
            f"mandatory_coverage_pct={c.mandatory_coverage_pct} does not match recalculated value round(({actual_m_fulfilled}/{actual_m_total})*100, 1) = {expected_m_cov}.",
            field_or_id="mandatory_coverage_pct"
        )

    if actual_o_total > 0:
        expected_o_cov = round((actual_o_fulfilled / actual_o_total) * 100.0, 1)
        if c.optional_coverage_pct != expected_o_cov:
            raise BenchmarkCrossFieldError(
                f"optional_coverage_pct={c.optional_coverage_pct} does not match recalculated value round(({actual_o_fulfilled}/{actual_o_total})*100, 1) = {expected_o_cov}.",
                field_or_id="optional_coverage_pct"
            )
    else:
        if c.optional_coverage_pct is not None:
            raise BenchmarkCrossFieldError(
                f"optional_coverage_pct must be None when optional_total is 0 (got {c.optional_coverage_pct}).",
                field_or_id="optional_coverage_pct"
            )

    # -----------------------------------------------------------------------
    # 8. Document Verdict Consistency
    # -----------------------------------------------------------------------
    m_reviews = sum(1 for sr in all_subreqs if sr.is_mandatory and sr.ground_truth.verification.expected_status == BenchmarkVerificationStatus.NEEDS_REVIEW)
    m_contras = sum(1 for sr in all_subreqs if sr.is_mandatory and sr.ground_truth.verification.expected_status == BenchmarkVerificationStatus.CONTRADICTED)

    if m_reviews > 0:
        calc_verdict = BenchmarkDocumentComplianceVerdict.CONDITIONALLY_COMPLIANT
    elif expected_m_cov == 100.0 and m_contras == 0:
        calc_verdict = BenchmarkDocumentComplianceVerdict.COMPLIANT
    else:
        calc_verdict = BenchmarkDocumentComplianceVerdict.NON_COMPLIANT

    if c.verdict != calc_verdict:
        raise BenchmarkCrossFieldError(
            f"Declared document verdict='{c.verdict.value}' does not match calculated verdict='{calc_verdict.value}' "
            f"(mandatory_coverage={expected_m_cov}%, mandatory_reviews={m_reviews}, mandatory_contradictions={m_contras}).",
            field_or_id="verdict"
        )


# ---------------------------------------------------------------------------
# Layer C: Document Integrity & Evidence Grounding
# ---------------------------------------------------------------------------

def validate_document_integrity(
    manifest: BenchmarkManifest,
    base_dir: Optional[Union[str, Path]] = None
) -> None:
    """
    Validates physical document existence, SHA-256 hashes, page count,
    target block ID existence, page number consistency, and excerpt text grounding.

    Args:
        manifest: Strongly typed BenchmarkManifest instance.
        base_dir: Optional base directory to resolve relative document paths.
                  Defaults to the repository root directory.

    Raises:
        BenchmarkIntegrityError: If document hash, page, or evidence grounding fails.
    """
    if base_dir is None:
        # Defaults to repository root (two levels up from src/benchmarks)
        root_dir = Path(__file__).resolve().parent.parent.parent
    else:
        root_dir = Path(base_dir).resolve()

    # 1. Resolve Document Paths
    ref_rel = Path(manifest.metadata.reference_document.path)
    tgt_rel = Path(manifest.metadata.target_document.path)

    ref_path = (root_dir / ref_rel).resolve() if not ref_rel.is_absolute() else ref_rel
    tgt_path = (root_dir / tgt_rel).resolve() if not tgt_rel.is_absolute() else tgt_rel

    # 2. Check File Existence
    if not ref_path.is_file():
        raise BenchmarkIntegrityError(f"Reference template PDF not found at: {ref_path}", document_or_block=str(ref_rel))
    if not tgt_path.is_file():
        raise BenchmarkIntegrityError(f"Target document PDF not found at: {tgt_path}", document_or_block=str(tgt_rel))

    # 3. Verify SHA-256 Hashes
    actual_ref_sha = _compute_sha256(ref_path)
    if actual_ref_sha != manifest.metadata.reference_document.sha256.lower():
        raise BenchmarkIntegrityError(
            f"Reference PDF SHA-256 mismatch: expected {manifest.metadata.reference_document.sha256}, got {actual_ref_sha}.",
            document_or_block=str(ref_rel)
        )

    actual_tgt_sha = _compute_sha256(tgt_path)
    if actual_tgt_sha != manifest.metadata.target_document.sha256.lower():
        raise BenchmarkIntegrityError(
            f"Target PDF SHA-256 mismatch: expected {manifest.metadata.target_document.sha256}, got {actual_tgt_sha}.",
            document_or_block=str(tgt_rel)
        )

    # 4. Parse Target PDF with existing TargetPDFParser
    parser = TargetPDFParser(min_total_chars=10)
    try:
        parsed_doc = parser.parse_document(str(tgt_path))
    except Exception as exc:
        raise BenchmarkIntegrityError(
            f"Failed to parse target document '{tgt_rel.name}': {exc}",
            document_or_block=str(tgt_rel)
        ) from exc

    # 5. Page Count Validation
    if manifest.metadata.target_document.total_pages != parsed_doc.total_pages:
        raise BenchmarkIntegrityError(
            f"Target document total_pages mismatch: manifest declares {manifest.metadata.target_document.total_pages}, "
            f"but parsed document has {parsed_doc.total_pages} page(s).",
            document_or_block=str(tgt_rel)
        )

    # 6. Evidence Grounding Verification
    block_map = {b.block_id: b for b in parsed_doc.blocks}

    for parent in manifest.parent_requirements:
        for sr in parent.sub_requirements:
            evidence_list = sr.ground_truth.retrieval.expected_evidence
            for ev in evidence_list:
                # Block ID existence
                if ev.block_id not in block_map:
                    raise BenchmarkIntegrityError(
                        f"Sub-requirement '{sr.sub_id}' references unknown block_id '{ev.block_id}'. "
                        f"Parsed target document blocks: {sorted(list(block_map.keys()))}.",
                        document_or_block=ev.block_id
                    )

                matched_block = block_map[ev.block_id]

                # Page number consistency
                if ev.page != matched_block.page_number:
                    raise BenchmarkIntegrityError(
                        f"Sub-requirement '{sr.sub_id}' references block_id '{ev.block_id}' on page {ev.page}, "
                        f"but the block is actually located on page {matched_block.page_number}.",
                        document_or_block=ev.block_id
                    )

                # Excerpt grounding verification
                if ev.excerpt:
                    norm_excerpt = _normalize_text(ev.excerpt)
                    norm_block_text = _normalize_text(matched_block.text)

                    if norm_excerpt not in norm_block_text:
                        raise BenchmarkIntegrityError(
                            f"Sub-requirement '{sr.sub_id}' evidence excerpt not found in block '{ev.block_id}'.\n"
                            f"  Excerpt:    '{ev.excerpt}'\n"
                            f"  Block Text: '{matched_block.text[:120]}...'",
                            document_or_block=ev.block_id
                        )


# ---------------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------------

def load_and_validate_manifest(
    path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
    check_document_integrity: bool = True
) -> BenchmarkManifest:
    """
    Loads, deserializes, and validates a benchmark manifest through all three validation layers.

    1. Layer A: Validates raw JSON structure against the Draft 2020-12 schema.
    2. Instantiates strongly typed BenchmarkManifest data models.
    3. Layer B: Validates cross-field rules, ID uniqueness, and coverage consistency.
    4. Layer C (optional): Validates document existence, hashes, and evidence grounding.

    Args:
        path: Path to the benchmark manifest JSON file.
        base_dir: Base directory for resolving relative PDF paths (defaults to repository root).
        check_document_integrity: If True, executes Layer C document integrity checks.

    Returns:
        The validated BenchmarkManifest instance.

    Raises:
        BenchmarkValidationError: If validation fails in any layer.
    """
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Benchmark manifest file not found at: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        try:
            raw_data = json.load(f)
        except json.JSONDecodeError as exc:
            raise BenchmarkSchemaError(
                message=f"Invalid JSON syntax in manifest: {exc}",
                json_path="/",
                schema_path="/"
            ) from exc

    # Layer A: JSON Schema Validation
    validate_schema(raw_data)

    # Data Model Deserialization
    manifest = BenchmarkManifest.from_dict(raw_data)

    # Layer B: Python Cross-Field Validation
    validate_cross_field_rules(manifest)

    # Layer C: Document Integrity Validation
    if check_document_integrity:
        validate_document_integrity(manifest, base_dir=base_dir)

    return manifest

