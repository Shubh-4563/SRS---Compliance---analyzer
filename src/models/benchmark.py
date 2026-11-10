"""
V2 Declarative Benchmark Manifest Data Models (Phase 2.6.2.2).
Provides strongly typed dataclasses and enumerations for benchmark metadata,
ground-truth annotations, parent and atomic requirements, and expected compliance results.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

from src.models.verification import VerificationStatus
from src.models.compliance import ParentComplianceStatus, DocumentComplianceVerdict


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class BenchmarkEvidencePresence(str, Enum):
    """Categorization of evidence presence in the target document."""
    EXISTS = "EXISTS"                      # Positive or contradicting evidence exists
    EXPLICIT_OMISSION = "EXPLICIT_OMISSION"  # Explicit disclaimer or omission statement exists
    ABSENT = "ABSENT"                      # Requirement is entirely unmentioned across target document


class BenchmarkEvidenceType(str, Enum):
    """Semantic type of evidence expected for a sub-requirement in benchmark evaluation."""
    FACTUAL_STATEMENT = "FACTUAL_STATEMENT"
    NUMERIC_TARGET = "NUMERIC_TARGET"
    SAFETY_CONSTRAINT = "SAFETY_CONSTRAINT"
    INTERFACE_SPECIFICATION = "INTERFACE_SPECIFICATION"


# Aliases for consistent naming and interoperability with existing core models
BenchmarkVerificationStatus = VerificationStatus
BenchmarkParentComplianceStatus = ParentComplianceStatus
BenchmarkDocumentComplianceVerdict = DocumentComplianceVerdict


# ---------------------------------------------------------------------------
# Document and Metadata Models
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkDocumentMetadata:
    """Metadata and integrity hash for reference or target PDF documents."""
    path: str
    filename: str
    sha256: str
    total_pages: Optional[int] = None  # Populated for target documents

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "path": self.path,
            "filename": self.filename,
            "sha256": self.sha256,
        }
        if self.total_pages is not None:
            data["total_pages"] = self.total_pages
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkDocumentMetadata":
        return cls(
            path=data["path"],
            filename=data["filename"],
            sha256=data["sha256"],
            total_pages=data.get("total_pages"),
        )


@dataclass
class BenchmarkMetadata:
    """Metadata describing a benchmark test pair."""
    pair_id: str
    domain: str
    title: str
    description: str
    reference_document: BenchmarkDocumentMetadata
    target_document: BenchmarkDocumentMetadata
    tags: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "pair_id": self.pair_id,
            "domain": self.domain,
            "title": self.title,
            "description": self.description,
            "reference_document": self.reference_document.to_dict(),
            "target_document": self.target_document.to_dict(),
            "tags": list(self.tags),
        }
        if self.created_at is not None:
            data["created_at"] = self.created_at
        if self.updated_at is not None:
            data["updated_at"] = self.updated_at
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkMetadata":
        return cls(
            pair_id=data["pair_id"],
            domain=data["domain"],
            title=data["title"],
            description=data["description"],
            reference_document=BenchmarkDocumentMetadata.from_dict(data["reference_document"]),
            target_document=BenchmarkDocumentMetadata.from_dict(data["target_document"]),
            tags=data["tags"],
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


# ---------------------------------------------------------------------------
# Ground-Truth Annotation Models
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkEvidenceAnnotation:
    """Annotation identifying an evidence block in the parsed target document."""
    block_id: str
    page: int
    excerpt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "block_id": self.block_id,
            "page": self.page,
        }
        if self.excerpt is not None:
            data["excerpt"] = self.excerpt
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkEvidenceAnnotation":
        return cls(
            block_id=data["block_id"],
            page=data["page"],
            excerpt=data.get("excerpt"),
        )


@dataclass
class BenchmarkRetrievalGroundTruth:
    """Retrieval ground truth indicating expected evidence blocks or absence."""
    evidence_presence: BenchmarkEvidencePresence
    expected_evidence: List[BenchmarkEvidenceAnnotation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_presence": self.evidence_presence.value,
            "expected_evidence": [e.to_dict() for e in self.expected_evidence],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkRetrievalGroundTruth":
        return cls(
            evidence_presence=BenchmarkEvidencePresence(data["evidence_presence"]),
            expected_evidence=[
                BenchmarkEvidenceAnnotation.from_dict(e) for e in data["expected_evidence"]
            ],
        )


@dataclass
class BenchmarkVerificationGroundTruth:
    """Verification ground truth defining expected atomic compliance status."""
    expected_status: BenchmarkVerificationStatus
    expected_rationale_notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_status": self.expected_status.value,
            "expected_rationale_notes": self.expected_rationale_notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkVerificationGroundTruth":
        return cls(
            expected_status=BenchmarkVerificationStatus(data["expected_status"]),
            expected_rationale_notes=data["expected_rationale_notes"],
        )


@dataclass
class BenchmarkGroundTruth:
    """Composite ground-truth container covering retrieval and verification."""
    retrieval: BenchmarkRetrievalGroundTruth
    verification: BenchmarkVerificationGroundTruth

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieval": self.retrieval.to_dict(),
            "verification": self.verification.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkGroundTruth":
        return cls(
            retrieval=BenchmarkRetrievalGroundTruth.from_dict(data["retrieval"]),
            verification=BenchmarkVerificationGroundTruth.from_dict(data["verification"]),
        )


# ---------------------------------------------------------------------------
# Requirement Models
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkSubRequirement:
    """Atomic verifiable sub-requirement with associated ground-truth annotations."""
    sub_id: str
    claim_text: str
    expected_evidence_type: BenchmarkEvidenceType
    is_mandatory: bool
    decomposition_rationale: str
    ground_truth: BenchmarkGroundTruth

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sub_id": self.sub_id,
            "claim_text": self.claim_text,
            "expected_evidence_type": self.expected_evidence_type.value,
            "is_mandatory": self.is_mandatory,
            "decomposition_rationale": self.decomposition_rationale,
            "ground_truth": self.ground_truth.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkSubRequirement":
        return cls(
            sub_id=data["sub_id"],
            claim_text=data["claim_text"],
            expected_evidence_type=BenchmarkEvidenceType(data["expected_evidence_type"]),
            is_mandatory=data["is_mandatory"],
            decomposition_rationale=data["decomposition_rationale"],
            ground_truth=BenchmarkGroundTruth.from_dict(data["ground_truth"]),
        )


@dataclass
class BenchmarkParentRequirement:
    """Parent template section requirement containing atomic sub-requirements."""
    parent_id: str
    section_name: str
    topic_label: str
    description: str
    is_mandatory: bool
    expected_compliance_status: BenchmarkParentComplianceStatus
    sub_requirements: List[BenchmarkSubRequirement] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "section_name": self.section_name,
            "topic_label": self.topic_label,
            "description": self.description,
            "is_mandatory": self.is_mandatory,
            "expected_compliance_status": self.expected_compliance_status.value,
            "sub_requirements": [sr.to_dict() for sr in self.sub_requirements],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkParentRequirement":
        return cls(
            parent_id=data["parent_id"],
            section_name=data["section_name"],
            topic_label=data["topic_label"],
            description=data["description"],
            is_mandatory=data["is_mandatory"],
            expected_compliance_status=BenchmarkParentComplianceStatus(data["expected_compliance_status"]),
            sub_requirements=[
                BenchmarkSubRequirement.from_dict(sr) for sr in data["sub_requirements"]
            ],
        )


# ---------------------------------------------------------------------------
# Compliance Statistics and Manifest Models
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkExpectedStatusCounts:
    """Breakdown of expected verification status counts across all atomic requirements."""
    fulfilled: int = 0
    explicit_omission: int = 0
    contradicted: int = 0
    missing: int = 0
    needs_review: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "FULFILLED": self.fulfilled,
            "EXPLICIT_OMISSION": self.explicit_omission,
            "CONTRADICTED": self.contradicted,
            "MISSING": self.missing,
            "NEEDS_REVIEW": self.needs_review,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkExpectedStatusCounts":
        return cls(
            fulfilled=data["FULFILLED"],
            explicit_omission=data["EXPLICIT_OMISSION"],
            contradicted=data["CONTRADICTED"],
            missing=data["MISSING"],
            needs_review=data["NEEDS_REVIEW"],
        )


@dataclass
class BenchmarkExpectedCompliance:
    """Document-level expected compliance statistics, coverage percentages, and verdict."""
    verdict: BenchmarkDocumentComplianceVerdict
    mandatory_coverage_pct: float
    optional_coverage_pct: Optional[float]
    mandatory_fulfilled: int
    mandatory_total: int
    optional_fulfilled: int
    optional_total: int
    expected_status_counts: BenchmarkExpectedStatusCounts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "mandatory_coverage_pct": self.mandatory_coverage_pct,
            "optional_coverage_pct": self.optional_coverage_pct,
            "mandatory_fulfilled": self.mandatory_fulfilled,
            "mandatory_total": self.mandatory_total,
            "optional_fulfilled": self.optional_fulfilled,
            "optional_total": self.optional_total,
            "expected_status_counts": self.expected_status_counts.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkExpectedCompliance":
        counts_data = data["expected_status_counts"]
        if isinstance(counts_data, BenchmarkExpectedStatusCounts):
            counts = counts_data
        else:
            counts = BenchmarkExpectedStatusCounts.from_dict(counts_data)

        return cls(
            verdict=BenchmarkDocumentComplianceVerdict(data["verdict"]),
            mandatory_coverage_pct=data["mandatory_coverage_pct"],
            optional_coverage_pct=data["optional_coverage_pct"],
            mandatory_fulfilled=data["mandatory_fulfilled"],
            mandatory_total=data["mandatory_total"],
            optional_fulfilled=data["optional_fulfilled"],
            optional_total=data["optional_total"],
            expected_status_counts=counts,
        )


@dataclass
class BenchmarkManifest:
    """Complete declarative benchmark pair manifest."""
    schema_version: str
    metadata: BenchmarkMetadata
    parent_requirements: List[BenchmarkParentRequirement]
    expected_document_compliance: BenchmarkExpectedCompliance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metadata": self.metadata.to_dict(),
            "parent_requirements": [p.to_dict() for p in self.parent_requirements],
            "expected_document_compliance": self.expected_document_compliance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkManifest":
        return cls(
            schema_version=data["schema_version"],
            metadata=BenchmarkMetadata.from_dict(data["metadata"]),
            parent_requirements=[
                BenchmarkParentRequirement.from_dict(p) for p in data["parent_requirements"]
            ],
            expected_document_compliance=BenchmarkExpectedCompliance.from_dict(
                data["expected_document_compliance"]
            ),
        )
