"""
V2 Compliance Report Formatter and Exporter (Phase 2.5).
Formats DocumentComplianceReport objects into structured JSON dictionaries,
human-readable Markdown audit reports (no tables), and concise terminal summaries.
Decoupled from ML inference; supports complete evidence provenance and cross-platform execution.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from src.models.compliance import (
    ParentComplianceStatus,
    DocumentComplianceVerdict,
    ParentPointCompliance,
    DocumentComplianceReport,
)
from src.models.verification import (
    VerificationStatus,
    EvidenceEvaluation,
    SubRequirementVerificationResult,
)


class ComplianceReportFormatter:
    """
    Renders structured audit reports from DocumentComplianceReport instances.
    Provides JSON, Markdown, and Terminal output representations while preserving
    complete evidence provenance and avoiding markdown tables.
    """

    SCHEMA_VERSION = "2.0"

    def __init__(
        self,
        dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
        retrieval_method: str = "Hybrid (BM25 + MiniLM Dense + RRF)",
        top_k_candidates: int = 5,
    ):
        self.dense_model_name = dense_model_name
        self.nli_model_name = nli_model_name
        self.retrieval_method = retrieval_method
        self.top_k_candidates = top_k_candidates

    # --------------------------------------------------------------------------
    # 1. JSON Report Generation
    # --------------------------------------------------------------------------

    def to_json(
        self,
        report: DocumentComplianceReport,
        metadata_override: Optional[Dict[str, Any]] = None,
        timestamp_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Converts a DocumentComplianceReport into a complete, JSON-serializable dictionary.
        Preserves all evidence provenance, candidate rankings, and NLI scores.
        """
        ts = timestamp_override or datetime.now(timezone.utc).isoformat()

        target_page = getattr(report, "target_page", None)
        scope_str = f"single_page (page {target_page})" if target_page is not None else "entire_document"

        # Build Metadata
        meta: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "generated_at": ts,
            "reference_template": {
                "title": report.template_title,
                "path": str(report.template_path),
                "total_parent_points": report.total_parent_points,
                "total_sub_requirements": report.total_sub_requirements,
            },
            "target_document": {
                "filename": report.target_document_name,
                "path": str(report.target_document_path),
                "total_pages": report.total_pages,
                "target_page": target_page,
                "audit_scope": scope_str,
            },
            "engine": {
                "dense_model": self.dense_model_name,
                "nli_model": self.nli_model_name,
                "retrieval_method": self.retrieval_method,
                "top_k_candidates": self.top_k_candidates,
            },
        }
        if metadata_override:
            meta.update(metadata_override)

        # Build Executive Summary
        executive_summary = {
            "verdict": self._enum_val(report.verdict),
            "mandatory_coverage_pct": (
                round(report.mandatory_coverage_pct, 2)
                if report.mandatory_coverage_pct is not None
                else None
            ),
            "optional_coverage_pct": (
                round(report.optional_coverage_pct, 2)
                if report.optional_coverage_pct is not None
                else None
            ),
            "total_mandatory": report.total_mandatory,
            "total_optional": report.total_optional,
            "mandatory_fulfilled": report.mandatory_fulfilled,
            "optional_fulfilled": report.optional_fulfilled,
            "total_parent_points": report.total_parent_points,
            "total_sub_requirements": report.total_sub_requirements,
            "atomic_status_counts": {
                "fulfilled": report.fulfilled_count,
                "explicit_omission": report.explicit_omission_count,
                "contradicted": report.contradicted_count,
                "missing": report.missing_count,
                "needs_review": report.needs_review_count,
            },
            "unresolved_review_count": report.unresolved_count,
            "summary_text": report.summary,
        }

        # Parent Points List
        parent_results = [
            self._format_parent_point_dict(p) for p in report.parent_points
        ]

        # Flat Sub-requirements List
        atomic_results = []
        for p in report.parent_points:
            for sr in p.sub_results:
                atomic_results.append(self._format_sub_requirement_dict(sr))

        # Human Review Queue
        review_queue = [
            self._format_human_review_dict(sr) for sr in report.items_requiring_review
        ]

        return {
            "metadata": meta,
            "executive_summary": executive_summary,
            "human_review_queue": review_queue,
            "parent_results": parent_results,
            "atomic_results": atomic_results,
        }

    def _format_evaluation_dict(self, ev: EvidenceEvaluation) -> Dict[str, Any]:
        """Serializes an EvidenceEvaluation object with full candidate provenance."""
        cand = ev.candidate
        block_type_val = (
            cand.block_type.value if hasattr(cand.block_type, "value") else str(cand.block_type)
        )
        return {
            "block_id": cand.block_id,
            "page_number": cand.page_number,
            "section_heading": cand.section_heading,
            "block_type": block_type_val,
            "exact_source_text": ev.premise_text or cand.text,
            "rrf_rank": cand.rrf_rank,
            "rrf_score": round(cand.rrf_score, 6),
            "bm25_score": round(cand.bm25_score, 4),
            "bm25_rank": cand.bm25_rank,
            "dense_score": round(cand.dense_score, 4),
            "dense_rank": cand.dense_rank,
            "nli_predicted_label": ev.predicted_label,
            "entailment_prob": round(ev.entailment_prob, 4),
            "contradiction_prob": round(ev.contradiction_prob, 4),
            "neutral_prob": round(ev.neutral_prob, 4),
            "is_explicit_omission": cand.is_explicit_omission,
            "has_numeric_content": ev.has_numeric_content,
            "numeric_tokens": list(ev.numeric_tokens),
            "hypothesis_text": ev.hypothesis_text,
        }

    def _format_sub_requirement_dict(
        self, sr: SubRequirementVerificationResult
    ) -> Dict[str, Any]:
        """Serializes an atomic verification result with complete provenance."""
        ev_type_val = (
            sr.expected_evidence_type.value
            if hasattr(sr.expected_evidence_type, "value")
            else str(sr.expected_evidence_type)
        )
        return {
            "sub_id": sr.sub_id,
            "parent_id": sr.parent_point_id,
            "original_reference_text": sr.original_text,
            "claim_text": sr.claim_text,
            "expected_evidence_type": ev_type_val,
            "is_mandatory": sr.is_mandatory,
            "verification_status": self._enum_val(sr.status),
            "confidence": round(sr.confidence, 4),
            "rationale": sr.rationale,
            "needs_review": sr.needs_review,
            "source_page": sr.source_page,
            "section_name": sr.section_name,
            "topic_label": sr.topic_label,
            "best_evidence": (
                self._format_evaluation_dict(sr.best_evidence)
                if sr.best_evidence
                else None
            ),
            "all_evaluated_candidates": [
                self._format_evaluation_dict(e) for e in sr.all_evaluations
            ],
        }

    def _format_parent_point_dict(self, p: ParentPointCompliance) -> Dict[str, Any]:
        """Serializes a ParentPointCompliance object."""
        return {
            "point_id": p.point_id,
            "section_name": p.section_name,
            "topic": p.topic_label,
            "description": p.description,
            "raw_text": p.raw_text,
            "parent_compliance_status": self._enum_val(p.status),
            "mandatory_fulfilled": p.mandatory_fulfilled,
            "mandatory_total": p.mandatory_total,
            "optional_fulfilled": p.optional_fulfilled,
            "optional_total": p.optional_total,
            "findings": list(p.findings),
            "source_page": p.source_page,
            "sub_requirement_ids": [sr.sub_id for sr in p.sub_results],
        }

    def _format_human_review_dict(
        self, sr: SubRequirementVerificationResult
    ) -> Dict[str, Any]:
        """Extracts priority human-review details with positive and conflicting evidence."""
        positive_evidence = None
        conflicting_evidence = None

        if sr.best_evidence:
            if sr.best_evidence.entailment_prob >= 0.50:
                positive_evidence = self._format_evaluation_dict(sr.best_evidence)

        # Look for conflicting or disclaimer candidate
        for e in sr.all_evaluations:
            if e.candidate.is_explicit_omission or e.contradiction_prob >= 0.40:
                conflicting_evidence = self._format_evaluation_dict(e)
                break

        relevant_pages = sorted(
            list(
                {
                    e.candidate.page_number
                    for e in sr.all_evaluations
                    if e.candidate.page_number
                }
            )
        )

        return {
            "sub_id": sr.sub_id,
            "parent_point_id": sr.parent_point_id,
            "claim_text": sr.claim_text,
            "is_mandatory": sr.is_mandatory,
            "rationale": sr.rationale,
            "relevant_source_pages": relevant_pages,
            "positive_evidence": positive_evidence,
            "conflicting_evidence": conflicting_evidence,
            "all_evaluated_candidates": [
                self._format_evaluation_dict(e) for e in sr.all_evaluations
            ],
        }

    # --------------------------------------------------------------------------
    # 2. Markdown Report Generation (NO MARKDOWN TABLES)
    # --------------------------------------------------------------------------

    def to_markdown(
        self,
        report: DocumentComplianceReport,
        timestamp_override: Optional[str] = None,
    ) -> str:
        """
        Generates a clean, professional Markdown compliance audit report.
        Strictly avoids Markdown tables, using headings, lists, and quotes instead.
        Safely escapes text extracted from target documents.
        """
        ts = timestamp_override or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        target_page = getattr(report, "target_page", None)
        if target_page is not None:
            scope_desc = f"Single Page (Page {target_page} of {report.total_pages or 'N/A'})"
        else:
            scope_desc = f"Entire Document ({report.total_pages or 'N/A'} pages)"

        lines: List[str] = []

        # 1. Title and Audit Metadata
        lines.append("# SRS & Technical Document Compliance Audit Report\n")
        lines.append("## Audit Metadata\n")
        lines.append(f"- **Reference Standard Template**: `{self._safe_md(report.template_title)}`")
        lines.append(f"- **Reference Template Path**: `{self._safe_md(str(report.template_path))}`")
        lines.append(f"- **Target Document**: `{self._safe_md(report.target_document_name)}`")
        lines.append(f"- **Target Document Path**: `{self._safe_md(str(report.target_document_path))}`")
        lines.append(f"- **Audit Scope**: {scope_desc}")
        lines.append(f"- **Audit Generated At**: {ts}")
        lines.append(f"- **Verification Engine**: V2 Hybrid Engine ({self.retrieval_method} + {self.nli_model_name})\n")

        # 2. Executive Summary
        lines.append("## 1. Executive Summary\n")
        lines.append(self._safe_md(report.summary))
        lines.append("")

        # 3. Compliance Verdict and Coverage
        lines.append("## 2. Compliance Verdict and Coverage\n")
        v_str = self._enum_val(report.verdict)
        lines.append(f"- **Overall Compliance Verdict**: **{v_str}**")

        mand_cov_str = (
            f"**{report.mandatory_coverage_pct:.1f}%** ({report.mandatory_fulfilled} of {report.total_mandatory} mandatory criteria fulfilled)"
            if report.mandatory_coverage_pct is not None
            else "N/A"
        )
        lines.append(f"- **Mandatory Requirement Coverage**: {mand_cov_str}")

        opt_cov_str = (
            f"**{report.optional_coverage_pct:.1f}%** ({report.optional_fulfilled} of {report.total_optional} optional recommendations fulfilled)"
            if report.optional_coverage_pct is not None
            else "N/A (No optional recommendations defined)"
        )
        lines.append(f"- **Optional Recommendation Coverage**: {opt_cov_str}")
        lines.append(f"- **Total Reference Points Evaluated**: {report.total_parent_points}")
        lines.append(f"- **Total Atomic Sub-Requirements**: {report.total_sub_requirements}\n")

        # 4. Atomic Status Counts
        lines.append("## 3. Atomic Verification Status Breakdown\n")
        lines.append(f"- **FULFILLED**: {report.fulfilled_count} sub-requirement(s) logically entailed by document evidence")
        lines.append(f"- **EXPLICIT_OMISSION**: {report.explicit_omission_count} sub-requirement(s) explicitly disclaimed or omitted in target text")
        lines.append(f"- **CONTRADICTED**: {report.contradicted_count} sub-requirement(s) directly contradicted by target text")
        lines.append(f"- **MISSING**: {report.missing_count} sub-requirement(s) unmentioned with no relevant evidence")
        lines.append(f"- **NEEDS_REVIEW**: {report.needs_review_count} sub-requirement(s) requiring human review due to ambiguous or conflicting evidence\n")

        # 5. Priority Human-Review Items
        lines.append("## 4. Priority Human-Review Items\n")
        if not report.items_requiring_review:
            lines.append("No unresolved items require human review. All atomic criteria were deterministically verified.\n")
        else:
            lines.append(
                f"**Attention**: {len(report.items_requiring_review)} atomic item(s) require human auditor inspection before a definitive verdict can be finalized.\n"
            )
            for item in report.items_requiring_review:
                lines.append(f"### Review Required: [{item.sub_id}] (Parent: {item.parent_point_id})\n")
                lines.append(f"- **Claim**: {self._safe_md(item.claim_text)}")
                lines.append(f"- **Mandatory**: {'Yes' if item.is_mandatory else 'No'}")
                lines.append(f"- **Review Reason**: {self._safe_md(item.rationale)}")
                if item.source_page:
                    lines.append(f"- **Source Reference Page**: Page {item.source_page}")

                # Display positive vs conflicting evidence excerpts
                for ev in item.all_evaluations:
                    cand = ev.candidate
                    label = ev.predicted_label.upper()
                    om_flag = " [EXPLICIT OMISSION]" if cand.is_explicit_omission else ""
                    lines.append(f"\n  Candidate Block `{cand.block_id}` (Page {cand.page_number}){om_flag}:")
                    lines.append(f"  - Predicted Label: `{label}` (Entailment: {ev.entailment_prob:.2f}, Contradiction: {ev.contradiction_prob:.2f}, Neutral: {ev.neutral_prob:.2f})")
                    lines.append(f"  - Text Excerpt:\n    > {self._safe_md(ev.premise_text)}")
                lines.append("")

        # 6. Parent-Level Compliance Findings
        lines.append("## 5. Parent-Level Compliance Findings\n")
        for p in report.parent_points:
            p_status = self._enum_val(p.status)
            topic_str = f" — {p.topic_label}" if p.topic_label else ""
            lines.append(f"### Parent Point {p.point_id}: {self._safe_md(p.section_name)}{topic_str}\n")
            lines.append(f"- **Status**: **{p_status}**")
            lines.append(f"- **Mandatory Criteria**: {p.mandatory_fulfilled} of {p.mandatory_total} fulfilled")
            if p.optional_total > 0:
                lines.append(f"- **Optional Recommendations**: {p.optional_fulfilled} of {p.optional_total} fulfilled")
            if p.raw_text:
                lines.append(f"- **Requirement Statement**:\n  > {self._safe_md(p.raw_text)}")

            lines.append("- **Findings**:")
            for f in p.findings:
                lines.append(f"  - {self._safe_md(f)}")
            lines.append("")

        # 7. Individual Sub-Requirement Results
        lines.append("## 6. Detailed Atomic Sub-Requirement Verifications\n")
        for p in report.parent_points:
            for sr in p.sub_results:
                status_str = self._enum_val(sr.status)
                mand_tag = "Mandatory" if sr.is_mandatory else "Optional"
                lines.append(f"### [{status_str}] {sr.sub_id} ({mand_tag})\n")
                lines.append(f"- **Parent Point**: {sr.parent_point_id} ({self._safe_md(sr.section_name or '')})")
                lines.append(f"- **Verifiable Claim**: {self._safe_md(sr.claim_text)}")
                if sr.original_text:
                    lines.append(f"- **Source Reference Text**:\n  > {self._safe_md(sr.original_text)}")
                ev_type_val = (
                    sr.expected_evidence_type.value
                    if hasattr(sr.expected_evidence_type, "value")
                    else str(sr.expected_evidence_type)
                )
                lines.append(f"- **Expected Evidence Form**: `{ev_type_val}`")
                lines.append(f"- **Confidence**: `{sr.confidence:.2f}`")
                lines.append(f"- **Verification Rationale**: {self._safe_md(sr.rationale)}")

                if sr.best_evidence:
                    b_cand = sr.best_evidence.candidate
                    lines.append(f"- **Primary Evidence Block**: `{b_cand.block_id}` (Page {b_cand.page_number}, Section: {self._safe_md(b_cand.section_heading or 'None')})")
                    lines.append(f"  - Excerpt:\n    > {self._safe_md(sr.best_evidence.premise_text)}")
                    lines.append(
                        f"  - Scores: Entailment = `{sr.best_evidence.entailment_prob:.2f}`, "
                        f"Contradiction = `{sr.best_evidence.contradiction_prob:.2f}`, "
                        f"Neutral = `{sr.best_evidence.neutral_prob:.2f}`"
                    )
                    if sr.best_evidence.has_numeric_content:
                        tokens = ", ".join(sr.best_evidence.numeric_tokens)
                        lines.append(f"  - Measured Numeric Tokens: `{tokens}`")
                else:
                    lines.append("- **Primary Evidence Block**: None (No candidate met the verification threshold)")
                lines.append("")

        # 8. Explicit Omissions and Scope Disclaimers
        lines.append("## 7. Explicit Omissions and Scope Disclaimers\n")
        omissions = [
            sr for p in report.parent_points for sr in p.sub_results
            if sr.status == VerificationStatus.EXPLICIT_OMISSION
        ]
        if not omissions:
            lines.append("No explicit omissions or scope disclaimers were recorded.\n")
        else:
            lines.append("The target document explicitly disclaimed or stated as out-of-scope the following requirements:\n")
            for om in omissions:
                lines.append(f"- **[{om.sub_id}]** (Parent {om.parent_point_id}): {self._safe_md(om.claim_text)}")
                if om.best_evidence:
                    b_cand = om.best_evidence.candidate
                    lines.append(f"  - Disclaiming Block: `{b_cand.block_id}` (Page {b_cand.page_number})")
                    lines.append(f"  - Target Text:\n    > {self._safe_md(om.best_evidence.premise_text)}")
                    lines.append(f"  - Contradiction/Exclusion Probability: `{om.best_evidence.contradiction_prob:.2f}`")
            lines.append("")

        # 9. Contradictions
        lines.append("## 8. Contradictions\n")
        contradictions = [
            sr for p in report.parent_points for sr in p.sub_results
            if sr.status == VerificationStatus.CONTRADICTED
        ]
        if not contradictions:
            lines.append("No direct contradictions detected between target document claims and reference requirements.\n")
        else:
            lines.append("The following reference requirements were directly contradicted by statements in the target document:\n")
            for contra in contradictions:
                lines.append(f"- **[{contra.sub_id}]** (Parent {contra.parent_point_id}): {self._safe_md(contra.claim_text)}")
                if contra.best_evidence:
                    b_cand = contra.best_evidence.candidate
                    lines.append(f"  - Contradicting Block: `{b_cand.block_id}` (Page {b_cand.page_number})")
                    lines.append(f"  - Target Text:\n    > {self._safe_md(contra.best_evidence.premise_text)}")
                    lines.append(f"  - Contradiction Probability: `{contra.best_evidence.contradiction_prob:.2f}`")
            lines.append("")

        # 10. Evidence Traceability Appendix
        lines.append("## 9. Evidence Traceability Appendix\n")
        lines.append("Mapping of verified criteria to target document structural blocks and retrieval provenance:\n")
        for p in report.parent_points:
            for sr in p.sub_results:
                if sr.best_evidence:
                    c = sr.best_evidence.candidate
                    lines.append(
                        f"- **{sr.sub_id}** -> Block `{c.block_id}` | Page {c.page_number} | "
                        f"Type `{c.block_type.value if hasattr(c.block_type, 'value') else c.block_type}` | "
                        f"RRF Rank #{c.rrf_rank} (Score: {c.rrf_score:.4f}) | "
                        f"Status: `{self._enum_val(sr.status)}`"
                    )
                else:
                    lines.append(f"- **{sr.sub_id}** -> None (No evidence retrieved) | Status: `{self._enum_val(sr.status)}`")
        lines.append("")

        return "\n".join(lines)

    # --------------------------------------------------------------------------
    # 3. Terminal Summary Output
    # --------------------------------------------------------------------------

    def to_terminal(
        self,
        report: DocumentComplianceReport,
        json_path: Optional[str] = None,
        md_path: Optional[str] = None,
    ) -> str:
        """
        Generates a concise, Windows-safe terminal summary of compliance results.
        Excludes verbose candidate evaluations to keep CLI output clean and readable.
        """
        target_page = getattr(report, "target_page", None)
        if target_page is not None:
            scope_str = f"Single Page (Page {target_page} of {report.total_pages or 'N/A'})"
        else:
            scope_str = f"Entire Document ({report.total_pages or 'N/A'} page(s))"

        # Count parent points by status
        p_counts = {
            "FULLY_COMPLIANT": 0,
            "PARTIALLY_COMPLIANT": 0,
            "NON_COMPLIANT": 0,
            "NEEDS_REVIEW": 0,
        }
        for p in report.parent_points:
            st = self._enum_val(p.status)
            if st in p_counts:
                p_counts[st] += 1
            else:
                p_counts[st] = 1

        lines: List[str] = []
        lines.append("=" * 80)
        lines.append("                    V2 COMPLIANCE AUDIT REPORT SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Target Document    : {report.target_document_name}")
        lines.append(f"Reference Template : {report.template_title}")
        lines.append(f"Audit Scope        : {scope_str}")
        lines.append("-" * 80)

        v_val = self._enum_val(report.verdict)
        lines.append(f"Overall Verdict    : {v_val}")

        m_cov_str = (
            f"{report.mandatory_coverage_pct:.1f}% ({report.mandatory_fulfilled}/{report.total_mandatory})"
            if report.mandatory_coverage_pct is not None
            else "N/A"
        )
        lines.append(f"Mandatory Coverage : {m_cov_str}")

        o_cov_str = (
            f"{report.optional_coverage_pct:.1f}% ({report.optional_fulfilled}/{report.total_optional})"
            if report.optional_coverage_pct is not None
            else "N/A"
        )
        lines.append(f"Optional Coverage  : {o_cov_str}")
        lines.append("-" * 80)

        lines.append("Parent Requirement Points Breakdown:")
        lines.append(f"  - FULLY_COMPLIANT     : {p_counts.get('FULLY_COMPLIANT', 0)}")
        lines.append(f"  - PARTIALLY_COMPLIANT : {p_counts.get('PARTIALLY_COMPLIANT', 0)}")
        lines.append(f"  - NON_COMPLIANT       : {p_counts.get('NON_COMPLIANT', 0)}")
        lines.append(f"  - NEEDS_REVIEW        : {p_counts.get('NEEDS_REVIEW', 0)}")
        lines.append(f"  - TOTAL PARENT POINTS : {report.total_parent_points}")
        lines.append("")

        lines.append("Atomic Sub-Requirement Breakdown:")
        lines.append(f"  - FULFILLED           : {report.fulfilled_count}")
        lines.append(f"  - EXPLICIT_OMISSION   : {report.explicit_omission_count}")
        lines.append(f"  - CONTRADICTED        : {report.contradicted_count}")
        lines.append(f"  - MISSING             : {report.missing_count}")
        lines.append(f"  - NEEDS_REVIEW        : {report.needs_review_count}")
        lines.append(f"  - TOTAL SUB-REQS      : {report.total_sub_requirements}")
        lines.append("-" * 80)

        if report.items_requiring_review:
            lines.append("Items Requiring Human Review:")
            for item in report.items_requiring_review:
                lines.append(f"  * [{item.sub_id}] ({item.parent_point_id}): {item.rationale}")
        else:
            lines.append("Human Review Queue : None. All items resolved deterministically.")

        if json_path or md_path:
            lines.append("-" * 80)
            if json_path:
                lines.append(f"Exported JSON Report     : {json_path}")
            if md_path:
                lines.append(f"Exported Markdown Report : {md_path}")

        lines.append("=" * 80)
        return "\n".join(lines)

    # --------------------------------------------------------------------------
    # 4. File Export Helpers
    # --------------------------------------------------------------------------

    def export_json(
        self,
        report: DocumentComplianceReport,
        file_path: Union[str, Path],
        indent: int = 2,
        metadata_override: Optional[Dict[str, Any]] = None,
        timestamp_override: Optional[str] = None,
    ) -> Path:
        """Exports the formatted JSON report to a file using UTF-8 encoding."""
        out_path = Path(file_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_json(
            report,
            metadata_override=metadata_override,
            timestamp_override=timestamp_override,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return out_path

    def export_markdown(
        self,
        report: DocumentComplianceReport,
        file_path: Union[str, Path],
        timestamp_override: Optional[str] = None,
    ) -> Path:
        """Exports the formatted Markdown report to a file using UTF-8 encoding."""
        out_path = Path(file_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        content = self.to_markdown(report, timestamp_override=timestamp_override)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        return out_path

    # --------------------------------------------------------------------------
    # 5. Helpers
    # --------------------------------------------------------------------------

    @staticmethod
    def _enum_val(val: Any) -> str:
        """Returns string representation of an enum or standard value."""
        if hasattr(val, "value"):
            return str(val.value)
        return str(val)

    @staticmethod
    def _safe_md(text: Optional[str]) -> str:
        """
        Escapes special characters in text extracted from target documents
        to prevent breaking Markdown layout or formatting.
        Preserves Unicode characters (°C, ±, —).
        """
        if not text:
            return ""
        # Normalize newlines and carriage returns
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        # Replace consecutive backticks that might trigger code fence breaks
        t = re.sub(r"`{2,}", "'", t)
        # Escape stray backticks that could break inline code spans
        t = t.replace("`", "\\`")
        return t

