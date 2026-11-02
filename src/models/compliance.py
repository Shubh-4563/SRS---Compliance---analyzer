"""
V2 Coverage Aggregation and Compliance Scoring Data Models (Phase 2.4).
Defines parent-level compliance structures, document-level scoring rollups,
and audit verdicts with complete traceability.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

from src.models.verification import (
    VerificationStatus,
    SubRequirementVerificationResult,
)


class ParentComplianceStatus(str, Enum):
    """Compliance verification status for a parent template requirement point."""
    FULLY_COMPLIANT = "FULLY_COMPLIANT"          # All mandatory sub-requirements fulfilled
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"  # Some mandatory sub-requirements fulfilled, others incomplete
    NON_COMPLIANT = "NON_COMPLIANT"              # No mandatory sub-requirements fulfilled (all missing/omitted/contradicted)
    NEEDS_REVIEW = "NEEDS_REVIEW"                # At least one mandatory sub-requirement requires human review


class DocumentComplianceVerdict(str, Enum):
    """Overall document-level compliance assessment verdict."""
    COMPLIANT = "COMPLIANT"                      # All mandatory requirements fulfilled, no contradictions or reviews
    CONDITIONALLY_COMPLIANT = "CONDITIONALLY_COMPLIANT" # At least one mandatory requirement requires human review
    NON_COMPLIANT = "NON_COMPLIANT"              # Mandatory coverage below 100% with no unresolved review items


@dataclass
class ParentPointCompliance:
    """
    Compliance assessment for a single parent template point, synthesized
    from its constituent atomic sub-requirement verification results.
    """
    point_id: str
    section_name: str
    topic_label: Optional[str]
    description: str
    raw_text: str
    status: ParentComplianceStatus
    mandatory_fulfilled: int
    mandatory_total: int
    optional_fulfilled: int
    optional_total: int
    sub_results: List[SubRequirementVerificationResult] = field(default_factory=list)
    fulfilled_count: int = 0
    missing_count: int = 0
    explicit_omission_count: int = 0
    contradiction_count: int = 0
    needs_review_count: int = 0
    findings: List[str] = field(default_factory=list)
    source_page: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "section_name": self.section_name,
            "topic_label": self.topic_label,
            "description": self.description,
            "raw_text": self.raw_text,
            "status": self.status.value,
            "mandatory_fulfilled": self.mandatory_fulfilled,
            "mandatory_total": self.mandatory_total,
            "optional_fulfilled": self.optional_fulfilled,
            "optional_total": self.optional_total,
            "fulfilled_count": self.fulfilled_count,
            "missing_count": self.missing_count,
            "explicit_omission_count": self.explicit_omission_count,
            "contradiction_count": self.contradiction_count,
            "needs_review_count": self.needs_review_count,
            "findings": self.findings,
            "source_page": self.source_page,
            "sub_results": [r.to_dict() for r in self.sub_results],
        }


@dataclass
class DocumentComplianceReport:
    """
    Comprehensive compliance audit report for a target document against a reference template.
    Provides hierarchical rollup from atomic sub-requirements up to document-level verdicts.
    """
    target_document_path: str
    target_document_name: str
    template_title: str
    template_path: str
    total_parent_points: int
    total_sub_requirements: int
    parent_points: List[ParentPointCompliance] = field(default_factory=list)
    mandatory_coverage_pct: Optional[float] = None
    optional_coverage_pct: Optional[float] = None
    total_mandatory: int = 0
    total_optional: int = 0
    mandatory_fulfilled: int = 0
    optional_fulfilled: int = 0
    fulfilled_count: int = 0
    explicit_omission_count: int = 0
    contradicted_count: int = 0
    missing_count: int = 0
    needs_review_count: int = 0
    unresolved_count: int = 0
    verdict: DocumentComplianceVerdict = DocumentComplianceVerdict.NON_COMPLIANT
    summary: str = ""
    items_requiring_review: List[SubRequirementVerificationResult] = field(default_factory=list)
    total_pages: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_document_path": self.target_document_path,
            "target_document_name": self.target_document_name,
            "template_title": self.template_title,
            "template_path": self.template_path,
            "total_pages": self.total_pages,
            "total_parent_points": self.total_parent_points,
            "total_sub_requirements": self.total_sub_requirements,
            "mandatory_coverage_pct": (
                round(self.mandatory_coverage_pct, 2)
                if self.mandatory_coverage_pct is not None
                else None
            ),
            "optional_coverage_pct": (
                round(self.optional_coverage_pct, 2)
                if self.optional_coverage_pct is not None
                else None
            ),
            "total_mandatory": self.total_mandatory,
            "total_optional": self.total_optional,
            "mandatory_fulfilled": self.mandatory_fulfilled,
            "optional_fulfilled": self.optional_fulfilled,
            "fulfilled_count": self.fulfilled_count,
            "explicit_omission_count": self.explicit_omission_count,
            "contradicted_count": self.contradicted_count,
            "missing_count": self.missing_count,
            "needs_review_count": self.needs_review_count,
            "unresolved_count": self.unresolved_count,
            "verdict": self.verdict.value,
            "summary": self.summary,
            "parent_points": [p.to_dict() for p in self.parent_points],
            "items_requiring_review": [r.to_dict() for r in self.items_requiring_review],
        }

