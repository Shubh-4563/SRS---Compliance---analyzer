"""
V2 Verification Data Models for NLI-based Evidence Verification (Phase 2.3).
Captures NLI entailment/contradiction scores, candidate evidence evaluations,
explicit omissions, and sub-requirement verification states with complete provenance.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

from src.models.requirement import EvidenceType, SubRequirement
from src.models.retrieval import CandidateEvidence


class VerificationStatus(str, Enum):
    """Compliance verification status for an atomic sub-requirement."""
    FULFILLED = "FULFILLED"                    # Sufficient evidence entails the requirement
    CONTRADICTED = "CONTRADICTED"              # Evidence directly contradicts the requirement
    EXPLICIT_OMISSION = "EXPLICIT_OMISSION"    # Document explicitly disclaims or omits this requirement
    MISSING = "MISSING"                        # No evidence entails the requirement (all candidates neutral/irrelevant)
    NEEDS_REVIEW = "NEEDS_REVIEW"              # Conflicting signals, ambiguous scores, or missing numeric values


@dataclass
class EvidenceEvaluation:
    """
    NLI evaluation of a single candidate evidence block against a sub-requirement claim.
    """
    candidate: CandidateEvidence
    premise_text: str                          # Exact text passed as premise to NLI
    hypothesis_text: str                       # Exact text passed as hypothesis to NLI
    entailment_prob: float                     # Softmax probability for entailment
    contradiction_prob: float                  # Softmax probability for contradiction
    neutral_prob: float                        # Softmax probability for neutral
    predicted_label: str                       # "entailment", "contradiction", or "neutral"
    has_numeric_content: bool = False          # True if candidate contains numeric values/units
    numeric_tokens: List[str] = field(default_factory=list) # Extracted numeric tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.candidate.block_id,
            "page_number": self.candidate.page_number,
            "section_heading": self.candidate.section_heading,
            "predicted_label": self.predicted_label,
            "entailment_prob": round(self.entailment_prob, 4),
            "contradiction_prob": round(self.contradiction_prob, 4),
            "neutral_prob": round(self.neutral_prob, 4),
            "is_explicit_omission": self.candidate.is_explicit_omission,
            "has_numeric_content": self.has_numeric_content,
            "numeric_tokens": self.numeric_tokens,
            "evidence_text": self.premise_text,
            "hypothesis_text": self.hypothesis_text,
        }


@dataclass
class SubRequirementVerificationResult:
    """
    Verification outcome for an atomic sub-requirement based on multi-candidate NLI analysis.
    Preserves exact provenance, rationale, and all candidate evaluations.
    """
    sub_id: str
    parent_point_id: str
    claim_text: str
    expected_evidence_type: EvidenceType
    status: VerificationStatus
    confidence: float                          # Confidence of the decision (0.0 to 1.0)
    is_mandatory: bool = True                  # Inherited from sub-requirement
    best_evidence: Optional[EvidenceEvaluation] = None
    all_evaluations: List[EvidenceEvaluation] = field(default_factory=list)
    rationale: str = ""
    needs_review: bool = False
    source_page: Optional[int] = None
    section_name: Optional[str] = None
    topic_label: Optional[str] = None
    original_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sub_id": self.sub_id,
            "parent_point_id": self.parent_point_id,
            "claim_text": self.claim_text,
            "expected_evidence_type": self.expected_evidence_type.value,
            "status": self.status.value,
            "confidence": round(self.confidence, 4),
            "is_mandatory": self.is_mandatory,
            "needs_review": self.needs_review,
            "rationale": self.rationale,
            "source_page": self.source_page,
            "section_name": self.section_name,
            "topic_label": self.topic_label,
            "original_text": self.original_text,
            "best_evidence": self.best_evidence.to_dict() if self.best_evidence else None,
            "evaluations_count": len(self.all_evaluations),
            "evaluations": [e.to_dict() for e in self.all_evaluations],
        }

