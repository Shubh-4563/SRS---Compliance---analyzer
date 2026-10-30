"""
V2 Core Data Models for Reference Templates and Atomic Requirements.
Designed for cross-platform execution (macOS & Windows) with clean traceability.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class EvidenceType(str, Enum):
    """Expected form of evidence required to satisfy an atomic sub-requirement."""
    FACTUAL_STATEMENT = "FACTUAL_STATEMENT"  # Declarative explanation, purpose, role
    NUMERIC_TARGET = "NUMERIC_TARGET"        # Specific numerical metric, tolerance, or capacity
    LIST_OF_ITEMS = "LIST_OF_ITEMS"          # Enumeration of items, equipment, or interfaces
    ORDERED_STEPS = "ORDERED_STEPS"          # Sequence of procedural or operational steps
    NEGATIVE_BOUNDARY = "NEGATIVE_BOUNDARY"  # Explicit scope boundary or exclusion


@dataclass
class SubRequirement:
    """
    An atomic, verifiable criterion decomposed from a parent template point.
    Preserves exact provenance back to the source template point.
    """
    sub_id: str                              # e.g. "B4.1", "B4.2", "A1.1"
    parent_point_id: str                     # e.g. "B4", "A1"
    claim_text: str                          # Declarative statement/hypothesis to verify
    expected_evidence_type: EvidenceType     # Expected nature of target evidence
    is_mandatory: bool = True                # Mandatory vs optional aspect
    source_page: Optional[int] = None        # Page number where parent requirement was defined
    section_name: Optional[str] = None       # Section context in template
    topic_label: Optional[str] = None        # Topic keyword from template
    original_text: str = ""                  # Full raw text of parent requirement
    decomposition_rationale: str = ""        # Why and how this sub-requirement was isolated
    is_decomposed: bool = True               # False if requirement could not be safely split
    needs_review: bool = False               # Flagged if decomposition was ambiguous

    def to_dict(self) -> Dict[str, Any]:
        """Convert sub-requirement to JSON-serializable dictionary."""
        return {
            "sub_id": self.sub_id,
            "parent_point_id": self.parent_point_id,
            "claim_text": self.claim_text,
            "expected_evidence_type": self.expected_evidence_type.value,
            "is_mandatory": self.is_mandatory,
            "source_page": self.source_page,
            "section_name": self.section_name,
            "topic_label": self.topic_label,
            "original_text": self.original_text,
            "decomposition_rationale": self.decomposition_rationale,
            "is_decomposed": self.is_decomposed,
            "needs_review": self.needs_review,
        }


@dataclass
class DecomposedTemplatePoint:
    """
    A reference-template point enriched with atomic sub-requirements.
    """
    point_id: str
    section_name: str
    topic_label: Optional[str]
    description: str
    raw_text: str
    source_page: Optional[int] = None
    sub_requirements: List[SubRequirement] = field(default_factory=list)
    is_safely_decomposed: bool = True
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "section_name": self.section_name,
            "topic_label": self.topic_label,
            "description": self.description,
            "raw_text": self.raw_text,
            "source_page": self.source_page,
            "is_safely_decomposed": self.is_safely_decomposed,
            "notes": self.notes,
            "sub_requirements": [s.to_dict() for s in self.sub_requirements],
        }


@dataclass
class DecomposedReferenceTemplate:
    """
    Structured checklist representing a full reference template ready for V2 verification.
    """
    title: str
    points: List[DecomposedTemplatePoint] = field(default_factory=list)
    total_sub_requirements: int = 0
    source_path: str = ""

    def __post_init__(self):
        if not self.total_sub_requirements:
            self.total_sub_requirements = sum(len(p.sub_requirements) for p in self.points)

