"""
V2 Core Data Models for Document Blocks, Structural Hierarchy, and Extracted Content.
Designed for cross-platform execution (macOS & Windows) with clean type annotations.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class BlockType(str, Enum):
    """Semantic type of an extracted document block."""
    HEADING = "HEADING"
    PARAGRAPH = "PARAGRAPH"
    LIST_ITEM = "LIST_ITEM"
    TABLE_ROW = "TABLE_ROW"
    DISCLAIMER = "DISCLAIMER"  # Explicit absence or scope exclusion statement


@dataclass
class DocumentBlock:
    """
    A discrete, semantically cohesive block of content extracted from a target document.
    Preserves exact provenance (page, line range, section hierarchy, parent headings).
    """
    block_id: str                          # Unique deterministic ID, e.g. "p1_b3"
    text: str                              # Clean text content of the block
    block_type: BlockType                  # Semantic classification
    page_number: int                       # 1-based page number where block begins
    section_heading: Optional[str] = None  # Immediate parent section title (e.g. "3. Connections")
    section_hierarchy: List[str] = field(default_factory=list)  # Full breadcrumb (e.g. ["Chapter 1", "3. Connections"])
    line_start: int = 0                    # Approximate line number in source text
    line_end: int = 0                      # Approximate line number end
    is_explicit_omission: bool = False     # True if block explicitly asserts absence or omission
    metadata: Dict[str, Any] = field(default_factory=dict)     # Additional properties (e.g. table columns, raw markup)

    def to_dict(self) -> Dict[str, Any]:
        """Convert block to JSON-serializable dictionary."""
        return {
            "block_id": self.block_id,
            "text": self.text,
            "block_type": self.block_type.value,
            "page_number": self.page_number,
            "section_heading": self.section_heading,
            "section_hierarchy": self.section_hierarchy,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "is_explicit_omission": self.is_explicit_omission,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        prefix = f"[{self.section_heading}] " if self.section_heading else ""
        return f"{prefix}{self.text}"


@dataclass
class TargetDocument:
    """
    Structured representation of an entire parsed target document or single page.
    """
    filename: str
    total_pages: int
    blocks: List[DocumentBlock] = field(default_factory=list)
    document_title: Optional[str] = None
    target_page: Optional[int] = None       # None if whole document, or specific 1-based page

    @property
    def headings(self) -> List[DocumentBlock]:
        return [b for b in self.blocks if b.block_type == BlockType.HEADING]

    @property
    def substantive_blocks(self) -> List[DocumentBlock]:
        """Returns non-heading blocks that contain candidate evidence."""
        return [b for b in self.blocks if b.block_type != BlockType.HEADING]

    def get_blocks_for_page(self, page_number: int) -> List[DocumentBlock]:
        return [b for b in self.blocks if b.page_number == page_number]

