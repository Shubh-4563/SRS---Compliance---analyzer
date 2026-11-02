"""
V2 Candidate Retrieval Data Structures.
Captures candidate ranking, retrieval source scores (BM25, Dense, RRF),
and exact provenance to source DocumentBlocks.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from src.models.document import DocumentBlock, BlockType


@dataclass
class CandidateEvidence:
    """
    A candidate evidence passage retrieved for a specific sub-requirement.
    Preserves exact provenance back to the source DocumentBlock.
    """
    block_id: str                      # Source DocumentBlock ID, e.g. "p1_b7"
    text: str                          # Exact text of the candidate evidence block
    block_type: BlockType              # Semantic type (PARAGRAPH, HEADING, TABLE_ROW, etc.)
    page_number: int                   # Source page in target document (1-based)
    section_heading: Optional[str]     # Immediate section heading
    rrf_score: float                   # Combined Reciprocal Rank Fusion score
    rrf_rank: int                      # Combined rank (1 = top candidate)
    bm25_score: float = 0.0            # Raw BM25 lexical score
    bm25_rank: Optional[int] = None    # Rank in lexical retrieval
    dense_score: float = 0.0           # Cosine similarity score from dense embeddings
    dense_rank: Optional[int] = None   # Rank in semantic retrieval
    is_explicit_omission: bool = False # True if block explicitly asserts omission
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_block(
        cls,
        block: DocumentBlock,
        rrf_score: float,
        rrf_rank: int,
        bm25_score: float = 0.0,
        bm25_rank: Optional[int] = None,
        dense_score: float = 0.0,
        dense_rank: Optional[int] = None
    ) -> "CandidateEvidence":
        return cls(
            block_id=block.block_id,
            text=block.text,
            block_type=block.block_type,
            page_number=block.page_number,
            section_heading=block.section_heading,
            rrf_score=round(rrf_score, 6),
            rrf_rank=rrf_rank,
            bm25_score=round(bm25_score, 4),
            bm25_rank=bm25_rank,
            dense_score=round(dense_score, 4),
            dense_rank=dense_rank,
            is_explicit_omission=block.is_explicit_omission,
            metadata=dict(block.metadata)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "text": self.text,
            "block_type": self.block_type.value,
            "page_number": self.page_number,
            "section_heading": self.section_heading,
            "rrf_score": self.rrf_score,
            "rrf_rank": self.rrf_rank,
            "bm25_score": self.bm25_score,
            "bm25_rank": self.bm25_rank,
            "dense_score": self.dense_score,
            "dense_rank": self.dense_rank,
            "is_explicit_omission": self.is_explicit_omission,
            "metadata": self.metadata,
        }

