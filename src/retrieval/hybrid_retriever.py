"""
V2 Hybrid Candidate Retrieval Engine with Reciprocal Rank Fusion (RRF).
Combines BM25 lexical search and MiniLM dense semantic search to produce
the Top-3 candidate evidence blocks for every sub-requirement.
"""
from typing import List, Dict, Tuple, Optional, Any

from src.models.document import DocumentBlock, TargetDocument
from src.models.requirement import SubRequirement
from src.models.retrieval import CandidateEvidence
from src.retrieval.bm25_search import BM25SearchEngine
from src.retrieval.dense_search import DenseSearchEngine


class HybridRetriever:
    """
    Combines lexical search (BM25) and dense semantic search (MiniLM) using
    Reciprocal Rank Fusion (RRF) to retrieve the top candidate passages for each sub-requirement.

    RRF formula:
        RRF_Score(d) = (w_bm25 / (k + rank_bm25(d))) + (w_dense / (k + rank_dense(d)))
    where default k = 60.
    """

    DEFAULT_K = 60
    DEFAULT_TOP_K = 3

    def __init__(
        self,
        bm25_engine: Optional[BM25SearchEngine] = None,
        dense_engine: Optional[DenseSearchEngine] = None,
        rrf_k: int = DEFAULT_K,
        weight_bm25: float = 1.0,
        weight_dense: float = 1.0
    ):
        self.bm25_engine = bm25_engine or BM25SearchEngine()
        self.dense_engine = dense_engine or DenseSearchEngine()
        self.rrf_k = rrf_k
        self.weight_bm25 = weight_bm25
        self.weight_dense = weight_dense
        self.indexed_document: Optional[TargetDocument] = None

    def index_document(self, document: TargetDocument, include_headings: bool = False):
        """
        Indexes the target document across both search engines.
        By default, include_headings=False ensures candidate evidence consists of
        substantive paragraphs, data rows, and omission statements rather than bare headings.
        """
        self.indexed_document = document
        self.bm25_engine.index_document(document, include_headings=include_headings)
        self.dense_engine.index_document(document, include_headings=include_headings)

    def retrieve_candidates_for_subrequirement(
        self,
        sub_req: SubRequirement,
        top_k: int = DEFAULT_TOP_K,
        candidate_pool_size: int = 15
    ) -> List[CandidateEvidence]:
        """
        Retrieves the top-k candidate evidence blocks for an individual atomic sub-requirement.
        Uses Reciprocal Rank Fusion to synthesize lexical and semantic candidate ranks.
        """
        if not self.indexed_document:
            raise RuntimeError("No document has been indexed. Call index_document() first.")

        # Construct search queries:
        # 1. Lexical query combines section context, topic label, and key nouns
        lexical_query = self._build_lexical_query(sub_req)
        # 2. Semantic query uses declarative claim text with section/topic framing
        semantic_query = self._build_semantic_query(sub_req)

        # Retrieve top candidates from each search engine
        bm25_results = self.bm25_engine.search(lexical_query, top_k=candidate_pool_size)
        dense_results = self.dense_engine.search(semantic_query, top_k=candidate_pool_size)

        # Merge rankings using Reciprocal Rank Fusion
        fused_candidates = self._reciprocal_rank_fusion(
            bm25_results=bm25_results,
            dense_results=dense_results,
            top_k=top_k
        )

        return fused_candidates

    def retrieve_candidates_for_all(
        self,
        sub_requirements: List[SubRequirement],
        top_k: int = DEFAULT_TOP_K
    ) -> Dict[str, List[CandidateEvidence]]:
        """
        Retrieves top-k candidates for all sub-requirements in a checklist.
        Returns a dictionary mapping sub_id -> List[CandidateEvidence].
        """
        results: Dict[str, List[CandidateEvidence]] = {}
        for sub_req in sub_requirements:
            candidates = self.retrieve_candidates_for_subrequirement(sub_req, top_k=top_k)
            results[sub_req.sub_id] = candidates
        return results

    def _build_lexical_query(self, sub_req: SubRequirement) -> str:
        """
        Constructs an information-dense query string for BM25 lexical matching.
        Synthesizes original reference text, section title, topic label, and claim keywords.
        """
        tokens = []
        if sub_req.section_name:
            tokens.append(sub_req.section_name)
        if sub_req.topic_label and sub_req.topic_label != sub_req.section_name:
            tokens.append(sub_req.topic_label)
        # Include original reference text to preserve the original keywords
        if sub_req.original_text:
            tokens.append(sub_req.original_text)
        tokens.append(sub_req.claim_text)
        return " ".join(tokens)

    def _build_semantic_query(self, sub_req: SubRequirement) -> str:
        """
        Constructs a dual-perspective query for dense semantic matching,
        incorporating the declarative claim and original reference wording.
        """
        parts = []
        if sub_req.topic_label:
            parts.append(f"{sub_req.topic_label}:")
        if sub_req.original_text:
            parts.append(sub_req.original_text)
        parts.append(sub_req.claim_text)
        return " ".join(parts)

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[DocumentBlock, float, int]],
        dense_results: List[Tuple[DocumentBlock, float, int]],
        top_k: int
    ) -> List[CandidateEvidence]:
        """
        Computes Reciprocal Rank Fusion scores across lexical and semantic ranks.
        """
        block_map: Dict[str, DocumentBlock] = {}
        bm25_info: Dict[str, Tuple[float, int]] = {}
        dense_info: Dict[str, Tuple[float, int]] = {}

        # Record BM25 scores and ranks
        for block, score, rank in bm25_results:
            bid = block.block_id
            block_map[bid] = block
            bm25_info[bid] = (score, rank)

        # Record Dense scores and ranks
        for block, score, rank in dense_results:
            bid = block.block_id
            block_map[bid] = block
            dense_info[bid] = (score, rank)

        # Compute combined RRF score for all candidate blocks
        rrf_scores: Dict[str, float] = {}
        for bid in block_map:
            score = 0.0
            if bid in bm25_info:
                bm25_rank = bm25_info[bid][1]
                score += self.weight_bm25 * (1.0 / (self.rrf_k + bm25_rank))
            if bid in dense_info:
                dense_rank = dense_info[bid][1]
                score += self.weight_dense * (1.0 / (self.rrf_k + dense_rank))
            rrf_scores[bid] = score

        # Sort blocks descending by RRF score
        sorted_bids = sorted(rrf_scores.keys(), key=lambda b: rrf_scores[b], reverse=True)

        candidates: List[CandidateEvidence] = []
        for final_rank, bid in enumerate(sorted_bids[:top_k], start=1):
            block = block_map[bid]
            bm25_s, bm25_r = bm25_info.get(bid, (0.0, None))
            dense_s, dense_r = dense_info.get(bid, (0.0, None))

            cand = CandidateEvidence.from_block(
                block=block,
                rrf_score=rrf_scores[bid],
                rrf_rank=final_rank,
                bm25_score=bm25_s,
                bm25_rank=bm25_r,
                dense_score=dense_s,
                dense_rank=dense_r
            )
            candidates.append(cand)

        return candidates
