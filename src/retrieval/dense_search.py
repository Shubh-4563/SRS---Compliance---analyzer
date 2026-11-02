"""
V2 MiniLM Dense Semantic Search Engine.
Encodes DocumentBlocks and sub-requirement queries using all-MiniLM-L6-v2
for conceptual paraphrases and semantic candidate retrieval.
"""
from typing import List, Tuple, Optional, Dict, Any

from src.models.document import DocumentBlock, TargetDocument


class DenseSearchEngine:
    """
    Dense semantic search engine using sentence-transformers (all-MiniLM-L6-v2).
    Indexes DocumentBlock embeddings and returns top-k candidate blocks ranked by cosine similarity.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.MODEL_NAME
        self._model = None
        self._util = None
        self.blocks: List[DocumentBlock] = []
        self.block_embeddings = None

    def _load_model(self):
        """Lazy load sentence_transformers to avoid overhead when not needed."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer, util
                # Prefer local offline cache first to avoid network timeouts on firewalled machines
                try:
                    self._model = SentenceTransformer(self.model_name, local_files_only=True)
                except Exception:
                    self._model = SentenceTransformer(self.model_name)
                self._util = util
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load embedding model '{self.model_name}': {exc}. "
                    "Ensure sentence-transformers and PyTorch are installed."
                ) from exc

    def index_document(self, document: TargetDocument, include_headings: bool = True):
        """
        Encodes all blocks in the TargetDocument into dense vectors.
        """
        self._load_model()
        if include_headings:
            self.blocks = list(document.blocks)
        else:
            self.blocks = list(document.substantive_blocks)

        if not self.blocks:
            self.block_embeddings = None
            return

        texts_to_encode = []
        for block in self.blocks:
            prefix = f"[{block.section_heading}] " if block.section_heading else ""
            texts_to_encode.append(f"{prefix}{block.text}")

        self.block_embeddings = self._model.encode(
            texts_to_encode,
            convert_to_tensor=True,
            show_progress_bar=False
        )

    def search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentBlock, float, int]]:
        """
        Performs dense semantic search for a given query string.

        Returns:
            List of tuples: (DocumentBlock, cosine_similarity, rank_1_based)
        """
        if self.block_embeddings is None or not self.blocks:
            return []

        self._load_model()
        query_emb = self._model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        sim_scores = self._util.cos_sim(query_emb, self.block_embeddings)[0]

        scored_blocks = [(self.blocks[i], float(sim_scores[i].item())) for i in range(len(self.blocks))]
        scored_blocks.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (block, score) in enumerate(scored_blocks[:top_k], start=1):
            results.append((block, score, rank))

        return results

