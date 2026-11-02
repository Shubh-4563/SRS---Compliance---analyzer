"""
V2 BM25 Lexical Search Engine.
Builds an inverted index of TargetDocument blocks using rank-bm25 with
cross-platform text tokenization and stopword removal.
"""
import re
from typing import List, Tuple, Dict, Any, Optional

from rank_bm25 import BM25Okapi

from src.models.document import DocumentBlock, TargetDocument


# Minimal cross-platform English stopwords
ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "during", "or", "other", "so", "this"
}

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")


class BM25SearchEngine:
    """
    Lexical retrieval engine based on Okapi BM25.
    Indexes DocumentBlock text plus section heading context for high recall on
    technical terms, acronyms, equipment names, and numerical units.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.blocks: List[DocumentBlock] = []

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text into lowercase alphanumeric tokens, filtering single-char noise
        and standard stopwords, with basic suffix normalization.
        """
        if not text:
            return []
        raw_tokens = TOKEN_PATTERN.findall(text.lower())
        tokens: List[str] = []
        for t in raw_tokens:
            if t in ENGLISH_STOPWORDS or len(t) <= 1:
                continue
            # Simple suffix normalization (e.g. 'observations' -> 'observation', 'recorded' -> 'record')
            if t.endswith("ies") and len(t) > 4:
                norm_t = t[:-3] + "y"
            elif t.endswith("ed") and len(t) > 4 and not t.endswith(("eed", "bed", "red")):
                norm_t = t[:-2]
                if norm_t.endswith("i"):
                    norm_t = norm_t[:-1] + "y"
            elif t.endswith("ing") and len(t) > 5:
                norm_t = t[:-3]
            elif t.endswith("es") and len(t) > 4 and not t.endswith(("sses", "ches", "shes")):
                norm_t = t[:-2]
            elif t.endswith("s") and len(t) > 3 and not t.endswith("ss"):
                norm_t = t[:-1]
            else:
                norm_t = t
            tokens.append(norm_t)
        return tokens

    def index_document(self, document: TargetDocument, include_headings: bool = True):
        """
        Indexes all candidate blocks in the TargetDocument.
        """
        if include_headings:
            self.blocks = list(document.blocks)
        else:
            self.blocks = list(document.substantive_blocks)

        corpus_tokens = []
        for block in self.blocks:
            # Enrich document text with section heading for contextual lexical matching
            heading_prefix = f"{block.section_heading} " if block.section_heading else ""
            full_text = f"{heading_prefix}{block.text}"
            tokens = self.tokenize(full_text)
            corpus_tokens.append(tokens)

        if corpus_tokens:
            self.bm25 = BM25Okapi(corpus_tokens, k1=self.k1, b=self.b)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentBlock, float, int]]:
        """
        Performs BM25 lexical search for a given query string.

        Returns:
            List of tuples: (DocumentBlock, bm25_score, rank_1_based)
        """
        if not self.bm25 or not self.blocks:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            # Fallback to character split if tokenization filtered all words
            query_tokens = [w.lower() for w in query.split() if len(w) > 1]
            if not query_tokens:
                return []

        scores = self.bm25.get_scores(query_tokens)

        # Pair each block with its score
        scored_blocks = [(self.blocks[i], float(scores[i])) for i in range(len(self.blocks))]
        # Sort descending by score
        scored_blocks.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (block, score) in enumerate(scored_blocks[:top_k], start=1):
            results.append((block, score, rank))

        return results
