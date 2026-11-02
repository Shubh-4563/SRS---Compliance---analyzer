"""
Unit and regression tests for V2 Hybrid Candidate Retrieval Engine (Phase 2.2).
Verifies:
1. BM25 lexical retrieval and tokenization.
2. Dense semantic retrieval with all-MiniLM-L6-v2.
3. Reciprocal Rank Fusion (RRF) rank calculation.
4. Top-3 candidate retrieval and provenance preservation (block_id, page, section_heading).
Cross-platform compatible (macOS and Windows).
"""
import os
import unittest
from pathlib import Path

from src.models.document import BlockType, DocumentBlock, TargetDocument
from src.models.requirement import SubRequirement, EvidenceType
from src.models.retrieval import CandidateEvidence
from src.parsers.pdf_parser import TargetPDFParser
from src.parsers.requirement_decomposer import RequirementDecomposer
from src.retrieval.bm25_search import BM25SearchEngine
from src.retrieval.dense_search import DenseSearchEngine
from src.retrieval.hybrid_retriever import HybridRetriever
from src.pdf_extractor import PDFExtractor


class TestV2HybridRetrieval(unittest.TestCase):
    """Unit tests for V2 Retrieval components."""

    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.a_target_path = os.path.join(cls.base_dir, "data", "test_packs", "A_target_lab_report.pdf")
        cls.b_target_path = os.path.join(cls.base_dir, "data", "test_packs", "B_target_technical_document.pdf")
        cls.a_template_path = os.path.join(cls.base_dir, "data", "test_packs", "A_reference_lab_report_template.pdf")
        cls.b_template_path = os.path.join(cls.base_dir, "data", "test_packs", "B_reference_technical_template.pdf")

        cls.parser = TargetPDFParser()
        cls.decomposer = RequirementDecomposer()
        cls.extractor = PDFExtractor()

    def test_bm25_tokenization_and_search(self):
        """Verifies BM25 indexing and lexical exact keyword matching."""
        if not os.path.isfile(self.a_target_path):
            self.skipTest("Missing Target A")

        doc_a = self.parser.parse_document(self.a_target_path)
        bm25 = BM25SearchEngine()
        bm25.index_document(doc_a, include_headings=False)

        # Query exact terms from Materials section
        results = bm25.search("beaker thermometer stopwatch notebook", top_k=3)
        self.assertGreaterEqual(len(results), 1)
        top_block, score, rank = results[0]
        self.assertEqual(rank, 1)
        self.assertGreater(score, 0.0)
        self.assertEqual(top_block.section_heading, "2. Materials")
        self.assertIn("thermometer", top_block.text)

    def test_dense_semantic_search(self):
        """Verifies DenseSearchEngine encoding and cosine similarity ranking."""
        if not os.path.isfile(self.b_target_path):
            self.skipTest("Missing Target B")

        doc_b = self.parser.parse_document(self.b_target_path)
        dense = DenseSearchEngine()
        dense.index_document(doc_b, include_headings=False)

        # Semantic query for equipment health monitoring
        results = dense.search("How does the dashboard assist maintenance personnel in monitoring machinery status?", top_k=3)
        self.assertGreaterEqual(len(results), 1)
        top_block, score, rank = results[0]
        self.assertEqual(rank, 1)
        self.assertGreater(score, 0.25)
        # Should retrieve Overview paragraph as top semantic candidate
        self.assertEqual(top_block.section_heading, "1. Overview")
        self.assertIn("monitor equipment condition", top_block.text)

    def test_rrf_mathematical_fusion(self):
        """Verifies Reciprocal Rank Fusion calculation logic."""
        retriever = HybridRetriever(rrf_k=60)

        # Mock results
        b1 = DocumentBlock(block_id="b1", text="Text 1", block_type=BlockType.PARAGRAPH, page_number=1)
        b2 = DocumentBlock(block_id="b2", text="Text 2", block_type=BlockType.PARAGRAPH, page_number=1)

        # b1 is rank 1 in BM25, rank 2 in Dense
        # b2 is rank 2 in BM25, rank 1 in Dense
        bm25_res = [(b1, 5.0, 1), (b2, 3.0, 2)]
        dense_res = [(b2, 0.8, 1), (b1, 0.6, 2)]

        fused = retriever._reciprocal_rank_fusion(bm25_res, dense_res, top_k=2)
        self.assertEqual(len(fused), 2)

        # Both have sum: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
        expected_score = round((1.0 / 61) + (1.0 / 62), 6)
        self.assertEqual(fused[0].rrf_score, expected_score)
        self.assertEqual(fused[1].rrf_score, expected_score)

    def test_provenance_and_metadata_preservation(self):
        """Verifies that CandidateEvidence preserves all metadata from the source DocumentBlock."""
        retriever = HybridRetriever()
        doc_b = self.parser.parse_document(self.b_target_path)
        retriever.index_document(doc_b, include_headings=False)

        sub_req = SubRequirement(
            sub_id="B4.1",
            parent_point_id="B4",
            claim_text="The document identifies system users or web browser operators.",
            expected_evidence_type=EvidenceType.FACTUAL_STATEMENT,
            section_name="3. Connections",
            topic_label="Interfaces",
            original_text="Identify the users, external systems, hardware, or communication interfaces."
        )

        candidates = retriever.retrieve_candidates_for_subrequirement(sub_req, top_k=3)
        self.assertEqual(len(candidates), 3)

        for c in candidates:
            self.assertIsInstance(c, CandidateEvidence)
            self.assertTrue(c.block_id.startswith("p1_b"))
            self.assertEqual(c.page_number, 1)
            self.assertIsNotNone(c.section_heading)
            self.assertGreater(len(c.text), 10)
            self.assertIsInstance(c.rrf_score, float)
            self.assertIsInstance(c.rrf_rank, int)

    def test_hybrid_retrieval_pair_b_coverage(self):
        """Verifies that key ground truth passages appear in the Top-3 candidates for Pair B."""
        if not os.path.isfile(self.b_target_path) or not os.path.isfile(self.b_template_path):
            self.skipTest("Missing Pair B files")

        doc_b = self.parser.parse_document(self.b_target_path)
        tpl_b = self.extractor.extract_template(self.b_template_path)
        dec_b = self.decomposer.decompose_template(tpl_b)

        retriever = HybridRetriever()
        retriever.index_document(doc_b, include_headings=False)

        all_subreqs = [s for p in dec_b.points for s in p.sub_requirements]
        results = retriever.retrieve_candidates_for_all(all_subreqs, top_k=3)

        # B1.1 Purpose: Overview paragraph must be in Top-3
        b1_cands = [c.text for c in results["B1.1"]]
        self.assertTrue(any("monitor equipment condition" in c for c in b1_cands))

        # B3.1 Core functions: Capabilities paragraph must be in Top-3
        b3_cands = [c.text for c in results["B3.1"]]
        self.assertTrue(any("collects readings from connected sensors" in c for c in b3_cands))

        # B4.1 Interfaces: Web browser operator statement must be in Top-3
        b4_cands = [c.text for c in results["B4.1"]]
        self.assertTrue(any("Operators use a web browser" in c for c in b4_cands))

        # B6.1 Acceptance criteria: Explicit omission of numeric target must be in Top-3
        b6_cands = [c.text for c in results["B6.1"]]
        self.assertTrue(any("No numeric response-time target" in c for c in b6_cands))

        # B7.1 Testing: Operator validation statement must be in Top-3
        b7_cands = [c.text for c in results["B7.1"]]
        self.assertTrue(any("During testing, an operator will confirm" in c for c in b7_cands))


if __name__ == "__main__":
    unittest.main()

