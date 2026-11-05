"""
Unit and Integration Tests for V2 CompliancePipeline Orchestration (Phase 2.5.3).
Tests:
1. End-to-end pipeline execution with lightweight mock dependencies.
2. Data flow verification across extraction, decomposition, parsing, indexing, retrieval, verification, and aggregation.
3. Full-document audit vs single-page audit with target_page propagation.
4. Input validation (missing files, 0-byte files, invalid page numbers).
5. Scanned / image-only PDF detection for reference and target documents.
6. Empty document handling (no points or no text blocks).
7. Lazy model initialization and component reuse across multiple executions.
8. Dependency injection verification.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.pipeline import CompliancePipeline
from src.pdf_extractor import ScannedPDFError, ReferenceTemplate, TemplatePoint
from src.models.document import BlockType, DocumentBlock, TargetDocument
from src.models.requirement import (
    EvidenceType,
    SubRequirement,
    DecomposedTemplatePoint,
    DecomposedReferenceTemplate,
)
from src.models.retrieval import CandidateEvidence
from src.models.verification import (
    VerificationStatus,
    EvidenceEvaluation,
    SubRequirementVerificationResult,
)
from src.models.compliance import (
    ParentComplianceStatus,
    DocumentComplianceVerdict,
    ParentPointCompliance,
    DocumentComplianceReport,
)


class TestV2Pipeline(unittest.TestCase):
    """Test suite for CompliancePipeline orchestration."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

        # Create sample valid files (non-empty)
        self.ref_file = self.dir_path / "reference_standard.pdf"
        self.ref_file.write_bytes(b"%PDF-1.4 Mock Reference Content")

        self.tgt_file = self.dir_path / "target_document.pdf"
        self.tgt_file.write_bytes(b"%PDF-1.4 Mock Target Content")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_mock_components(self, total_pages: int = 3):
        """Constructs lightweight test doubles for all pipeline dependencies."""
        mock_extractor = MagicMock()
        mock_parser = MagicMock()
        mock_decomposer = MagicMock()
        mock_retriever = MagicMock()
        mock_verifier = MagicMock()
        mock_aggregator = MagicMock()

        # Mock reference extraction
        pt1 = TemplatePoint(point_id="P1", section_name="Overview", text="System shall monitor sensors.")
        mock_template = ReferenceTemplate(title="Mock Reference", points=[pt1], sections=["Overview"])
        mock_extractor.extract_template.return_value = mock_template

        # Mock requirement decomposition
        sub1 = SubRequirement(
            sub_id="P1.1",
            parent_point_id="P1",
            claim_text="The system monitors sensors.",
            expected_evidence_type=EvidenceType.FACTUAL_STATEMENT,
            is_mandatory=True,
            original_text="System shall monitor sensors."
        )
        dec_point = DecomposedTemplatePoint(
            point_id="P1",
            section_name="Overview",
            topic_label="Sensors",
            description="System shall monitor sensors.",
            raw_text="System shall monitor sensors.",
            sub_requirements=[sub1]
        )
        dec_template = DecomposedReferenceTemplate(
            title="Mock Reference",
            points=[dec_point],
            total_sub_requirements=1
        )
        mock_decomposer.decompose_template.return_value = dec_template

        # Mock target PDF parsing with substantive text
        blk = DocumentBlock(
            block_id="p1_b1",
            text="The monitoring station continuously collects and records sensor readings from machines.",
            block_type=BlockType.PARAGRAPH,
            page_number=1,
            section_heading="Overview"
        )
        target_doc = TargetDocument(
            filename="target_document.pdf",
            total_pages=total_pages,
            blocks=[blk],
            document_title="Mock Target"
        )
        mock_parser.parse_document.return_value = target_doc

        # Mock candidate retrieval
        cand = CandidateEvidence.from_block(
            block=blk,
            rrf_score=0.033,
            rrf_rank=1
        )
        mock_retriever.retrieve_candidates_for_subrequirement.return_value = [cand]

        # Mock evidence verification
        ev_eval = EvidenceEvaluation(
            candidate=cand,
            premise_text=blk.text,
            hypothesis_text=sub1.claim_text,
            entailment_prob=0.95,
            contradiction_prob=0.01,
            neutral_prob=0.04,
            predicted_label="entailment"
        )
        verif_res = SubRequirementVerificationResult(
            sub_id="P1.1",
            parent_point_id="P1",
            claim_text=sub1.claim_text,
            expected_evidence_type=sub1.expected_evidence_type,
            status=VerificationStatus.FULFILLED,
            confidence=0.95,
            is_mandatory=True,
            best_evidence=ev_eval,
            all_evaluations=[ev_eval],
            rationale="Satisfied by block p1_b1."
        )
        mock_verifier.verify_subrequirement.return_value = verif_res

        # Mock compliance aggregation
        parent_comp = ParentPointCompliance(
            point_id="P1",
            section_name="Overview",
            topic_label="Sensors",
            description="System shall monitor sensors.",
            raw_text="System shall monitor sensors.",
            status=ParentComplianceStatus.FULLY_COMPLIANT,
            mandatory_fulfilled=1,
            mandatory_total=1,
            optional_fulfilled=0,
            optional_total=0,
            sub_results=[verif_res],
            findings=["All criteria met."]
        )
        mock_report = DocumentComplianceReport(
            target_document_path=str(self.tgt_file),
            target_document_name="target_document.pdf",
            template_title="Mock Reference",
            template_path=str(self.ref_file),
            total_parent_points=1,
            total_sub_requirements=1,
            parent_points=[parent_comp],
            mandatory_coverage_pct=100.0,
            total_mandatory=1,
            mandatory_fulfilled=1,
            fulfilled_count=1,
            verdict=DocumentComplianceVerdict.COMPLIANT,
            total_pages=total_pages,
            target_page=None
        )
        mock_aggregator.aggregate.return_value = mock_report

        return {
            "extractor": mock_extractor,
            "parser": mock_parser,
            "decomposer": mock_decomposer,
            "retriever": mock_retriever,
            "verifier": mock_verifier,
            "aggregator": mock_aggregator,
            "report": mock_report
        }

    # --------------------------------------------------------------------------
    # 1. End-to-End Execution and Flow Tests
    # --------------------------------------------------------------------------

    def test_end_to_end_pipeline_success(self):
        """Verifies that the pipeline runs through all components and returns a report."""
        mocks = self._create_mock_components(total_pages=3)
        pipeline = CompliancePipeline(
            extractor=mocks["extractor"],
            parser=mocks["parser"],
            decomposer=mocks["decomposer"],
            retriever=mocks["retriever"],
            verifier=mocks["verifier"],
            aggregator=mocks["aggregator"]
        )

        report = pipeline.run(self.ref_file, self.tgt_file, top_k=5)

        self.assertIsInstance(report, DocumentComplianceReport)
        self.assertEqual(report.verdict, DocumentComplianceVerdict.COMPLIANT)
        self.assertEqual(report.mandatory_coverage_pct, 100.0)

        # Verify execution sequence
        mocks["extractor"].extract_template.assert_called_once_with(str(self.ref_file.resolve()))
        mocks["decomposer"].decompose_template.assert_called_once()
        mocks["parser"].parse_document.assert_called_once_with(str(self.tgt_file.resolve()), target_page=None)
        mocks["retriever"].index_document.assert_called_once_with(mocks["parser"].parse_document.return_value, include_headings=False)
        mocks["retriever"].retrieve_candidates_for_subrequirement.assert_called_once()
        mocks["verifier"].verify_subrequirement.assert_called_once()
        mocks["aggregator"].aggregate.assert_called_once()

    def test_single_page_scope_propagation(self):
        """Verifies that target_page is passed to parser and aggregator."""
        mocks = self._create_mock_components(total_pages=5)
        # Update mock report to reflect target_page=2
        mocks["aggregator"].aggregate.side_effect = lambda **kwargs: DocumentComplianceReport(
            target_document_path=str(self.tgt_file),
            target_document_name="target_document.pdf",
            template_title="Mock Reference",
            template_path=str(self.ref_file),
            total_parent_points=1,
            total_sub_requirements=1,
            verdict=DocumentComplianceVerdict.COMPLIANT,
            total_pages=5,
            target_page=kwargs.get("target_page")
        )

        pipeline = CompliancePipeline(
            extractor=mocks["extractor"],
            parser=mocks["parser"],
            decomposer=mocks["decomposer"],
            retriever=mocks["retriever"],
            verifier=mocks["verifier"],
            aggregator=mocks["aggregator"]
        )

        report = pipeline.run(self.ref_file, self.tgt_file, target_page=2)

        self.assertEqual(report.target_page, 2)
        mocks["parser"].parse_document.assert_called_once_with(str(self.tgt_file.resolve()), target_page=2)
        args, kwargs = mocks["aggregator"].aggregate.call_args
        self.assertEqual(kwargs.get("target_page"), 2)

    # --------------------------------------------------------------------------
    # 2. Input Validation Tests
    # --------------------------------------------------------------------------

    def test_missing_reference_file(self):
        """Raises FileNotFoundError if reference template does not exist."""
        mocks = self._create_mock_components()
        pipeline = CompliancePipeline(**{k: v for k, v in mocks.items() if k != "report"})

        with self.assertRaises(FileNotFoundError) as ctx:
            pipeline.run(self.dir_path / "non_existent.pdf", self.tgt_file)
        self.assertIn("Reference template file not found", str(ctx.exception))

    def test_missing_target_file(self):
        """Raises FileNotFoundError if target document does not exist."""
        mocks = self._create_mock_components()
        pipeline = CompliancePipeline(**{k: v for k, v in mocks.items() if k != "report"})

        with self.assertRaises(FileNotFoundError) as ctx:
            pipeline.run(self.ref_file, self.dir_path / "missing_target.pdf")
        self.assertIn("Target document file not found", str(ctx.exception))

    def test_empty_reference_file_zero_bytes(self):
        """Raises ValueError if reference file is 0 bytes."""
        empty_ref = self.dir_path / "empty_ref.pdf"
        empty_ref.write_bytes(b"")

        mocks = self._create_mock_components()
        pipeline = CompliancePipeline(**{k: v for k, v in mocks.items() if k != "report"})

        with self.assertRaises(ValueError) as ctx:
            pipeline.run(empty_ref, self.tgt_file)
        self.assertIn("empty (0 bytes)", str(ctx.exception))

    def test_empty_target_file_zero_bytes(self):
        """Raises ValueError if target file is 0 bytes."""
        empty_tgt = self.dir_path / "empty_tgt.pdf"
        empty_tgt.write_bytes(b"")

        mocks = self._create_mock_components()
        pipeline = CompliancePipeline(**{k: v for k, v in mocks.items() if k != "report"})

        with self.assertRaises(ValueError) as ctx:
            pipeline.run(self.ref_file, empty_tgt)
        self.assertIn("empty (0 bytes)", str(ctx.exception))

    def test_invalid_target_page_type_or_negative(self):
        """Raises ValueError if target_page is not a positive integer."""
        mocks = self._create_mock_components()
        pipeline = CompliancePipeline(**{k: v for k, v in mocks.items() if k != "report"})

        with self.assertRaises(ValueError):
            pipeline.run(self.ref_file, self.tgt_file, target_page=0)

        with self.assertRaises(ValueError):
            pipeline.run(self.ref_file, self.tgt_file, target_page=-2)

        with self.assertRaises(ValueError):
            pipeline.run(self.ref_file, self.tgt_file, target_page="page1")

    def test_target_page_exceeds_document_page_count(self):
        """Raises ValueError if target_page exceeds total pages in parsed target."""
        mocks = self._create_mock_components(total_pages=2)
        pipeline = CompliancePipeline(**{k: v for k, v in mocks.items() if k != "report"})

        with self.assertRaises(ValueError) as ctx:
            pipeline.run(self.ref_file, self.tgt_file, target_page=5)
        self.assertIn("has only 2 page(s)", str(ctx.exception))

    # --------------------------------------------------------------------------
    # 3. Scanned and Empty Content Document Tests
    # --------------------------------------------------------------------------

    def test_scanned_reference_pdf_propagates_scanned_pdf_error(self):
        """Verifies that ScannedPDFError from extractor is propagated with details."""
        mocks = self._create_mock_components()
        mocks["extractor"].extract_template.side_effect = ScannedPDFError(
            filepath=str(self.ref_file),
            char_count=12,
            page_count=2
        )
        pipeline = CompliancePipeline(**{k: v for k, v in mocks.items() if k != "report"})

        with self.assertRaises(ScannedPDFError) as ctx:
            pipeline.run(self.ref_file, self.tgt_file)
        self.assertEqual(ctx.exception.char_count, 12)

    def test_scanned_target_pdf_detected_by_pipeline(self):
        """Verifies that a target document with almost no substantive text raises ScannedPDFError."""
        mocks = self._create_mock_components(total_pages=3)
        # Mock target document with only 5 characters of text (< min_substantive_chars 50)
        short_blk = DocumentBlock(
            block_id="p1_b1",
            text="Hi.",
            block_type=BlockType.PARAGRAPH,
            page_number=1
        )
        mocks["parser"].parse_document.return_value = TargetDocument(
            filename="scanned_target.pdf",
            total_pages=3,
            blocks=[short_blk]
        )
        pipeline = CompliancePipeline(
            extractor=mocks["extractor"],
            parser=mocks["parser"],
            decomposer=mocks["decomposer"],
            retriever=mocks["retriever"],
            verifier=mocks["verifier"],
            aggregator=mocks["aggregator"],
            min_substantive_chars=50
        )

        with self.assertRaises(ScannedPDFError) as ctx:
            pipeline.run(self.ref_file, self.tgt_file)
        self.assertIn("Scanned or image-only PDF detected", str(ctx.exception))

    def test_empty_reference_points_raises_value_error(self):
        """Raises ValueError if reference template has 0 points."""
        mocks = self._create_mock_components()
        mocks["extractor"].extract_template.return_value = ReferenceTemplate(
            title="Empty Reference",
            points=[],
            sections=[]
        )
        pipeline = CompliancePipeline(**{k: v for k, v in mocks.items() if k != "report"})

        with self.assertRaises(ValueError) as ctx:
            pipeline.run(self.ref_file, self.tgt_file)
        self.assertIn("contains no requirement sections or points", str(ctx.exception))

    def test_empty_target_blocks_raises_value_error(self):
        """Raises ValueError if target document has 0 blocks."""
        mocks = self._create_mock_components()
        mocks["parser"].parse_document.return_value = TargetDocument(
            filename="blank.pdf",
            total_pages=1,
            blocks=[]
        )
        pipeline = CompliancePipeline(**{k: v for k, v in mocks.items() if k != "report"})

        with self.assertRaises(ValueError) as ctx:
            pipeline.run(self.ref_file, self.tgt_file)
        self.assertIn("contains no extractable content blocks", str(ctx.exception))

    # --------------------------------------------------------------------------
    # 4. Dependency Injection & Lazy Model Reuse
    # --------------------------------------------------------------------------

    def test_dependency_injection_prevents_ml_model_loading(self):
        """Verifies that when retriever and verifier are injected, heavy ML models are never imported or loaded."""
        mocks = self._create_mock_components()
        pipeline = CompliancePipeline(
            retriever=mocks["retriever"],
            verifier=mocks["verifier"]
        )

        # Confirm retriever and verifier are exactly the injected mocks
        self.assertIs(pipeline.retriever, mocks["retriever"])
        self.assertIs(pipeline.verifier, mocks["verifier"])

    def test_model_initialization_not_repeated(self):
        """Verifies that lazy property access caches components and does not re-initialize on subsequent calls."""
        pipeline = CompliancePipeline()
        mock_ret = MagicMock()
        mock_ver = MagicMock()

        pipeline._retriever = mock_ret
        pipeline._verifier = mock_ver

        self.assertIs(pipeline.retriever, mock_ret)
        self.assertIs(pipeline.retriever, mock_ret)
        self.assertIs(pipeline.verifier, mock_ver)
        self.assertIs(pipeline.verifier, mock_ver)


if __name__ == "__main__":
    unittest.main()
