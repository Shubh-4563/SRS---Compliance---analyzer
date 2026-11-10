"""
Unit and integration tests for V2 BenchmarkRunner & Accuracy Evaluator (Phase 2.6.3).
Covers:
- Retrieval metric computations (Hit@1, Hit@3, Hit@5, MRR, Precision, Recall).
- Verification metric computations (Accuracy, Macro-F1, Confusion Matrix, per-status metrics).
- Hierarchical compliance metrics (Parent accuracy, verdict match, coverage difference).
- Discrepancy detection (RETRIEVAL_MISS, VERIFICATION_MISMATCH, PARENT_STATUS_MISMATCH, VERDICT_MISMATCH).
- Terminal and Markdown formatting validation (ensuring zero markdown tables).
- End-to-end integration execution on production Pair A and Pair B manifests.
"""
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.benchmarks.benchmark_runner import BenchmarkRunner
from src.benchmarks.manifest_validator import load_and_validate_manifest
from src.models.benchmark import (
    BenchmarkManifest,
    BenchmarkMetadata,
    BenchmarkDocumentMetadata,
    BenchmarkParentRequirement,
    BenchmarkSubRequirement,
    BenchmarkGroundTruth,
    BenchmarkRetrievalGroundTruth,
    BenchmarkEvidenceAnnotation,
    BenchmarkVerificationGroundTruth,
    BenchmarkExpectedCompliance,
    BenchmarkExpectedStatusCounts,
    BenchmarkEvidencePresence,
    BenchmarkEvidenceType,
    BenchmarkVerificationStatus,
    BenchmarkParentComplianceStatus,
    BenchmarkDocumentComplianceVerdict,
)
from src.models.compliance import (
    DocumentComplianceReport,
    ParentPointCompliance,
    ParentComplianceStatus,
    DocumentComplianceVerdict,
)
from src.models.document import BlockType, DocumentBlock
from src.models.requirement import EvidenceType, SubRequirement
from src.models.retrieval import CandidateEvidence
from src.models.verification import (
    VerificationStatus,
    EvidenceEvaluation,
    SubRequirementVerificationResult,
)


class TestBenchmarkRunner(unittest.TestCase):
    """Test suite for BenchmarkRunner metric calculations and evaluation workflows."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.benchmarks_dir = cls.project_root / "data" / "benchmarks"
        cls.pair_a_path = cls.benchmarks_dir / "pair_a_lab_report.json"
        cls.pair_b_path = cls.benchmarks_dir / "pair_b_srs.json"

    def _create_mock_manifest(self) -> BenchmarkManifest:
        """Constructs a minimal valid BenchmarkManifest for unit testing."""
        return BenchmarkManifest(
            schema_version="2.0.0",
            metadata=BenchmarkMetadata(
                pair_id="PAIR_MOCK",
                domain="Testing",
                title="Mock Benchmark",
                description="Mock manifest for testing",
                reference_document=BenchmarkDocumentMetadata(
                    path="data/test_packs/A_reference_lab_report_template.pdf",
                    filename="A_reference_lab_report_template.pdf",
                    sha256="b8256074931289363371bd6f4b5948015a7df5784217f2c066167b0dbfa7de42",
                ),
                target_document=BenchmarkDocumentMetadata(
                    path="data/test_packs/A_target_lab_report.pdf",
                    filename="A_target_lab_report.pdf",
                    sha256="c8dfb2b85907c5ebba2c7c915621a0d988cc81314effd43143c2cf7b276dee50",
                    total_pages=1,
                ),
                tags=["mock"],
            ),
            parent_requirements=[
                BenchmarkParentRequirement(
                    parent_id="M1",
                    section_name="Mock Section",
                    topic_label="Mock Topic",
                    description="Mock Description",
                    is_mandatory=True,
                    expected_compliance_status=BenchmarkParentComplianceStatus.FULLY_COMPLIANT,
                    sub_requirements=[
                        BenchmarkSubRequirement(
                            sub_id="M1.1",
                            claim_text="Mock claim 1",
                            expected_evidence_type=BenchmarkEvidenceType.FACTUAL_STATEMENT,
                            is_mandatory=True,
                            decomposition_rationale="Rationale 1",
                            ground_truth=BenchmarkGroundTruth(
                                retrieval=BenchmarkRetrievalGroundTruth(
                                    evidence_presence=BenchmarkEvidencePresence.EXISTS,
                                    expected_evidence=[
                                        BenchmarkEvidenceAnnotation(block_id="p1_b3", page=1, excerpt="evidence 1")
                                    ],
                                ),
                                verification=BenchmarkVerificationGroundTruth(
                                    expected_status=BenchmarkVerificationStatus.FULFILLED,
                                    expected_rationale_notes="Notes 1",
                                ),
                            ),
                        ),
                        BenchmarkSubRequirement(
                            sub_id="M1.2",
                            claim_text="Mock claim 2",
                            expected_evidence_type=BenchmarkEvidenceType.FACTUAL_STATEMENT,
                            is_mandatory=True,
                            decomposition_rationale="Rationale 2",
                            ground_truth=BenchmarkGroundTruth(
                                retrieval=BenchmarkRetrievalGroundTruth(
                                    evidence_presence=BenchmarkEvidencePresence.EXPLICIT_OMISSION,
                                    expected_evidence=[
                                        BenchmarkEvidenceAnnotation(block_id="p1_b14", page=1, excerpt="disclaimer")
                                    ],
                                ),
                                verification=BenchmarkVerificationGroundTruth(
                                    expected_status=BenchmarkVerificationStatus.EXPLICIT_OMISSION,
                                    expected_rationale_notes="Notes 2",
                                ),
                            ),
                        ),
                    ],
                )
            ],
            expected_document_compliance=BenchmarkExpectedCompliance(
                mandatory_total=2,
                mandatory_fulfilled=1,
                mandatory_coverage_pct=50.0,
                optional_total=0,
                optional_fulfilled=0,
                optional_coverage_pct=None,
                expected_status_counts=BenchmarkExpectedStatusCounts(
                    fulfilled=1,
                    explicit_omission=1,
                    contradicted=0,
                    missing=0,
                    needs_review=0,
                ),
                verdict=BenchmarkDocumentComplianceVerdict.NON_COMPLIANT,
            ),
        )

    def _create_mock_report(
        self,
        m1_1_status: VerificationStatus = VerificationStatus.FULFILLED,
        m1_1_block: str = "p1_b3",
        m1_2_status: VerificationStatus = VerificationStatus.EXPLICIT_OMISSION,
        m1_2_block: str = "p1_b14",
        parent_status: ParentComplianceStatus = ParentComplianceStatus.FULLY_COMPLIANT,
        verdict: DocumentComplianceVerdict = DocumentComplianceVerdict.NON_COMPLIANT,
    ) -> DocumentComplianceReport:
        """Constructs a mock DocumentComplianceReport matching or deviating from mock manifest."""
        def make_cand(block_id: str, rank: int = 1) -> CandidateEvidence:
            b = DocumentBlock(
                block_id=block_id,
                page_number=1,
                text=f"Sample text in {block_id}",
                block_type=BlockType.PARAGRAPH,
            )
            return CandidateEvidence.from_block(b, rrf_score=0.03, rrf_rank=rank)

        cand_1 = make_cand(m1_1_block, rank=1)
        eval_1 = EvidenceEvaluation(
            candidate=cand_1,
            premise_text="sample premise",
            hypothesis_text="sample hypothesis",
            entailment_prob=0.95,
            contradiction_prob=0.02,
            neutral_prob=0.03,
            predicted_label="entailment",
        )
        res_1 = SubRequirementVerificationResult(
            sub_id="M1.1",
            parent_point_id="M1",
            claim_text="Mock claim 1",
            expected_evidence_type=EvidenceType.FACTUAL_STATEMENT,
            status=m1_1_status,
            confidence=0.95,
            is_mandatory=True,
            best_evidence=eval_1,
            all_evaluations=[eval_1],
            rationale="Verified successfully",
        )

        cand_irrelevant = make_cand("p1_b99", rank=1)
        eval_irrelevant = EvidenceEvaluation(
            candidate=cand_irrelevant,
            premise_text="irrelevant premise",
            hypothesis_text="sample hypothesis 2",
            entailment_prob=0.01,
            contradiction_prob=0.01,
            neutral_prob=0.98,
            predicted_label="neutral",
        )

        cand_2 = make_cand(m1_2_block, rank=2)
        eval_2 = EvidenceEvaluation(
            candidate=cand_2,
            premise_text="sample disclaimer",
            hypothesis_text="sample hypothesis 2",
            entailment_prob=0.02,
            contradiction_prob=0.03,
            neutral_prob=0.95,
            predicted_label="neutral",
        )
        res_2 = SubRequirementVerificationResult(
            sub_id="M1.2",
            parent_point_id="M1",
            claim_text="Mock claim 2",
            expected_evidence_type=EvidenceType.FACTUAL_STATEMENT,
            status=m1_2_status,
            confidence=0.95,
            is_mandatory=True,
            best_evidence=eval_2,
            all_evaluations=[eval_irrelevant, eval_2],
            rationale="Disclaimed successfully",
        )

        parent_pt = ParentPointCompliance(
            point_id="M1",
            section_name="Mock Section",
            topic_label="Mock Topic",
            description="Mock Description",
            raw_text="Mock raw text",
            status=parent_status,
            mandatory_fulfilled=1 if m1_1_status == VerificationStatus.FULFILLED else 0,
            mandatory_total=2,
            optional_fulfilled=0,
            optional_total=0,
            sub_results=[res_1, res_2],
        )

        return DocumentComplianceReport(
            target_document_path="data/test_packs/A_target_lab_report.pdf",
            target_document_name="A_target_lab_report.pdf",
            template_title="Mock Reference Template",
            template_path="data/test_packs/A_reference_lab_report_template.pdf",
            total_parent_points=1,
            total_sub_requirements=2,
            parent_points=[parent_pt],
            mandatory_coverage_pct=50.0,
            optional_coverage_pct=None,
            total_mandatory=2,
            total_optional=0,
            mandatory_fulfilled=1,
            optional_fulfilled=0,
            fulfilled_count=1,
            explicit_omission_count=1,
            contradicted_count=0,
            missing_count=0,
            needs_review_count=0,
            verdict=verdict,
        )

    def test_compute_metrics_perfect_alignment(self):
        """Verifies compute_metrics when actual report perfectly matches manifest ground truth."""
        manifest = self._create_mock_manifest()
        report = self._create_mock_report()

        runner = BenchmarkRunner()
        eval_report = runner.compute_metrics(manifest, report, top_k=5)

        # Retrieval checks
        self.assertEqual(eval_report.retrieval_metrics.evaluated_count, 2)
        self.assertEqual(eval_report.retrieval_metrics.hit_at_1_pct, 50.0)  # M1.1 was rank 1, M1.2 was rank 2
        self.assertEqual(eval_report.retrieval_metrics.hit_at_3_pct, 100.0)
        self.assertEqual(eval_report.retrieval_metrics.hit_at_5_pct, 100.0)

        # Verification checks
        self.assertEqual(eval_report.verification_metrics.accuracy_pct, 100.0)
        self.assertEqual(eval_report.verification_metrics.macro_f1, 1.0)
        self.assertEqual(eval_report.verification_metrics.correct_predictions, 2)

        # Compliance checks
        self.assertEqual(eval_report.compliance_metrics.parent_accuracy_pct, 100.0)
        self.assertTrue(eval_report.compliance_metrics.verdict_matches)
        self.assertEqual(eval_report.compliance_metrics.mandatory_coverage_diff, 0.0)

        # Discrepancies
        self.assertEqual(len(eval_report.discrepancies), 0)

    def test_compute_metrics_detects_retrieval_miss(self):
        """Verifies that missing an expected evidence block is recorded as a RETRIEVAL_MISS discrepancy."""
        manifest = self._create_mock_manifest()
        # M1.1 retrieves wrong block
        report = self._create_mock_report(m1_1_block="p1_b99")

        runner = BenchmarkRunner()
        eval_report = runner.compute_metrics(manifest, report, top_k=5)

        # Hit@5 drops to 50%
        self.assertEqual(eval_report.retrieval_metrics.hit_at_5_pct, 50.0)

        # Discrepancy logged
        ret_disc = [d for d in eval_report.discrepancies if d.category == "RETRIEVAL_MISS"]
        self.assertEqual(len(ret_disc), 1)
        self.assertEqual(ret_disc[0].sub_id, "M1.1")

    def test_compute_metrics_detects_verification_mismatch(self):
        """Verifies that an incorrect verification status is recorded as a VERIFICATION_MISMATCH."""
        manifest = self._create_mock_manifest()
        # M1.1 predicts MISSING instead of FULFILLED
        report = self._create_mock_report(m1_1_status=VerificationStatus.MISSING)

        runner = BenchmarkRunner()
        eval_report = runner.compute_metrics(manifest, report, top_k=5)

        self.assertEqual(eval_report.verification_metrics.accuracy_pct, 50.0)
        self.assertEqual(eval_report.verification_metrics.correct_predictions, 1)

        v_disc = [d for d in eval_report.discrepancies if d.category == "VERIFICATION_MISMATCH"]
        self.assertEqual(len(v_disc), 1)
        self.assertEqual(v_disc[0].sub_id, "M1.1")
        self.assertEqual(v_disc[0].expected, "FULFILLED")
        self.assertEqual(v_disc[0].actual, "MISSING")

    def test_compute_metrics_detects_verdict_mismatch(self):
        """Verifies that an incorrect document verdict triggers a VERDICT_MISMATCH discrepancy."""
        manifest = self._create_mock_manifest()
        # Report claims COMPLIANT instead of NON_COMPLIANT
        report = self._create_mock_report(verdict=DocumentComplianceVerdict.COMPLIANT)

        runner = BenchmarkRunner()
        eval_report = runner.compute_metrics(manifest, report, top_k=5)

        self.assertFalse(eval_report.compliance_metrics.verdict_matches)
        v_disc = [d for d in eval_report.discrepancies if d.category == "VERDICT_MISMATCH"]
        self.assertEqual(len(v_disc), 1)
        self.assertEqual(v_disc[0].expected, "NON_COMPLIANT")
        self.assertEqual(v_disc[0].actual, "COMPLIANT")

    def test_formatting_contains_no_markdown_tables(self):
        """Verifies that neither terminal summary nor markdown report contain markdown table characters."""
        manifest = self._create_mock_manifest()
        report = self._create_mock_report()

        runner = BenchmarkRunner()
        eval_report = runner.compute_metrics(manifest, report, top_k=5)

        term_output = eval_report.format_terminal_summary()
        self.assertNotIn("|---|", term_output)
        self.assertIn("BENCHMARK ACCURACY EVALUATION", term_output)

        md_output = eval_report.to_markdown()
        self.assertNotIn("|---|", md_output)
        self.assertIn("# Benchmark Evaluation Report", md_output)

    def test_e2e_evaluate_pair_a(self):
        """Runs live end-to-end evaluation on production Pair A benchmark manifest."""
        runner = BenchmarkRunner()
        report_a = runner.evaluate_manifest(self.pair_a_path, base_dir=self.project_root, top_k=5)

        self.assertEqual(report_a.pair_id, "PAIR_A")
        self.assertEqual(report_a.verification_metrics.accuracy_pct, 100.0)
        self.assertEqual(report_a.verification_metrics.correct_predictions, 10)
        self.assertEqual(report_a.compliance_metrics.parent_accuracy_pct, 100.0)
        self.assertTrue(report_a.compliance_metrics.verdict_matches)
        self.assertEqual(report_a.compliance_metrics.actual_verdict, "NON_COMPLIANT")
        self.assertEqual(report_a.compliance_metrics.mandatory_coverage_diff, 0.0)
        self.assertEqual(len(report_a.discrepancies), 0)

    def test_e2e_evaluate_pair_b(self):
        """Runs live end-to-end evaluation on production Pair B benchmark manifest."""
        runner = BenchmarkRunner()
        report_b = runner.evaluate_manifest(self.pair_b_path, base_dir=self.project_root, top_k=5)

        self.assertEqual(report_b.pair_id, "PAIR_B")
        self.assertEqual(report_b.verification_metrics.accuracy_pct, 100.0)
        self.assertEqual(report_b.verification_metrics.correct_predictions, 12)
        self.assertEqual(report_b.compliance_metrics.parent_accuracy_pct, 100.0)
        self.assertTrue(report_b.compliance_metrics.verdict_matches)
        self.assertEqual(report_b.compliance_metrics.actual_verdict, "CONDITIONALLY_COMPLIANT")
        self.assertEqual(report_b.compliance_metrics.mandatory_coverage_diff, 0.0)
        self.assertEqual(len(report_b.discrepancies), 0)


if __name__ == "__main__":
    unittest.main()
