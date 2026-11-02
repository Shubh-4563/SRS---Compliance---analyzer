"""
Unit and targeted tests for V2 NLI-based Evidence Verification (Phase 2.3).
Verifies:
1. NLI entailment classification.
2. Contradiction classification.
3. Neutral and irrelevant evidence handling.
4. Explicit omission disclaimer detection.
5. Numeric target evidence verification with token presence checking.
6. Conflicting evidence detection (NEEDS_REVIEW).
7. Traceability and provenance preservation.
Cross-platform compatible (macOS and Windows).
"""
import os
import unittest

from src.models.document import BlockType, DocumentBlock
from src.models.requirement import EvidenceType, SubRequirement
from src.models.retrieval import CandidateEvidence
from src.models.verification import (
    VerificationStatus,
    EvidenceEvaluation,
    SubRequirementVerificationResult,
)
from src.verification.nli_verifier import NLIVerifier
from src.verification.evidence_verifier import EvidenceVerifier


class TestV2EvidenceVerification(unittest.TestCase):
    """Targeted tests for NLI Verification Engine."""

    @classmethod
    def setUpClass(cls):
        # Initialize verifier with offline local cache
        cls.nli = NLIVerifier(local_files_only=True)
        cls.verifier = EvidenceVerifier(cls.nli)

    def _make_candidate(
        self,
        block_id: str,
        text: str,
        is_omission: bool = False,
        block_type: BlockType = BlockType.PARAGRAPH
    ) -> CandidateEvidence:
        block = DocumentBlock(
            block_id=block_id,
            text=text,
            block_type=block_type,
            page_number=1,
            section_heading="Test Section",
            is_explicit_omission=is_omission
        )
        return CandidateEvidence.from_block(
            block=block,
            rrf_score=0.03,
            rrf_rank=1
        )

    def test_nli_entailment_classification(self):
        """Verifies that factually supporting evidence yields high entailment."""
        sub_req = SubRequirement(
            sub_id="T1.1",
            parent_point_id="T1",
            claim_text="The application collects readings from connected sensors and displays current status.",
            expected_evidence_type=EvidenceType.FACTUAL_STATEMENT
        )
        cand = self._make_candidate(
            "p1_b5",
            "The application collects readings from connected sensors, displays current status, and stores historical measurements."
        )

        res = self.verifier.verify_subrequirement(sub_req, [cand])
        self.assertEqual(res.status, VerificationStatus.FULFILLED)
        self.assertGreaterEqual(res.confidence, 0.50)
        self.assertIsNotNone(res.best_evidence)
        self.assertEqual(res.best_evidence.predicted_label, "entailment")

    def test_nli_contradiction_classification(self):
        """Verifies that contradictory evidence yields CONTRADICTED status."""
        sub_req = SubRequirement(
            sub_id="T2.1",
            parent_point_id="T2",
            claim_text="The dashboard directly controls the equipment's physical operation.",
            expected_evidence_type=EvidenceType.NEGATIVE_BOUNDARY
        )
        cand = self._make_candidate(
            "p1_b3",
            "It is designed for the monitoring station and does not control the equipment's physical operation."
        )

        res = self.verifier.verify_subrequirement(sub_req, [cand])
        self.assertEqual(res.status, VerificationStatus.CONTRADICTED)
        self.assertGreaterEqual(res.confidence, 0.50)
        self.assertEqual(res.best_evidence.predicted_label, "contradiction")

    def test_nli_neutral_and_irrelevant_evidence(self):
        """Verifies that irrelevant candidate evidence produces neutral probability and MISSING status."""
        sub_req = SubRequirement(
            sub_id="A2.1",
            parent_point_id="A2",
            claim_text="The document lists the equipment, instruments, or apparatus used in the experiment.",
            expected_evidence_type=EvidenceType.LIST_OF_ITEMS
        )
        # Irrelevant conclusion candidate (discusses cooling process, neutral regarding equipment)
        cand = self._make_candidate(
            "p1_b13",
            "The water cooled during the ten-minute observation, so the experiment demonstrated the intended cooling process."
        )

        res = self.verifier.verify_subrequirement(sub_req, [cand])
        self.assertEqual(res.status, VerificationStatus.MISSING)
        self.assertEqual(res.best_evidence.predicted_label, "neutral")
        self.assertGreaterEqual(res.best_evidence.neutral_prob, 0.70)

    def test_explicit_omission_handling(self):
        """Verifies that disclaimer blocks with is_explicit_omission produce EXPLICIT_OMISSION."""
        sub_req = SubRequirement(
            sub_id="A7.1",
            parent_point_id="A7",
            claim_text="The document identifies laboratory hazards, safety risks, or precautions taken during the experiment.",
            expected_evidence_type=EvidenceType.FACTUAL_STATEMENT
        )
        disclaimer = self._make_candidate(
            "p1_b14",
            "No specific hazards or safety precautions are discussed in this report.",
            is_omission=True,
            block_type=BlockType.DISCLAIMER
        )

        res = self.verifier.verify_subrequirement(sub_req, [disclaimer])
        self.assertEqual(res.status, VerificationStatus.EXPLICIT_OMISSION)
        self.assertGreaterEqual(res.confidence, 0.90)
        self.assertEqual(res.best_evidence.predicted_label, "contradiction")

    def test_numeric_target_validation(self):
        """Verifies that NUMERIC_TARGET requirements require concrete numbers and units."""
        numeric_subreq = SubRequirement(
            sub_id="A4.1",
            parent_point_id="A4",
            claim_text="The document records measured numerical values, data points, or observations collected during the experiment.",
            expected_evidence_type=EvidenceType.NUMERIC_TARGET
        )

        # 1. Candidate with actual numbers and units -> FULFILLED
        cand_with_numbers = self._make_candidate(
            "p1_b9",
            "The recorded temperatures were: start 70 degrees C; 2 minutes 61 degrees C; 4 minutes 54 degrees C.",
            block_type=BlockType.TABLE_ROW
        )
        res_num = self.verifier.verify_subrequirement(numeric_subreq, [cand_with_numbers])
        self.assertEqual(res_num.status, VerificationStatus.FULFILLED)
        self.assertTrue(res_num.best_evidence.has_numeric_content)
        self.assertIn("70", res_num.best_evidence.numeric_tokens[0])

        # 2. Candidate with narrative statement but lacking numbers -> NEEDS_REVIEW
        cand_prose_no_numbers = self._make_candidate(
            "p1_b13",
            "The water cooled during the observation, so the experiment demonstrated the intended cooling process."
        )
        # Force an evaluation where NLI entails narrative text but has_numeric_content is False
        eval_no_num = EvidenceEvaluation(
            candidate=cand_prose_no_numbers,
            premise_text=cand_prose_no_numbers.text,
            hypothesis_text=numeric_subreq.claim_text,
            entailment_prob=0.85,
            contradiction_prob=0.01,
            neutral_prob=0.14,
            predicted_label="entailment",
            has_numeric_content=False,
            numeric_tokens=[]
        )
        # Use an isolated mock verifier without mutating class state
        class MockNLI:
            def evaluate_candidates(self, candidates, sub_req):
                return [eval_no_num]

        verifier_custom = EvidenceVerifier(MockNLI())
        res_prose = verifier_custom.verify_subrequirement(numeric_subreq, [cand_prose_no_numbers])
        self.assertEqual(res_prose.status, VerificationStatus.NEEDS_REVIEW)
        self.assertTrue(res_prose.needs_review)
        self.assertIn("lacks concrete measurable numeric values", res_prose.rationale)

    def test_conflicting_evidence_detection(self):
        """Verifies that competing entailment and contradiction candidates trigger NEEDS_REVIEW."""
        sub_req = SubRequirement(
            sub_id="B4.3",
            parent_point_id="B4",
            claim_text="The document specifies communication protocols for data exchange.",
            expected_evidence_type=EvidenceType.FACTUAL_STATEMENT
        )

        # Candidate 1: Claims protocol is supported
        c1 = self._make_candidate("p1_b7", "The system specifies standard TCP/IP communication protocol for data exchange.")
        # Candidate 2: Claims protocol is not specified
        c2 = self._make_candidate("p1_b8", "The document does not specify the communication protocol.", is_omission=False)

        res = self.verifier.verify_subrequirement(sub_req, [c1, c2])
        self.assertEqual(res.status, VerificationStatus.NEEDS_REVIEW)
        self.assertTrue(res.needs_review)
        self.assertIn("Conflicting evidence detected", res.rationale)

    def test_subrequirement_traceability_preservation(self):
        """Verifies that verification results preserve complete provenance back to the source template."""
        sub_req = SubRequirement(
            sub_id="B1.1",
            parent_point_id="B1",
            claim_text="The document explains what the system or product is intended to do.",
            expected_evidence_type=EvidenceType.FACTUAL_STATEMENT,
            source_page=1,
            section_name="Purpose and Intended Use",
            topic_label="Purpose",
            original_text="Explain what the system or product is intended to do and the problem it solves."
        )
        cand = self._make_candidate(
            "p1_b3",
            "The dashboard helps maintenance staff monitor equipment condition and identify degraded operation early."
        )

        res = self.verifier.verify_subrequirement(sub_req, [cand])
        self.assertEqual(res.sub_id, "B1.1")
        self.assertEqual(res.parent_point_id, "B1")
        self.assertEqual(res.source_page, 1)
        self.assertEqual(res.section_name, "Purpose and Intended Use")
        self.assertEqual(res.topic_label, "Purpose")
        self.assertEqual(res.original_text, sub_req.original_text)
        self.assertEqual(len(res.all_evaluations), 1)

        # Verify serialization
        res_dict = res.to_dict()
        self.assertEqual(res_dict["status"], "FULFILLED")
        self.assertEqual(res_dict["best_evidence"]["block_id"], "p1_b3")

    def test_negative_boundary_clause_isolation(self):
        """Phase 2.3.1 Issue 1: Boundary clauses (does not control) must not contradict positive claims."""
        sub_req = SubRequirement(
            sub_id="B3.1",
            parent_point_id="B3",
            claim_text="The document describes the main functional actions the application or system performs.",
            expected_evidence_type=EvidenceType.FACTUAL_STATEMENT
        )
        cand_with_boundary = self._make_candidate(
            "p1_b3",
            "The dashboard helps maintenance staff monitor equipment condition and identify developing problems. "
            "It is designed for the monitoring station and does not control the equipment's physical operation."
        )

        res = self.verifier.verify_subrequirement(sub_req, [cand_with_boundary])
        self.assertEqual(res.status, VerificationStatus.FULFILLED)
        self.assertGreaterEqual(res.confidence, 0.50)
        self.assertLess(res.best_evidence.contradiction_prob, 0.30)

    def test_meta_discourse_sensitivity_resolution(self):
        """Phase 2.3.1 Issue 2: Meta-discourse ('The document explains...') recognizes substantive evidence."""
        sub_req = SubRequirement(
            sub_id="B1.2",
            parent_point_id="B1",
            claim_text="The document explains the problem the system solves or its operational purpose for users.",
            expected_evidence_type=EvidenceType.FACTUAL_STATEMENT
        )
        cand = self._make_candidate(
            "p1_b3",
            "The dashboard helps maintenance staff monitor equipment condition and identify developing problems before they interrupt daily operations."
        )

        res = self.verifier.verify_subrequirement(sub_req, [cand])
        self.assertEqual(res.status, VerificationStatus.FULFILLED)
        self.assertGreaterEqual(res.confidence, 0.70)
        self.assertEqual(res.best_evidence.predicted_label, "entailment")

    def test_b1_2_false_contradiction_resolution(self):
        """Phase 2.3.1 Issue 3: Fix B1.2 classified as FULFILLED, resolving the false contradiction."""
        sub_req = SubRequirement(
            sub_id="B1.2",
            parent_point_id="B1",
            claim_text="The document explains the problem the system solves or its operational purpose for users.",
            expected_evidence_type=EvidenceType.FACTUAL_STATEMENT
        )
        cand_p1_b3 = self._make_candidate(
            "p1_b3",
            "The dashboard helps maintenance staff monitor equipment condition and identify developing problems before they interrupt daily operations. "
            "It is designed for the monitoring station and does not control the equipment's physical operation."
        )

        res = self.verifier.verify_subrequirement(sub_req, [cand_p1_b3])
        self.assertNotEqual(res.status, VerificationStatus.CONTRADICTED)
        self.assertEqual(res.status, VerificationStatus.FULFILLED)
        self.assertGreaterEqual(res.confidence, 0.70)

    def test_explicit_omission_decision_rules_consistency(self):
        """Phase 2.3.1 Issue 4: Consistent decision rules for explicit omissions and partial coverage."""
        # 1. Explicit omission with NO positive evidence -> EXPLICIT_OMISSION (B4.3, B6.1)
        sub_req_b43 = SubRequirement(
            sub_id="B4.3",
            parent_point_id="B4",
            claim_text="The document specifies communication protocols or technical interface details for data exchange.",
            expected_evidence_type=EvidenceType.FACTUAL_STATEMENT,
            topic_label="Interfaces"
        )
        disclaimer_b43 = self._make_candidate(
            "p1_b8",
            "The document does not specify the communication protocol or list the full set of external software interfaces.",
            is_omission=True,
            block_type=BlockType.DISCLAIMER
        )
        res_omission = self.verifier.verify_subrequirement(sub_req_b43, [disclaimer_b43])
        self.assertEqual(res_omission.status, VerificationStatus.EXPLICIT_OMISSION)
        self.assertGreaterEqual(res_omission.confidence, 0.90)

        # 2. Positive evidence covering only part of requirement + disclaimer -> NEEDS_REVIEW
        cand_hazard = self._make_candidate(
            "p1_b10",
            "If a sensor stops sending readings, the dashboard displays a warning and marks that sensor status as unavailable."
        )
        disclaimer_hazard = self._make_candidate(
            "p1_b11",
            "The document does not explain recovery steps or other hazard responses.",
            is_omission=True,
            block_type=BlockType.DISCLAIMER
        )
        sub_req_b51 = SubRequirement(
            sub_id="B5.1",
            parent_point_id="B5",
            claim_text="The document describes hazards, faults, or failure conditions and how the system detects or warns of them.",
            expected_evidence_type=EvidenceType.FACTUAL_STATEMENT,
            topic_label="Safety"
        )
        res_partial = self.verifier.verify_subrequirement(sub_req_b51, [cand_hazard, disclaimer_hazard])
        self.assertEqual(res_partial.status, VerificationStatus.NEEDS_REVIEW)
        self.assertTrue(res_partial.needs_review)


if __name__ == "__main__":
    unittest.main()
