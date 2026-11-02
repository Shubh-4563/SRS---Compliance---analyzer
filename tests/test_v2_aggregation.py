"""
Unit tests for V2 Coverage Aggregation and Compliance Scoring (Phase 2.4).
Verifies:
1. All mandatory requirements fulfilled (FULLY_COMPLIANT -> COMPLIANT).
2. Some fulfilled and some missing (PARTIALLY_COMPLIANT -> CONDITIONALLY_COMPLIANT).
3. Explicit omissions (NON_COMPLIANT if no positive evidence).
4. All requirements missing (NON_COMPLIANT).
5. Contradicted requirements (NON_COMPLIANT with contradiction counts).
6. NEEDS_REVIEW propagation (Parent and Document escalate to NEEDS_REVIEW / CONDITIONALLY_COMPLIANT).
7. Optional requirements do not affect mandatory coverage.
8. Optional requirements all fulfilled without mandatory (PARTIALLY_COMPLIANT, no division by zero).
9. Parent with no mandatory requirements (handled gracefully).
10. Empty template (handled gracefully).
11. Multiple parents rollup.
12. Missing and duplicate verification results handled deterministically.
13. Document-level verdicts (COMPLIANT, CONDITIONALLY_COMPLIANT, NON_COMPLIANT).
14. Deterministic scoring without model inference.
"""
import unittest

from src.models.requirement import (
    EvidenceType,
    SubRequirement,
    DecomposedTemplatePoint,
    DecomposedReferenceTemplate,
)
from src.models.verification import (
    VerificationStatus,
    SubRequirementVerificationResult,
)
from src.models.compliance import (
    ParentComplianceStatus,
    DocumentComplianceVerdict,
    ParentPointCompliance,
    DocumentComplianceReport,
)
from src.verification.compliance_aggregator import ComplianceAggregator


class TestV2ComplianceAggregation(unittest.TestCase):
    """Test suite for deterministic compliance aggregation and scoring."""

    def setUp(self):
        self.aggregator = ComplianceAggregator()

    def _make_sub_req(self, sub_id: str, parent_id: str, is_mandatory: bool = True) -> SubRequirement:
        return SubRequirement(
            sub_id=sub_id,
            parent_point_id=parent_id,
            claim_text=f"Claim for {sub_id}",
            expected_evidence_type=EvidenceType.FACTUAL_STATEMENT,
            is_mandatory=is_mandatory,
            section_name="Section",
            topic_label="Topic",
            original_text="Original text"
        )

    def _make_result(
        self,
        sub_id: str,
        parent_id: str,
        status: VerificationStatus,
        is_mandatory: bool = True,
        confidence: float = 0.95,
        rationale: str = "Test rationale"
    ) -> SubRequirementVerificationResult:
        return SubRequirementVerificationResult(
            sub_id=sub_id,
            parent_point_id=parent_id,
            claim_text=f"Claim for {sub_id}",
            expected_evidence_type=EvidenceType.FACTUAL_STATEMENT,
            status=status,
            confidence=confidence,
            is_mandatory=is_mandatory,
            rationale=rationale,
            needs_review=(status == VerificationStatus.NEEDS_REVIEW)
        )

    def test_all_mandatory_fulfilled(self):
        """Rule A: All mandatory sub-requirements FULFILLED -> FULLY_COMPLIANT, document COMPLIANT."""
        sr1 = self._make_sub_req("P1.1", "P1", is_mandatory=True)
        sr2 = self._make_sub_req("P1.2", "P1", is_mandatory=True)
        point = DecomposedTemplatePoint("P1", "Sec", "Top", "Desc", "Raw", sub_requirements=[sr1, sr2])
        template = DecomposedReferenceTemplate("Template", points=[point])

        res1 = self._make_result("P1.1", "P1", VerificationStatus.FULFILLED, is_mandatory=True)
        res2 = self._make_result("P1.2", "P1", VerificationStatus.FULFILLED, is_mandatory=True)

        report = self.aggregator.aggregate(template, [res1, res2])

        self.assertEqual(len(report.parent_points), 1)
        p1 = report.parent_points[0]
        self.assertEqual(p1.status, ParentComplianceStatus.FULLY_COMPLIANT)
        self.assertEqual(p1.mandatory_fulfilled, 2)
        self.assertEqual(p1.mandatory_total, 2)
        self.assertEqual(report.mandatory_coverage_pct, 100.0)
        self.assertIsNone(report.optional_coverage_pct)
        self.assertEqual(report.verdict, DocumentComplianceVerdict.COMPLIANT)

    def test_some_fulfilled_some_missing(self):
        """Rule B: Some mandatory FULFILLED and some MISSING -> Parent PARTIALLY_COMPLIANT, Document NON_COMPLIANT (no reviews)."""
        sr1 = self._make_sub_req("P1.1", "P1", is_mandatory=True)
        sr2 = self._make_sub_req("P1.2", "P1", is_mandatory=True)
        point = DecomposedTemplatePoint("P1", "Sec", "Top", "Desc", "Raw", sub_requirements=[sr1, sr2])
        template = DecomposedReferenceTemplate("Template", points=[point])

        res1 = self._make_result("P1.1", "P1", VerificationStatus.FULFILLED, is_mandatory=True)
        res2 = self._make_result("P1.2", "P1", VerificationStatus.MISSING, is_mandatory=True)

        report = self.aggregator.aggregate(template, [res1, res2])

        p1 = report.parent_points[0]
        self.assertEqual(p1.status, ParentComplianceStatus.PARTIALLY_COMPLIANT)
        self.assertEqual(p1.mandatory_fulfilled, 1)
        self.assertEqual(p1.mandatory_total, 2)
        self.assertEqual(report.mandatory_coverage_pct, 50.0)
        self.assertEqual(report.verdict, DocumentComplianceVerdict.NON_COMPLIANT)

    def test_explicit_omissions(self):
        """Rule C: All mandatory EXPLICIT_OMISSION -> NON_COMPLIANT."""
        sr1 = self._make_sub_req("P1.1", "P1", is_mandatory=True)
        point = DecomposedTemplatePoint("P1", "Sec", "Top", "Desc", "Raw", sub_requirements=[sr1])
        template = DecomposedReferenceTemplate("Template", points=[point])

        res1 = self._make_result("P1.1", "P1", VerificationStatus.EXPLICIT_OMISSION, is_mandatory=True)

        report = self.aggregator.aggregate(template, [res1])

        p1 = report.parent_points[0]
        self.assertEqual(p1.status, ParentComplianceStatus.NON_COMPLIANT)
        self.assertEqual(p1.explicit_omission_count, 1)
        self.assertEqual(p1.missing_count, 0)
        self.assertEqual(report.mandatory_coverage_pct, 0.0)
        self.assertEqual(report.verdict, DocumentComplianceVerdict.NON_COMPLIANT)

    def test_all_requirements_missing(self):
        """Rule C: All mandatory MISSING -> NON_COMPLIANT."""
        sr1 = self._make_sub_req("P1.1", "P1", is_mandatory=True)
        point = DecomposedTemplatePoint("P1", "Sec", "Top", "Desc", "Raw", sub_requirements=[sr1])
        template = DecomposedReferenceTemplate("Template", points=[point])

        res1 = self._make_result("P1.1", "P1", VerificationStatus.MISSING, is_mandatory=True)

        report = self.aggregator.aggregate(template, [res1])

        p1 = report.parent_points[0]
        self.assertEqual(p1.status, ParentComplianceStatus.NON_COMPLIANT)
        self.assertEqual(p1.missing_count, 1)
        self.assertEqual(p1.explicit_omission_count, 0)
        self.assertEqual(report.mandatory_coverage_pct, 0.0)
        self.assertEqual(report.verdict, DocumentComplianceVerdict.NON_COMPLIANT)

    def test_contradicted_requirements(self):
        """Rule E: All mandatory CONTRADICTED -> NON_COMPLIANT while preserving contradiction counts."""
        sr1 = self._make_sub_req("P1.1", "P1", is_mandatory=True)
        point = DecomposedTemplatePoint("P1", "Sec", "Top", "Desc", "Raw", sub_requirements=[sr1])
        template = DecomposedReferenceTemplate("Template", points=[point])

        res1 = self._make_result("P1.1", "P1", VerificationStatus.CONTRADICTED, is_mandatory=True)

        report = self.aggregator.aggregate(template, [res1])

        p1 = report.parent_points[0]
        self.assertEqual(p1.status, ParentComplianceStatus.NON_COMPLIANT)
        self.assertEqual(p1.contradiction_count, 1)
        self.assertEqual(report.contradicted_count, 1)
        self.assertEqual(report.mandatory_coverage_pct, 0.0)
        self.assertEqual(report.verdict, DocumentComplianceVerdict.NON_COMPLIANT)

    def test_needs_review_propagation(self):
        """Rule D: Any mandatory sub-requirement NEEDS_REVIEW escalates parent and document."""
        sr1 = self._make_sub_req("P1.1", "P1", is_mandatory=True)
        sr2 = self._make_sub_req("P1.2", "P1", is_mandatory=True)
        point = DecomposedTemplatePoint("P1", "Sec", "Top", "Desc", "Raw", sub_requirements=[sr1, sr2])
        template = DecomposedReferenceTemplate("Template", points=[point])

        res1 = self._make_result("P1.1", "P1", VerificationStatus.FULFILLED, is_mandatory=True)
        res2 = self._make_result("P1.2", "P1", VerificationStatus.NEEDS_REVIEW, is_mandatory=True)

        report = self.aggregator.aggregate(template, [res1, res2])

        p1 = report.parent_points[0]
        self.assertEqual(p1.status, ParentComplianceStatus.NEEDS_REVIEW)
        self.assertEqual(report.verdict, DocumentComplianceVerdict.CONDITIONALLY_COMPLIANT)
        self.assertEqual(len(report.items_requiring_review), 1)
        self.assertEqual(report.items_requiring_review[0].sub_id, "P1.2")

    def test_optional_requirements_do_not_affect_mandatory_coverage(self):
        """Optional requirements must not alter mandatory coverage percentage."""
        sr_m1 = self._make_sub_req("P1.1", "P1", is_mandatory=True)
        sr_o1 = self._make_sub_req("P1.2", "P1", is_mandatory=False)
        point = DecomposedTemplatePoint("P1", "Sec", "Top", "Desc", "Raw", sub_requirements=[sr_m1, sr_o1])
        template = DecomposedReferenceTemplate("Template", points=[point])

        # Mandatory fulfilled, optional missing
        res_m1 = self._make_result("P1.1", "P1", VerificationStatus.FULFILLED, is_mandatory=True)
        res_o1 = self._make_result("P1.2", "P1", VerificationStatus.MISSING, is_mandatory=False)

        report = self.aggregator.aggregate(template, [res_m1, res_o1])

        p1 = report.parent_points[0]
        self.assertEqual(p1.status, ParentComplianceStatus.FULLY_COMPLIANT)
        self.assertEqual(report.mandatory_coverage_pct, 100.0)
        self.assertEqual(report.optional_coverage_pct, 0.0)
        self.assertEqual(p1.mandatory_fulfilled, 1)
        self.assertEqual(p1.optional_fulfilled, 0)
        self.assertEqual(report.verdict, DocumentComplianceVerdict.COMPLIANT)

    def test_optional_requirements_all_fulfilled_without_mandatory(self):
        """If parent has only optional requirements, do not divide by zero and do not classify as FULLY_COMPLIANT."""
        sr_o1 = self._make_sub_req("P1.1", "P1", is_mandatory=False)
        sr_o2 = self._make_sub_req("P1.2", "P1", is_mandatory=False)
        point = DecomposedTemplatePoint("P1", "Sec", "Top", "Desc", "Raw", sub_requirements=[sr_o1, sr_o2])
        template = DecomposedReferenceTemplate("Template", points=[point])

        res_o1 = self._make_result("P1.1", "P1", VerificationStatus.FULFILLED, is_mandatory=False)
        res_o2 = self._make_result("P1.2", "P1", VerificationStatus.FULFILLED, is_mandatory=False)

        report = self.aggregator.aggregate(template, [res_o1, res_o2])

        p1 = report.parent_points[0]
        self.assertNotEqual(p1.status, ParentComplianceStatus.FULLY_COMPLIANT)
        self.assertEqual(p1.status, ParentComplianceStatus.PARTIALLY_COMPLIANT)
        self.assertIsNone(report.mandatory_coverage_pct)
        self.assertEqual(report.optional_coverage_pct, 100.0)
        self.assertEqual(p1.mandatory_total, 0)
        self.assertEqual(p1.optional_total, 2)

    def test_parent_with_no_subrequirements(self):
        """Handles parent point with no sub-requirements without error."""
        point = DecomposedTemplatePoint("P1", "Sec", "Top", "Desc", "Raw", sub_requirements=[])
        template = DecomposedReferenceTemplate("Template", points=[point])

        report = self.aggregator.aggregate(template, [])

        p1 = report.parent_points[0]
        self.assertEqual(p1.status, ParentComplianceStatus.NON_COMPLIANT)
        self.assertIsNone(report.mandatory_coverage_pct)
        self.assertEqual(report.total_sub_requirements, 0)

    def test_empty_template(self):
        """Handles empty reference template gracefully."""
        template = DecomposedReferenceTemplate("Empty", points=[])
        report = self.aggregator.aggregate(template, [])

        self.assertEqual(report.total_parent_points, 0)
        self.assertEqual(report.total_sub_requirements, 0)
        self.assertIsNone(report.mandatory_coverage_pct)
        self.assertEqual(report.verdict, DocumentComplianceVerdict.NON_COMPLIANT)

    def test_multiple_parents_aggregation(self):
        """Rollup across multiple parent points computes correct document metrics."""
        sr1 = self._make_sub_req("A1.1", "A1", is_mandatory=True)
        sr2 = self._make_sub_req("A2.1", "A2", is_mandatory=True)
        sr3 = self._make_sub_req("A2.2", "A2", is_mandatory=True)

        p1 = DecomposedTemplatePoint("A1", "Sec1", "Top1", "D1", "R1", sub_requirements=[sr1])
        p2 = DecomposedTemplatePoint("A2", "Sec2", "Top2", "D2", "R2", sub_requirements=[sr2, sr3])
        template = DecomposedReferenceTemplate("Multi", points=[p1, p2])

        res1 = self._make_result("A1.1", "A1", VerificationStatus.FULFILLED)
        res2 = self._make_result("A2.1", "A2", VerificationStatus.FULFILLED)
        res3 = self._make_result("A2.2", "A2", VerificationStatus.EXPLICIT_OMISSION)

        report = self.aggregator.aggregate(template, [res1, res2, res3])

        self.assertEqual(len(report.parent_points), 2)
        self.assertEqual(report.parent_points[0].status, ParentComplianceStatus.FULLY_COMPLIANT)
        self.assertEqual(report.parent_points[1].status, ParentComplianceStatus.PARTIALLY_COMPLIANT)
        self.assertAlmostEqual(report.mandatory_coverage_pct, (2/3) * 100.0, places=2)
        self.assertEqual(report.verdict, DocumentComplianceVerdict.NON_COMPLIANT)

    def test_missing_and_duplicate_verification_results(self):
        """Missing results are marked MISSING; duplicate results are resolved deterministically."""
        sr1 = self._make_sub_req("P1.1", "P1", is_mandatory=True)
        sr2 = self._make_sub_req("P1.2", "P1", is_mandatory=True)
        point = DecomposedTemplatePoint("P1", "Sec", "Top", "Desc", "Raw", sub_requirements=[sr1, sr2])
        template = DecomposedReferenceTemplate("Template", points=[point])

        # Input provides duplicate for P1.1, and nothing for P1.2
        res1_old = self._make_result("P1.1", "P1", VerificationStatus.MISSING)
        res1_new = self._make_result("P1.1", "P1", VerificationStatus.FULFILLED)

        report = self.aggregator.aggregate(template, [res1_old, res1_new])

        self.assertEqual(report.parent_points[0].mandatory_fulfilled, 1)
        self.assertEqual(report.parent_points[0].missing_count, 1)  # P1.2 synthesized as MISSING
        self.assertEqual(report.parent_points[0].sub_results[0].status, VerificationStatus.FULFILLED)
        self.assertEqual(report.parent_points[0].sub_results[1].status, VerificationStatus.MISSING)

    def test_document_verdicts(self):
        """Verifies updated 3-tier verdicts: COMPLIANT, CONDITIONALLY_COMPLIANT, NON_COMPLIANT."""
        sr1 = self._make_sub_req("P1.1", "P1", is_mandatory=True)
        sr2 = self._make_sub_req("P1.2", "P1", is_mandatory=True)
        point = DecomposedTemplatePoint("P1", "Sec", "Top", "Desc", "Raw", sub_requirements=[sr1, sr2])
        template = DecomposedReferenceTemplate("Template", points=[point])

        # Case 1: 100% fulfilled, 0 review, 0 contradiction -> COMPLIANT
        r1 = self._make_result("P1.1", "P1", VerificationStatus.FULFILLED)
        r2 = self._make_result("P1.2", "P1", VerificationStatus.FULFILLED)
        rep1 = self.aggregator.aggregate(template, [r1, r2])
        self.assertEqual(rep1.verdict, DocumentComplianceVerdict.COMPLIANT)

        # Case 2: Pair A scenario: 1 fulfilled, 1 explicit omission, 0 review -> NON_COMPLIANT
        r_om = self._make_result("P1.2", "P1", VerificationStatus.EXPLICIT_OMISSION)
        rep2 = self.aggregator.aggregate(template, [r1, r_om])
        self.assertEqual(rep2.mandatory_coverage_pct, 50.0)
        self.assertEqual(rep2.needs_review_count, 0)
        self.assertEqual(rep2.verdict, DocumentComplianceVerdict.NON_COMPLIANT)

        # Case 3: Pair B scenario: 1 fulfilled, 1 needs review -> CONDITIONALLY_COMPLIANT
        r_rev = self._make_result("P1.2", "P1", VerificationStatus.NEEDS_REVIEW)
        rep3 = self.aggregator.aggregate(template, [r1, r_rev])
        self.assertEqual(rep3.needs_review_count, 1)
        self.assertEqual(rep3.verdict, DocumentComplianceVerdict.CONDITIONALLY_COMPLIANT)

        # Case 4: 0% fulfilled, 0 review -> NON_COMPLIANT
        r_mis1 = self._make_result("P1.1", "P1", VerificationStatus.MISSING)
        r_mis2 = self._make_result("P1.2", "P1", VerificationStatus.MISSING)
        rep4 = self.aggregator.aggregate(template, [r_mis1, r_mis2])
        self.assertEqual(rep4.verdict, DocumentComplianceVerdict.NON_COMPLIANT)

        # Case 5: 1 fulfilled, 1 contradiction, 0 review -> NON_COMPLIANT
        r_contra = self._make_result("P1.2", "P1", VerificationStatus.CONTRADICTED)
        rep5 = self.aggregator.aggregate(template, [r1, r_contra])
        self.assertEqual(rep5.verdict, DocumentComplianceVerdict.NON_COMPLIANT)


if __name__ == "__main__":
    unittest.main()

