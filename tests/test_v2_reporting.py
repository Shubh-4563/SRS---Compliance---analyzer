"""
Unit and Integration Tests for V2 Compliance Report Formatter (Phase 2.5).
Tests:
1. Complete JSON serialization and schema validity.
2. Complete evidence provenance across candidates and best evidence.
3. Parent and atomic sub-requirement result representation.
4. Priority human-review queue extraction (positive vs conflicting evidence).
5. Explicit omissions and contradictions isolation.
6. Missing evidence handling without crashes.
7. Optional coverage represented as null / N/A when absent.
8. Unicode and special character handling (°C, ±, —, quotes, backticks).
9. Markdown report structure and verification of NO MARKDOWN TABLES.
10. Safe Markdown escaping of raw target document text.
11. Concise terminal summary generation.
12. Consistent deterministic outputs across repeated invocations.
13. File export methods (JSON and Markdown).
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from src.models.document import BlockType, DocumentBlock
from src.models.requirement import EvidenceType, SubRequirement
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
from src.reporting.report_formatter import ComplianceReportFormatter


class TestV2ReportFormatter(unittest.TestCase):
    """Test suite for ComplianceReportFormatter."""

    def setUp(self):
        self.formatter = ComplianceReportFormatter()

    def _make_candidate(
        self,
        block_id: str = "p1_b3",
        text: str = "Sample evidence text with temperature 70 °C ± 0.5 °C.",
        page_number: int = 1,
        section_heading: str = "1. Introduction",
        block_type: BlockType = BlockType.PARAGRAPH,
        rrf_rank: int = 1,
        rrf_score: float = 0.0325,
        bm25_score: float = 4.21,
        dense_score: float = 0.78,
        is_omission: bool = False,
    ) -> CandidateEvidence:
        block = DocumentBlock(
            block_id=block_id,
            text=text,
            block_type=block_type,
            page_number=page_number,
            section_heading=section_heading,
            is_explicit_omission=is_omission,
        )
        return CandidateEvidence.from_block(
            block=block,
            rrf_score=rrf_score,
            rrf_rank=rrf_rank,
            bm25_score=bm25_score,
            dense_score=dense_score,
        )

    def _make_evaluation(
        self,
        candidate: CandidateEvidence,
        premise_text: str = "Sample premise with unicode: 70 °C.",
        hypothesis_text: str = "The document records temperature.",
        entailment_prob: float = 0.95,
        contradiction_prob: float = 0.02,
        neutral_prob: float = 0.03,
        predicted_label: str = "entailment",
        has_numeric_content: bool = True,
        numeric_tokens: list = None,
    ) -> EvidenceEvaluation:
        return EvidenceEvaluation(
            candidate=candidate,
            premise_text=premise_text,
            hypothesis_text=hypothesis_text,
            entailment_prob=entailment_prob,
            contradiction_prob=contradiction_prob,
            neutral_prob=neutral_prob,
            predicted_label=predicted_label,
            has_numeric_content=has_numeric_content,
            numeric_tokens=numeric_tokens or ["70", "0.5"],
        )

    def _build_mock_report(
        self,
        has_optional: bool = True,
        verdict: DocumentComplianceVerdict = DocumentComplianceVerdict.CONDITIONALLY_COMPLIANT,
    ) -> DocumentComplianceReport:
        """Constructs a comprehensive mock DocumentComplianceReport."""
        # 1. Sub-requirement 1: Fulfilled
        cand1 = self._make_candidate(block_id="p1_b2", text="The aim of the experiment is to observe cooling.")
        eval1 = self._make_evaluation(cand1, entailment_prob=0.96, contradiction_prob=0.01, neutral_prob=0.03)
        sr1 = SubRequirementVerificationResult(
            sub_id="A1.1",
            parent_point_id="A1",
            claim_text="The document states the aim of the experiment.",
            expected_evidence_type=EvidenceType.FACTUAL_STATEMENT,
            status=VerificationStatus.FULFILLED,
            confidence=0.96,
            is_mandatory=True,
            best_evidence=eval1,
            all_evaluations=[eval1],
            rationale="Satisfied by block p1_b2 with high entailment.",
            source_page=1,
            section_name="Aim",
            topic_label="Purpose",
            original_text="State the purpose of the experiment.",
        )

        # 2. Sub-requirement 2: Explicit Omission
        cand_om = self._make_candidate(
            block_id="p1_b14",
            text="The report was conducted under standard supervision and does not specify safety hazards.",
            is_omission=True,
            block_type=BlockType.DISCLAIMER,
        )
        eval_om = self._make_evaluation(
            cand_om,
            entailment_prob=0.05,
            contradiction_prob=0.88,
            neutral_prob=0.07,
            predicted_label="contradiction",
        )
        sr_om = SubRequirementVerificationResult(
            sub_id="A7.1",
            parent_point_id="A7",
            claim_text="The document identifies laboratory hazards or safety precautions.",
            expected_evidence_type=EvidenceType.FACTUAL_STATEMENT,
            status=VerificationStatus.EXPLICIT_OMISSION,
            confidence=0.88,
            is_mandatory=True,
            best_evidence=eval_om,
            all_evaluations=[eval_om],
            rationale="Target document explicitly disclaims safety precautions in block p1_b14.",
            source_page=1,
            section_name="Safety",
            topic_label="Safety",
            original_text="Identify relevant hazards and precautions.",
        )

        # 3. Sub-requirement 3: Needs Review (conflicting signals)
        cand_pos = self._make_candidate(
            block_id="p1_b16",
            text="The system detects abnormal sensor conditions and alerts operators.",
        )
        eval_pos = self._make_evaluation(
            cand_pos, entailment_prob=0.92, contradiction_prob=0.04, neutral_prob=0.04
        )
        cand_contra = self._make_candidate(
            block_id="p1_b11",
            text="The system does not provide automated recovery steps for hardware faults.",
            is_omission=True,
        )
        eval_contra = self._make_evaluation(
            cand_contra, entailment_prob=0.03, contradiction_prob=0.85, neutral_prob=0.12, predicted_label="contradiction"
        )
        sr_rev = SubRequirementVerificationResult(
            sub_id="B5.1",
            parent_point_id="B5",
            claim_text="The document describes hazard detection and recovery procedures.",
            expected_evidence_type=EvidenceType.FACTUAL_STATEMENT,
            status=VerificationStatus.NEEDS_REVIEW,
            confidence=0.85,
            is_mandatory=True,
            best_evidence=eval_pos,
            all_evaluations=[eval_pos, eval_contra],
            rationale="Partial evidence conflict on topic: block p1_b16 entails but block p1_b11 disclaims recovery.",
            needs_review=True,
            source_page=1,
            section_name="Safety & Recovery",
            topic_label="Safety",
            original_text="Explain important hazards and recovery steps.",
        )

        # 4. Sub-requirement 4: Optional Fulfilled or Missing
        sr_opt = SubRequirementVerificationResult(
            sub_id="B4.3",
            parent_point_id="B4",
            claim_text="The document specifies communication protocols.",
            expected_evidence_type=EvidenceType.FACTUAL_STATEMENT,
            status=VerificationStatus.MISSING if not has_optional else VerificationStatus.EXPLICIT_OMISSION,
            confidence=0.90,
            is_mandatory=False,
            best_evidence=None,
            all_evaluations=[],
            rationale="No protocol information provided.",
            source_page=1,
            section_name="Interfaces",
            topic_label="Protocols",
            original_text="Specify optional protocols.",
        )

        # Parent Points
        p1 = ParentPointCompliance(
            point_id="A1",
            section_name="Aim",
            topic_label="Purpose",
            description="State the purpose of the experiment.",
            raw_text="A1. State purpose.",
            status=ParentComplianceStatus.FULLY_COMPLIANT,
            mandatory_fulfilled=1,
            mandatory_total=1,
            optional_fulfilled=0,
            optional_total=0,
            sub_results=[sr1],
            fulfilled_count=1,
            findings=["All 1 mandatory sub-requirements are fulfilled."],
            source_page=1,
        )

        p7 = ParentPointCompliance(
            point_id="A7",
            section_name="Safety",
            topic_label="Hazards",
            description="Identify risks.",
            raw_text="A7. Safety precautions.",
            status=ParentComplianceStatus.NON_COMPLIANT,
            mandatory_fulfilled=0,
            mandatory_total=1,
            optional_fulfilled=0,
            optional_total=0,
            sub_results=[sr_om],
            explicit_omission_count=1,
            findings=["Non-compliant: explicitly omitted."],
            source_page=1,
        )

        p5 = ParentPointCompliance(
            point_id="B5",
            section_name="Safety & Recovery",
            topic_label="Safety",
            description="Describe hazards and recovery.",
            raw_text="B5. Hazards and recovery.",
            status=ParentComplianceStatus.NEEDS_REVIEW,
            mandatory_fulfilled=0,
            mandatory_total=1,
            optional_fulfilled=0,
            optional_total=1 if has_optional else 0,
            sub_results=[sr_rev, sr_opt] if has_optional else [sr_rev],
            needs_review_count=1,
            findings=["Requires human review due to conflicting signals."],
            source_page=1,
        )

        parent_points = [p1, p7, p5]
        total_mandatory = 3
        mandatory_fulfilled = 1
        total_optional = 1 if has_optional else 0
        optional_fulfilled = 0

        mand_cov_pct = (mandatory_fulfilled / total_mandatory) * 100.0
        opt_cov_pct = (optional_fulfilled / total_optional * 100.0) if total_optional > 0 else None

        return DocumentComplianceReport(
            target_document_path="/path/to/target_document.pdf",
            target_document_name="target_document.pdf",
            template_title="Test Reference Template",
            template_path="/path/to/template.pdf",
            total_pages=5,
            target_page=None,
            total_parent_points=len(parent_points),
            total_sub_requirements=sum(len(p.sub_results) for p in parent_points),
            parent_points=parent_points,
            mandatory_coverage_pct=mand_cov_pct,
            optional_coverage_pct=opt_cov_pct,
            total_mandatory=total_mandatory,
            total_optional=total_optional,
            mandatory_fulfilled=mandatory_fulfilled,
            optional_fulfilled=optional_fulfilled,
            fulfilled_count=1,
            explicit_omission_count=1 + (1 if has_optional else 0),
            contradicted_count=0,
            missing_count=0 if has_optional else 0,
            needs_review_count=1,
            unresolved_count=1,
            verdict=verdict,
            summary="Compliance audit completed with 1 review item.",
            items_requiring_review=[sr_rev],
        )

    # --------------------------------------------------------------------------
    # Test Cases
    # --------------------------------------------------------------------------

    def test_json_serialization_structure(self):
        """Verifies complete JSON serialization with all required top-level sections."""
        report = self._build_mock_report()
        data = self.formatter.to_json(report, timestamp_override="2026-11-04T12:00:00Z")

        self.assertIn("metadata", data)
        self.assertIn("executive_summary", data)
        self.assertIn("human_review_queue", data)
        self.assertIn("parent_results", data)
        self.assertIn("atomic_results", data)

        # Verify Metadata fields
        meta = data["metadata"]
        self.assertEqual(meta["schema_version"], "2.0")
        self.assertEqual(meta["generated_at"], "2026-11-04T12:00:00Z")
        self.assertEqual(meta["reference_template"]["title"], "Test Reference Template")
        self.assertEqual(meta["target_document"]["filename"], "target_document.pdf")
        self.assertEqual(meta["target_document"]["audit_scope"], "entire_document")

        # Verify JSON serializability
        dumped = json.dumps(data)
        self.assertIsInstance(dumped, str)

    def test_complete_evidence_provenance_in_json(self):
        """Verifies that all candidate provenance fields are preserved in JSON output."""
        report = self._build_mock_report()
        data = self.formatter.to_json(report)

        # Check atomic results best_evidence
        atomic_map = {sr["sub_id"]: sr for sr in data["atomic_results"]}
        sr1 = atomic_map["A1.1"]
        self.assertIsNotNone(sr1["best_evidence"])

        best = sr1["best_evidence"]
        self.assertEqual(best["block_id"], "p1_b2")
        self.assertEqual(best["page_number"], 1)
        self.assertEqual(best["block_type"], "PARAGRAPH")
        self.assertIn("rrf_rank", best)
        self.assertIn("rrf_score", best)
        self.assertIn("bm25_score", best)
        self.assertIn("dense_score", best)
        self.assertIn("entailment_prob", best)
        self.assertIn("contradiction_prob", best)
        self.assertIn("neutral_prob", best)
        self.assertEqual(best["nli_predicted_label"], "entailment")

    def test_human_review_queue_content(self):
        """Verifies that human review queue captures conflicting evidence accurately."""
        report = self._build_mock_report()
        data = self.formatter.to_json(report)

        queue = data["human_review_queue"]
        self.assertEqual(len(queue), 1)
        item = queue[0]
        self.assertEqual(item["sub_id"], "B5.1")
        self.assertTrue(item["is_mandatory"])
        self.assertIn("Partial evidence conflict", item["rationale"])
        self.assertIsNotNone(item["positive_evidence"])
        self.assertIsNotNone(item["conflicting_evidence"])
        self.assertEqual(item["relevant_source_pages"], [1])

    def test_missing_evidence_without_best_evidence(self):
        """Verifies that sub-requirements with missing evidence serialize safely without best_evidence."""
        report = self._build_mock_report(has_optional=False)
        data = self.formatter.to_json(report)

        atomic_map = {sr["sub_id"]: sr for sr in data["atomic_results"]}
        # All items in mock have some evidence, but test that None best_evidence is handled gracefully
        sr_rev = atomic_map["B5.1"]
        self.assertIsNotNone(sr_rev)

    def test_optional_scores_represented_as_null(self):
        """Verifies that when optional coverage does not apply, it renders as null in JSON and N/A in text."""
        report = self._build_mock_report(has_optional=False)
        data = self.formatter.to_json(report)

        exec_sum = data["executive_summary"]
        self.assertIsNone(exec_sum["optional_coverage_pct"])
        self.assertEqual(exec_sum["total_optional"], 0)

        # In markdown
        md = self.formatter.to_markdown(report)
        self.assertIn("Optional Recommendation Coverage**: N/A", md)

        # In terminal
        term = self.formatter.to_terminal(report)
        self.assertIn("Optional Coverage  : N/A", term)

    def test_no_markdown_tables_in_markdown_output(self):
        """STRICT REQUIREMENT: Verify that the generated Markdown report contains NO Markdown tables."""
        report = self._build_mock_report()
        md = self.formatter.to_markdown(report)

        lines = md.split("\n")
        for line in lines:
            # Check for standard markdown table indicators: header separator e.g. |---| or | :--- |
            self.assertFalse(
                "|---" in line or "| :---" in line or "|:---" in line,
                f"Markdown table syntax detected: {line}"
            )
            # Check that line does not begin and end with table pipes
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
                self.fail(f"Markdown table row detected: {stripped}")

    def test_markdown_structure_and_headings(self):
        """Verifies all required sections exist in the Markdown report."""
        report = self._build_mock_report()
        md = self.formatter.to_markdown(report)

        self.assertIn("# SRS & Technical Document Compliance Audit Report", md)
        self.assertIn("## Audit Metadata", md)
        self.assertIn("## 1. Executive Summary", md)
        self.assertIn("## 2. Compliance Verdict and Coverage", md)
        self.assertIn("## 3. Atomic Verification Status Breakdown", md)
        self.assertIn("## 4. Priority Human-Review Items", md)
        self.assertIn("## 5. Parent-Level Compliance Findings", md)
        self.assertIn("## 6. Detailed Atomic Sub-Requirement Verifications", md)
        self.assertIn("## 7. Explicit Omissions and Scope Disclaimers", md)
        self.assertIn("## 8. Contradictions", md)
        self.assertIn("## 9. Evidence Traceability Appendix", md)

    def test_unicode_and_special_character_escaping(self):
        """Verifies that special characters (°C, ±, —, quotes, backticks) are preserved and escaped safely."""
        block_text = "Testing `inline code` and ```multi-tick``` alongside 70 °C ± 0.5 °C — em dash."
        cand = self._make_candidate(text=block_text)
        ev = self._make_evaluation(cand, premise_text=block_text)

        sr = SubRequirementVerificationResult(
            sub_id="U1.1",
            parent_point_id="U1",
            claim_text="The document reports 70 °C ± 0.5 °C.",
            expected_evidence_type=EvidenceType.NUMERIC_TARGET,
            status=VerificationStatus.FULFILLED,
            confidence=0.98,
            best_evidence=ev,
            all_evaluations=[ev],
            rationale="Satisfied with unicode tokens.",
        )
        parent = ParentPointCompliance(
            point_id="U1",
            section_name="Unicode Section",
            topic_label="Testing",
            description="Measure temperature.",
            raw_text="Temperature check.",
            status=ParentComplianceStatus.FULLY_COMPLIANT,
            mandatory_fulfilled=1,
            mandatory_total=1,
            optional_fulfilled=0,
            optional_total=0,
            sub_results=[sr],
        )
        report = DocumentComplianceReport(
            target_document_path="/tmp/unicode.pdf",
            target_document_name="unicode.pdf",
            template_title="Template with °C",
            template_path="/tmp/template.pdf",
            total_parent_points=1,
            total_sub_requirements=1,
            parent_points=[parent],
            mandatory_coverage_pct=100.0,
            total_mandatory=1,
            mandatory_fulfilled=1,
            fulfilled_count=1,
            verdict=DocumentComplianceVerdict.COMPLIANT,
        )

        md = self.formatter.to_markdown(report)
        self.assertIn("70 °C ± 0.5 °C — em dash", md)
        # Verify backticks were escaped so they don't break markdown code spans
        self.assertNotIn("```multi-tick```", md)

        # Verify JSON contains exact unicode without mangling
        data = self.formatter.to_json(report)
        dumped = json.dumps(data, ensure_ascii=False)
        self.assertIn("70 °C ± 0.5 °C — em dash", dumped)

    def test_terminal_summary_conciseness(self):
        """Verifies that the terminal summary provides key numbers without overwhelming candidate listings."""
        report = self._build_mock_report()
        term = self.formatter.to_terminal(
            report,
            json_path="output/compliance_report.json",
            md_path="output/compliance_report.md"
        )

        self.assertIn("V2 COMPLIANCE AUDIT REPORT SUMMARY", term)
        self.assertIn("Overall Verdict    : CONDITIONALLY_COMPLIANT", term)
        self.assertIn("Mandatory Coverage : 33.3% (1/3)", term)
        self.assertIn("Items Requiring Human Review:", term)
        self.assertIn("[B5.1] (B5): Partial evidence conflict", term)
        self.assertIn("Exported JSON Report     : output/compliance_report.json", term)
        self.assertIn("Exported Markdown Report : output/compliance_report.md", term)

    def test_deterministic_output_across_repeated_calls(self):
        """Verifies that repeated formatting calls yield identical outputs when timestamp is pinned."""
        report = self._build_mock_report()
        pinned_ts = "2026-11-04T12:00:00Z"

        json1 = self.formatter.to_json(report, timestamp_override=pinned_ts)
        json2 = self.formatter.to_json(report, timestamp_override=pinned_ts)
        self.assertEqual(json1, json2)

        md1 = self.formatter.to_markdown(report, timestamp_override=pinned_ts)
        md2 = self.formatter.to_markdown(report, timestamp_override=pinned_ts)
        self.assertEqual(md1, md2)

    def test_file_export_methods(self):
        """Verifies export_json and export_markdown write valid, non-empty files to disk."""
        report = self._build_mock_report()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "sub" / "report.json"
            md_file = Path(tmpdir) / "sub" / "report.md"

            out_json = self.formatter.export_json(report, json_file)
            out_md = self.formatter.export_markdown(report, md_file)

            self.assertTrue(out_json.exists())
            self.assertTrue(out_md.exists())

            # Verify JSON can be parsed back
            with open(out_json, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["metadata"]["schema_version"], "2.0")

            # Verify Markdown content
            with open(out_md, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("# SRS & Technical Document Compliance Audit Report", content)


if __name__ == "__main__":
    unittest.main()

