#!/usr/bin/env python3
"""
Unit and Regression Test Suite for SRS Compliance Analyzer (Version 0.1 & Version 0.2)
======================================================================================
Tests:
- R1 - R4 compliance checks (verifying Version 0.1 behavior remains unchanged)
- Backward compatibility on sample_srs.txt
- SemanticSectionMatcher with all-MiniLM-L6-v2 embeddings and cosine similarity
- Mandatory section alignment and omitted section detection on avionics_sample_srs.md
- Report generation (Terminal, Markdown, JSON)
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.analyzer import (
    Requirement,
    ReferenceStandard,
    RequirementExtractor,
    ComplianceAnalyzer,
    ReportGenerator,
    SemanticSectionMatcher,
    DocumentSection,
)


class TestVersion01Rules(unittest.TestCase):
    """Verifies that Version 0.1 rule checks (R1 - R4) remain 100% unchanged."""

    def setUp(self):
        self.standard = ReferenceStandard()
        self.analyzer = ComplianceAnalyzer(self.standard, enable_semantic_matching=False)

    def test_r1_unique_id(self):
        # Valid ID
        req1 = Requirement("REQ-001", "The system shall log in.", "REQ-001: The system shall log in.", 1)
        res1 = self.analyzer.check_r1_id(req1)
        self.assertTrue(res1.is_compliant)

        # Duplicate ID
        req1_dup = Requirement("REQ-001", "The system shall log out.", "REQ-001: The system shall log out.", 2)
        res1_dup = self.analyzer.check_r1_id(req1_dup)
        self.assertFalse(res1_dup.is_compliant)
        self.assertIn("Duplicate", res1_dup.reason)

        # Missing ID
        req_no_id = Requirement(None, "The system shall display altitude.", "The system shall display altitude.", 3)
        res_no_id = self.analyzer.check_r1_id(req_no_id)
        self.assertFalse(res_no_id.is_compliant)
        self.assertIn("missing", res_no_id.reason.lower())

    def test_r2_behavior_shall(self):
        # Valid 'shall'
        req_shall = Requirement("REQ-001", "The system shall compute trajectory.", "raw", 1)
        res_shall = self.analyzer.check_r2_behavior(req_shall)
        self.assertTrue(res_shall.is_compliant)

        # Weak modal 'should'
        req_should = Requirement("REQ-002", "The system should update display.", "raw", 2)
        res_should = self.analyzer.check_r2_behavior(req_should)
        self.assertFalse(res_should.is_compliant)
        self.assertIn("non-binding modal verb 'should'", res_should.reason)

    def test_r3_measurable(self):
        req_metric = Requirement("REQ-001", "The display shall refresh within 50 ms.", "raw", 1)
        res_metric = self.analyzer.check_r3_measurable(req_metric)
        self.assertTrue(res_metric.is_compliant)
        self.assertIn("within 50", res_metric.reason)

    def test_r4_vague_words(self):
        # Clean requirement
        req_clean = Requirement("REQ-001", "The system shall transmit telemetry at 10 Hz.", "raw", 1)
        res_clean = self.analyzer.check_r4_vague(req_clean)
        self.assertTrue(res_clean.is_compliant)

        # Vague word: quickly
        req_vague1 = Requirement("REQ-002", "The system shall boot quickly.", "raw", 2)
        res_vague1 = self.analyzer.check_r4_vague(req_vague1)
        self.assertFalse(res_vague1.is_compliant)
        self.assertIn("quickly", res_vague1.reason)

        # Vague word: user-friendly
        req_vague2 = Requirement("REQ-003", "The UI shall be user-friendly.", "raw", 3)
        res_vague2 = self.analyzer.check_r4_vague(req_vague2)
        self.assertFalse(res_vague2.is_compliant)
        self.assertIn("user-friendly", res_vague2.reason)

    def test_v01_sample_srs_benchmark(self):
        sample_path = os.path.join(BASE_DIR, "data", "sample_srs.txt")
        extractor = RequirementExtractor()
        requirements = extractor.extract_from_file(sample_path)
        self.assertEqual(len(requirements), 5)

        report = self.analyzer.audit_document(requirements)
        self.assertEqual(report.total_requirements, 5)
        self.assertEqual(report.compliant_count, 3)
        self.assertEqual(report.non_compliant_count, 2)
        self.assertEqual(report.compliance_percentage, 60.0)

        # REQ-001 -> Compliant
        self.assertTrue(report.audits[0].is_compliant)
        # REQ-002 -> Non-compliant (quickly)
        self.assertFalse(report.audits[1].is_compliant)
        self.assertEqual(report.audits[1].primary_reason, '"quickly" is vague.')
        # REQ-003 -> Compliant
        self.assertTrue(report.audits[2].is_compliant)
        # REQ-004 -> Non-compliant (user-friendly)
        self.assertFalse(report.audits[3].is_compliant)
        self.assertEqual(report.audits[3].primary_reason, '"user-friendly" is vague.')
        # REQ-005 -> Compliant
        self.assertTrue(report.audits[4].is_compliant)


class TestVersion02SemanticSectionMatching(unittest.TestCase):
    """Verifies Version 0.2 sentence-transformers (all-MiniLM-L6-v2) semantic section matching."""

    @classmethod
    def setUpClass(cls):
        cls.matcher = SemanticSectionMatcher(model_name="all-MiniLM-L6-v2", threshold=0.55)
        standard_path = os.path.join(BASE_DIR, "data", "reference_standard.json")
        cls.standard = ReferenceStandard(standard_path)

    def test_pairwise_semantic_similarity(self):
        # Direct equivalence
        score_exact = self.matcher.compute_similarity("Scope and System Overview", "Scope and System Overview")
        self.assertGreaterEqual(score_exact, 0.99)

        # Semantic paraphrase
        score_paraphrase = self.matcher.compute_similarity(
            "Scope and System Overview",
            "System Scope and Architectural Overview"
        )
        self.assertGreaterEqual(score_paraphrase, 0.75)

        # Irrelevant out-of-domain pair
        score_irrelevant = self.matcher.compute_similarity(
            "Safety and Failure Handling",
            "Cafeteria Lunch Menu and Catering Services"
        )
        self.assertLess(score_irrelevant, 0.30)

    def test_section_matching_on_avionics_srs(self):
        sample_path = os.path.join(BASE_DIR, "data", "avionics_sample_srs.md")
        extractor = RequirementExtractor()
        doc_sections = extractor.extract_sections_from_file(sample_path)
        self.assertGreaterEqual(len(doc_sections), 3)

        matches = self.matcher.match_sections(self.standard.sections, doc_sections, threshold=0.55)
        self.assertEqual(len(matches), 4)

        match_map = {m.reference_section_id: m for m in matches}

        # SEC-1 Scope matched
        self.assertTrue(match_map["SEC-1"].is_matched)
        self.assertGreaterEqual(match_map["SEC-1"].similarity_score, 0.80)

        # SEC-2 Functional matched
        self.assertTrue(match_map["SEC-2"].is_matched)
        self.assertGreaterEqual(match_map["SEC-2"].similarity_score, 0.80)

        # SEC-3 Interface matched
        self.assertTrue(match_map["SEC-3"].is_matched)
        self.assertGreaterEqual(match_map["SEC-3"].similarity_score, 0.80)

        # SEC-4 Safety missing from avionics sample
        self.assertFalse(match_map["SEC-4"].is_matched)
        self.assertLess(match_map["SEC-4"].similarity_score, 0.55)

    def test_full_pipeline_audit_with_sections(self):
        sample_path = os.path.join(BASE_DIR, "data", "avionics_sample_srs.md")
        extractor = RequirementExtractor()
        requirements = extractor.extract_from_file(sample_path)
        sections = extractor.extract_sections_from_file(sample_path)

        analyzer = ComplianceAnalyzer(self.standard, section_matcher=self.matcher, similarity_threshold=0.55)
        report = analyzer.audit_document(requirements, sections=sections)

        self.assertEqual(report.total_requirements, 7)
        self.assertEqual(len(report.section_matches), 4)
        self.assertEqual(report.sections_summary["matched_sections"], 3)
        self.assertEqual(report.sections_summary["missing_sections"], 1)
        self.assertEqual(report.sections_summary["coverage_percentage"], 75.0)

        # Verify JSON serialization
        json_data = ReportGenerator.to_json(report)
        self.assertEqual(json_data["version"], "0.3")
        self.assertEqual(len(json_data["section_matches"]), 4)

        # Verify Markdown serialization
        md_text = ReportGenerator.format_markdown(report)
        self.assertIn("Semantic Section Alignment (all-MiniLM-L6-v2)", md_text)
        self.assertIn("Scope and System Overview", md_text)
        self.assertIn("Safety and Failure Handling", md_text)


class TestVersion03SemanticRuleMatching(unittest.TestCase):
    """
    Verifies Version 0.3 semantic rule matching:
    - sentence-transformers (all-MiniLM-L6-v2) embedding of reference rules (R1-R4)
    - Cosine similarity between individual requirements and rules
    - Controlled demonstration tests: Compliant, Non-Compliant, Paraphrased (R1-R4), Irrelevant
    - Preservation of Version 0.1 R1-R4 rule checks and Version 0.2 section alignment
    """

    @classmethod
    def setUpClass(cls):
        cls.matcher = SemanticSectionMatcher(model_name="all-MiniLM-L6-v2", threshold=0.55)
        standard_path = os.path.join(BASE_DIR, "data", "reference_standard.json")
        cls.standard = ReferenceStandard(standard_path)
        cls.rules = cls.standard.rules

    def test_standard_rules_loaded(self):
        """Verifies that reference rules R1 - R4 are properly loaded into ReferenceStandard."""
        self.assertEqual(len(self.rules), 4)
        rule_ids = [r.id for r in self.rules]
        self.assertListEqual(rule_ids, ["R1", "R2", "R3", "R4"])
        for r in self.rules:
            self.assertTrue(len(r.name) > 0)
            self.assertTrue(len(r.description) > 0)

    def test_paraphrased_rule_matching(self):
        """
        Verifies that paraphrased requirement descriptions semantically match their
        corresponding reference rule (R1-R4) as the top-1 ranked rule.
        """
        # Paraphrased R1 (Unique ID)
        r1_paraphrase = "Every specification item requires a unique alphanumeric identifier tag."
        match_r1 = self.matcher.match_requirement_to_rules(r1_paraphrase, self.rules)
        self.assertEqual(match_r1.matched_rule_id, "R1")
        self.assertGreater(match_r1.similarity_score, 0.50)

        # Paraphrased R2 (Clear Behaviour / Shall)
        r2_paraphrase = "The software is strictly mandated to execute deterministic operational actions."
        match_r2 = self.matcher.match_requirement_to_rules(r2_paraphrase, self.rules)
        self.assertEqual(match_r2.matched_rule_id, "R2")
        self.assertGreater(match_r2.similarity_score, 0.25)

        # Paraphrased R3 (Measurable Criteria / Metrics)
        r3_paraphrase = "All response latencies and timing bounds must be quantitatively measurable."
        match_r3 = self.matcher.match_requirement_to_rules(r3_paraphrase, self.rules)
        self.assertEqual(match_r3.matched_rule_id, "R3")
        self.assertGreater(match_r3.similarity_score, 0.40)

        # Paraphrased R4 (Avoid Vague Language)
        r4_paraphrase = "Ambiguous subjective terms like quickly, fast, or user-friendly must be avoided."
        match_r4 = self.matcher.match_requirement_to_rules(r4_paraphrase, self.rules)
        self.assertEqual(match_r4.matched_rule_id, "R4")
        self.assertGreater(match_r4.similarity_score, 0.40)

    def test_non_compliant_vague_matching(self):
        """
        Verifies semantic matching on a non-compliant requirement with vague language.
        Should associate highest with R4 (Avoid Vague Language).
        """
        vague_statement = "The user interface should be easy, intuitive, and display information quickly."
        match = self.matcher.match_requirement_to_rules(vague_statement, self.rules)
        self.assertEqual(match.matched_rule_id, "R4")
        self.assertGreater(match.similarity_score, 0.10)
        self.assertIn("R4", match.rule_scores)

    def test_irrelevant_out_of_domain_matching(self):
        """
        Verifies that completely irrelevant out-of-domain text yields low similarity
        scores across all reference rules (< 0.15).
        """
        irrelevant_text = "The corporate cafeteria serves fresh sandwiches and hot soup between noon and 1:30 PM."
        match = self.matcher.match_requirement_to_rules(irrelevant_text, self.rules)
        # All individual rule scores must be very low
        for rule_id, score in match.rule_scores.items():
            self.assertLess(score, 0.15, f"Rule {rule_id} score unexpectedly high: {score}")

    def test_compliant_sample_requirement_matching(self):
        """
        Verifies semantic matching on a well-formed compliant SRS requirement.
        """
        compliant_text = "REQ-001: The system shall boot within 5 seconds of power-on."
        match = self.matcher.match_requirement_to_rules(compliant_text, self.rules)
        self.assertEqual(match.matched_rule_id, "R1")
        self.assertGreater(match.similarity_score, 0.30)

    def test_batch_rule_matching(self):
        """Verifies batch semantic matching of multiple requirements."""
        reqs = [
            "Every specification item requires a unique alphanumeric identifier tag.",
            "All response latencies and timing bounds must be quantitatively measurable.",
            "Ambiguous subjective terms like quickly, fast, or user-friendly must be avoided."
        ]
        results = self.matcher.match_requirements_to_rules(reqs, self.rules)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].matched_rule_id, "R1")
        self.assertEqual(results[1].matched_rule_id, "R3")
        self.assertEqual(results[2].matched_rule_id, "R4")

    def test_end_to_end_audit_preservation_and_enrichment(self):
        """
        Verifies that running a full document audit:
        1. Keeps Version 0.1 R1-R4 pass/fail logic intact (3 compliant, 2 non-compliant for sample_srs.txt).
        2. Populates matched_rule_id, matched_rule_name, and rule_similarity_score for all requirements.
        3. Exports valid Version 0.3 JSON and Markdown.
        """
        sample_path = os.path.join(BASE_DIR, "data", "sample_srs.txt")
        extractor = RequirementExtractor()
        requirements = extractor.extract_from_file(sample_path)

        analyzer = ComplianceAnalyzer(self.standard, section_matcher=self.matcher, similarity_threshold=0.55)
        report = analyzer.audit_document(requirements)

        # Version 0.1 metrics must be exactly preserved
        self.assertEqual(report.total_requirements, 5)
        self.assertEqual(report.compliant_count, 3)
        self.assertEqual(report.non_compliant_count, 2)
        self.assertEqual(report.compliance_percentage, 60.0)

        # Check Version 0.3 enrichments on each audit
        for audit in report.audits:
            self.assertIsNotNone(audit.matched_rule_id)
            self.assertIsNotNone(audit.matched_rule_name)
            self.assertIsNotNone(audit.rule_similarity_score)
            self.assertIsInstance(audit.rule_similarity_score, float)
            self.assertEqual(len(audit.rule_scores), 4)

        # Check ReportGenerator JSON
        json_output = ReportGenerator.to_json(report)
        self.assertEqual(json_output["version"], "0.3")
        for audit_dict in json_output["audits"]:
            self.assertIn("matched_rule_id", audit_dict)
            self.assertIn("matched_rule_name", audit_dict)
            self.assertIn("rule_similarity_score", audit_dict)
            self.assertIn("rule_scores", audit_dict)

        # Check ReportGenerator Markdown
        md_output = ReportGenerator.format_markdown(report)
        self.assertIn("(v0.3)", md_output)
        self.assertIn("Semantic Rule Alignment", md_output)


from src.pdf_extractor import (
    PDFExtractor,
    ScannedPDFError,
    PDFExtractionResult,
)


class TestVersion04PDFSupport(unittest.TestCase):
    """
    Verifies Version 0.4 PDF Input Support:
    - Digital PDF text extraction for SRS specifications and Reference Standards
    - Requirement and section parsing from PDF files
    - Reference standard section and rule extraction from PDF files
    - Graceful scanned / image-only PDF detection and OCR warning emission
    - End-to-end PDF-to-PDF compliance analysis preserving R1-R4 and semantic matching
    """

    @classmethod
    def setUpClass(cls):
        cls.pdf_extractor = PDFExtractor()
        cls.sample_pdf = os.path.join(BASE_DIR, "data", "sample_srs.pdf")
        cls.avionics_pdf = os.path.join(BASE_DIR, "data", "avionics_sample_srs.pdf")
        cls.standard_pdf = os.path.join(BASE_DIR, "data", "reference_standard.pdf")
        cls.scanned_pdf = os.path.join(BASE_DIR, "data", "scanned_sample_srs.pdf")

    def test_extract_text_from_digital_pdf(self):
        """Verifies text extraction from a valid digital PDF."""
        result = self.pdf_extractor.extract_text(self.sample_pdf)
        self.assertIsInstance(result, PDFExtractionResult)
        self.assertEqual(result.page_count, 1)
        self.assertGreater(result.char_count, 100)
        self.assertFalse(result.is_scanned)
        self.assertIn("REQ-001", result.text)
        self.assertIn("aircraft altitude", result.text)

    def test_extract_srs_requirements_from_pdf(self):
        """
        Verifies that extracting requirements from sample_srs.pdf produces
        identical items to sample_srs.txt and evaluates to 60.0% compliance (3 compliant, 2 non-compliant).
        """
        sections, reqs = self.pdf_extractor.extract_srs_document(self.sample_pdf)
        self.assertEqual(len(reqs), 5)
        req_ids = [r.req_id for r in reqs]
        self.assertListEqual(req_ids, ["REQ-001", "REQ-002", "REQ-003", "REQ-004", "REQ-005"])

        # Audit against reference standard
        standard = ReferenceStandard(os.path.join(BASE_DIR, "data", "reference_standard.json"))
        analyzer = ComplianceAnalyzer(standard, enable_semantic_matching=False)
        report = analyzer.audit_document(reqs)

        self.assertEqual(report.total_requirements, 5)
        self.assertEqual(report.compliant_count, 3)
        self.assertEqual(report.non_compliant_count, 2)
        self.assertEqual(report.compliance_percentage, 60.0)

    def test_extract_avionics_srs_sections_and_requirements(self):
        """Verifies extraction of structural sections and 7 requirements from avionics_sample_srs.pdf."""
        sections, reqs = self.pdf_extractor.extract_srs_document(self.avionics_pdf)
        self.assertGreaterEqual(len(sections), 3)
        section_titles = [s.clean_title for s in sections]
        self.assertIn("Scope and System Overview", section_titles)
        self.assertIn("Functional Requirements", section_titles)
        self.assertIn("Interface Requirements", section_titles)

        self.assertEqual(len(reqs), 7)
        # Verify specific requirements
        req_ids = [r.req_id for r in reqs if r.req_id]
        self.assertIn("REQ-FDM-001", req_ids)
        self.assertIn("REQ-FDM-002", req_ids)
        self.assertIn("REQ-FDM-003", req_ids)
        self.assertIn("REQ-FDM-004", req_ids)
        self.assertIn("REQ-INT-001", req_ids)
        self.assertIn("REQ-INT-002", req_ids)
        # Check requirement without ID
        unidentified = [r for r in reqs if not r.req_id]
        self.assertEqual(len(unidentified), 1)

    def test_extract_reference_standard_from_pdf(self):
        """Verifies extracting sections, rules, and parameters from reference_standard.pdf."""
        standard = self.pdf_extractor.extract_reference_standard(self.standard_pdf)
        self.assertEqual(standard.version, "0.4")
        self.assertEqual(standard.binding_modal, "shall")

        # Sections verification
        self.assertEqual(len(standard.sections), 4)
        sec_ids = [s.section_id for s in standard.sections]
        self.assertIn("SEC-1", sec_ids)
        self.assertIn("SEC-2", sec_ids)
        self.assertIn("SEC-3", sec_ids)
        self.assertIn("SEC-4", sec_ids)

        # Rules verification
        self.assertEqual(len(standard.rules), 4)
        rule_ids = [r.rule_id for r in standard.rules]
        self.assertIn("R1", rule_ids)
        self.assertIn("R2", rule_ids)
        self.assertIn("R3", rule_ids)
        self.assertIn("R4", rule_ids)

        # Vague words
        for word in ["quickly", "fast", "easy", "adequate", "user-friendly"]:
            self.assertIn(word, standard.vague_words)

    def test_scanned_pdf_graceful_detection(self):
        """
        Verifies that a scanned/image-only PDF with negligible extractable text
        is detected as scanned and raises ScannedPDFError indicating OCR is required.
        """
        # Detection without exception
        is_scanned = self.pdf_extractor.is_scanned_pdf(self.scanned_pdf)
        self.assertTrue(is_scanned)

        # Raising ScannedPDFError
        with self.assertRaises(ScannedPDFError) as ctx:
            self.pdf_extractor.extract_text(self.scanned_pdf, raise_if_scanned=True)

        err_msg = str(ctx.exception)
        self.assertIn("Optical Character Recognition (OCR) is required", err_msg)
        self.assertEqual(ctx.exception.char_count, 0)
        self.assertEqual(ctx.exception.page_count, 1)

    def test_end_to_end_pdf_to_pdf_audit(self):
        """
        Verifies end-to-end audit where both the reference standard and the SRS
        are supplied as PDF files, preserving section alignment and semantic rule matching.
        """
        # Load standard directly from PDF
        standard = ReferenceStandard(self.standard_pdf)
        self.assertEqual(len(standard.rules), 4)

        # Extract SRS from PDF via RequirementExtractor API
        extractor = RequirementExtractor()
        requirements = extractor.extract_from_file(self.avionics_pdf)
        sections = extractor.extract_sections_from_file(self.avionics_pdf)

        self.assertEqual(len(requirements), 7)
        self.assertGreaterEqual(len(sections), 3)

        analyzer = ComplianceAnalyzer(standard, similarity_threshold=0.55)
        report = analyzer.audit_document(requirements, sections=sections)

        self.assertEqual(report.total_requirements, 7)
        self.assertEqual(len(report.section_matches), 4)
        # Verify that audits contain semantic rule matching enrichments
        for audit in report.audits:
            self.assertIsNotNone(audit.matched_rule_id)
            self.assertIsNotNone(audit.matched_rule_name)
            self.assertIsNotNone(audit.rule_similarity_score)


if __name__ == "__main__":
    unittest.main()


