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
    TemplatePoint,
    ReferenceTemplate,
    PointMatchResult,
    TemplateComplianceReport,
    TemplateContentChecker,
)
from src.pdf_extractor import (
    PDFExtractor,
    extract_template_from_pdf,
    extract_text_from_pdf,
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


class TestPhase1TemplateContentChecker(unittest.TestCase):
    """
    Tests for Phase 1 Reference Template Content Checker:
    - Single-page target PDF extraction (--page N) and bounds checking
    - Extraction of required sections and points from template PDFs
    - Point-by-point semantic coverage classification (COVERED, PARTIAL, MISSING)
    - Closest matching target text evidence
    - Configurable similarity thresholds
    - Report generation (Terminal, Markdown, JSON)
    """

    def setUp(self):
        self.sample_pdf = os.path.join(BASE_DIR, "data", "sample_srs.pdf")
        self.avionics_pdf = os.path.join(BASE_DIR, "data", "avionics_sample_srs.pdf")
        self.standard_pdf = os.path.join(BASE_DIR, "data", "reference_standard.pdf")
        self.extractor = PDFExtractor()

    def test_single_page_pdf_extraction(self):
        """Verifies extracting a single page using page_number=1."""
        res_full = self.extractor.extract_text(self.sample_pdf)
        res_page1 = self.extractor.extract_text(self.sample_pdf, page_number=1)
        self.assertEqual(res_page1.page_count, 1)
        self.assertIn("REQ-001", res_page1.text)
        self.assertEqual(res_full.text, res_page1.text)

    def test_page_out_of_bounds_raises_error(self):
        """Verifies ValueError is raised when requesting a non-existent page number."""
        with self.assertRaises(ValueError) as ctx:
            self.extractor.extract_text(self.sample_pdf, page_number=5)
        self.assertIn("out of range", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx0:
            self.extractor.extract_text(self.sample_pdf, page_number=0)
        self.assertIn("out of range", str(ctx0.exception).lower())

    def test_extract_template_from_standard_pdf(self):
        """Verifies extracting points from reference_standard.pdf."""
        template = self.extractor.extract_template(self.standard_pdf)
        self.assertEqual(len(template.points), 8)
        point_ids = [p.point_id for p in template.points]
        self.assertIn("SEC-1", point_ids)
        self.assertIn("SEC-4", point_ids)
        self.assertIn("R1", point_ids)
        self.assertIn("R4", point_ids)
        # Verify text is populated
        for p in template.points:
            self.assertGreater(len(p.text), 10)
            self.assertIsNotNone(p.section_name)

    def test_extract_template_from_sample_srs_pdf(self):
        """Verifies extracting requirement points from sample_srs.pdf."""
        template = extract_template_from_pdf(self.sample_pdf)
        self.assertEqual(len(template.points), 5)
        point_ids = [p.point_id for p in template.points]
        self.assertIn("REQ-001", point_ids)
        self.assertIn("REQ-005", point_ids)

    def test_point_classification_covered_partial_missing(self):
        """
        Verifies classification of points into COVERED, PARTIAL, and MISSING
        based on configurable similarity thresholds.
        """
        template = ReferenceTemplate(
            title="Test Template",
            points=[
                TemplatePoint(
                    point_id="T-01",
                    section_name="Navigation",
                    text="The system shall display the aircraft altitude within 0.5 seconds."
                ),
                TemplatePoint(
                    point_id="T-02",
                    section_name="Navigation",
                    text="The autopilot shall hold heading within 1 degree."
                ),
                TemplatePoint(
                    point_id="T-03",
                    section_name="Cabin",
                    text="The cabin coffee machine shall brew espresso in under 2 minutes."
                )
            ]
        )

        target_chunks = [
            "REQ-001: The system shall display the aircraft altitude within 0.5 seconds.", # exact match for T-01
            "Autopilot heading hold mode maintains bank angle within +/- 2 degrees.",       # partial match for T-02
            "Telemetry is logged to the solid-state recorder at 100 Hz."                    # unrelated to T-03
        ]

        checker = TemplateContentChecker(covered_threshold=0.70, partial_threshold=0.50)
        report = checker.check_compliance(template, target_chunks)

        self.assertEqual(report.total_points, 3)
        self.assertEqual(report.covered_count, 1)
        self.assertEqual(report.partial_count, 1)
        self.assertEqual(report.missing_count, 1)

        res_map = {r.point_id: r for r in report.results}
        self.assertEqual(res_map["T-01"].status, "COVERED")
        self.assertGreaterEqual(res_map["T-01"].similarity_score, 0.70)
        self.assertIn("0.5 seconds", res_map["T-01"].matched_text)

        self.assertEqual(res_map["T-02"].status, "PARTIAL")
        self.assertGreaterEqual(res_map["T-02"].similarity_score, 0.50)
        self.assertLess(res_map["T-02"].similarity_score, 0.70)

        self.assertEqual(res_map["T-03"].status, "MISSING")
        self.assertLess(res_map["T-03"].similarity_score, 0.50)

    def test_configurable_thresholds_validation(self):
        """Verifies that invalid thresholds (partial > covered) raise ValueError."""
        with self.assertRaises(ValueError):
            TemplateContentChecker(covered_threshold=0.40, partial_threshold=0.80)

    def test_report_generation(self):
        """Verifies terminal, markdown, and JSON report generation for TemplateComplianceReport."""
        template = ReferenceTemplate(
            title="Sample Spec",
            points=[
                TemplatePoint("REQ-1", "Telemetry", "The system shall send ping every 10 seconds.")
            ]
        )
        target = ["The system shall send a heartbeat ping every 10 seconds."]
        checker = TemplateContentChecker(covered_threshold=0.70, partial_threshold=0.50)
        report = checker.check_compliance(template, target, target_name="unit_test.pdf", target_page=1, total_pages=1)

        # Terminal
        term_text = ReportGenerator.format_template_terminal(report)
        self.assertIn("REFERENCE-TEMPLATE CONTENT COMPLIANCE REPORT", term_text)
        self.assertIn("[COVERED]", term_text)
        self.assertIn("Page 1 of 1", term_text)

        # Markdown
        md_text = ReportGenerator.format_template_markdown(report)
        self.assertIn("# Reference-Template Content Compliance Report", md_text)
        self.assertIn("REQ-1", md_text)

        # JSON
        json_dict = ReportGenerator.to_template_json(report)
        self.assertEqual(json_dict["summary"]["covered_count"], 1)
        self.assertEqual(json_dict["results"][0]["status"], "COVERED")


class TestTemplateRegressionPacks(unittest.TestCase):
    """
    Regression tests for test document pairs (Pair B and Pair A):
    - Template point extraction for generic numbered/letter-numbered points (B1-B7, A1-A7)
    - Instruction-aware parsing ("Explain", "Describe", "State", "Identify")
    - Target evidence noise filtering (suppressing headings, banners, and synthetic test notes)
    - Meaningful evidence selection and threshold classification (COVERED, PARTIAL, MISSING)
    """

    @classmethod
    def setUpClass(cls):
        cls.pack_dir = os.path.join(BASE_DIR, "data", "test_packs")
        if not os.path.isdir(cls.pack_dir):
            cls.pack_dir = "/tmp/reference_target_test_pack"
        cls.extractor = PDFExtractor()
        cls.checker = TemplateContentChecker()

        cls.b_template_path = os.path.join(cls.pack_dir, "B_reference_technical_template.pdf")
        cls.b_target_path = os.path.join(cls.pack_dir, "B_target_technical_document.pdf")
        cls.a_template_path = os.path.join(cls.pack_dir, "A_reference_lab_report_template.pdf")
        cls.a_target_path = os.path.join(cls.pack_dir, "A_target_lab_report.pdf")

    def test_pair_b_template_extraction(self):
        """Verifies generic B1-B7 point extraction and instruction-aware topic/description parsing."""
        if not os.path.isfile(self.b_template_path):
            self.skipTest(f"Missing test file: {self.b_template_path}")

        template = self.extractor.extract_template(self.b_template_path)
        self.assertEqual(len(template.points), 7, f"Expected 7 points, got {len(template.points)}")

        expected_ids = [f"B{i}" for i in range(1, 8)]
        actual_ids = [p.point_id for p in template.points]
        self.assertEqual(actual_ids, expected_ids)

        expected_topics = {
            "B1": "Purpose",
            "B2": "Scope and boundaries",
            "B3": "Core functions",
            "B4": "Interfaces",
            "B5": "Safety",
            "B6": "Acceptance criteria",
            "B7": "Testing",
        }
        for p in template.points:
            self.assertEqual(p.topic_label, expected_topics[p.point_id])
            self.assertTrue(len(p.description) > 10)
            self.assertTrue(len(p.section_name) > 0)

        # Verify instructional wording was captured in descriptions
        self.assertTrue(template.points[0].description.startswith("Explain"))
        self.assertTrue(template.points[1].description.startswith("State"))
        self.assertTrue(template.points[2].description.startswith("Describe"))

    def test_pair_b_target_noise_filtering(self):
        """Verifies standalone headings, synthetic test notes, and banners are filtered from target chunks."""
        if not os.path.isfile(self.b_target_path):
            self.skipTest(f"Missing test file: {self.b_target_path}")

        chunks = self.extractor.extract_target_chunks(self.b_target_path)
        self.assertGreaterEqual(len(chunks), 8)

        # Disallowed noise patterns
        for chunk in chunks:
            self.assertFalse(
                chunk.startswith("Target document B") or "Synthetic test document" in chunk,
                f"Synthetic banner leaked into evidence: {chunk}"
            )
            self.assertFalse(
                "Designed test intent" in chunk or "Testing note" in chunk,
                f"Synthetic test note leaked into evidence: {chunk}"
            )
            self.assertFalse(
                chunk in ["1. Overview", "2. Main Capabilities", "3. External Connections", "4. Fault Response"],
                f"Standalone section heading leaked into evidence: {chunk}"
            )

        # Expected substantive sentences must be preserved
        substantive_found = any(
            "dashboard helps maintenance staff monitor equipment condition" in c for c in chunks
        )
        self.assertTrue(substantive_found, "Substantive overview sentence was mistakenly discarded.")

        testing_found = any(
            "During testing, an operator will confirm" in c for c in chunks
        )
        self.assertTrue(testing_found, "Substantive testing sentence was mistakenly discarded.")

    def test_pair_b_evidence_selection_and_classification(self):
        """Verifies evidence selection, multi-sentence matching, and explicit absence detection for Pair B."""
        if not (os.path.isfile(self.b_template_path) and os.path.isfile(self.b_target_path)):
            self.skipTest("Pair B test files not available")

        template = self.extractor.extract_template(self.b_template_path)
        chunks = self.extractor.extract_target_chunks(self.b_target_path)
        report = self.checker.check_compliance(template, chunks)

        res_map = {r.point_id: r for r in report.results}

        # B7 (Testing): Must select substantive testing sentence, not test note or banner
        self.assertEqual(res_map["B7"].status, "PARTIAL")
        self.assertGreaterEqual(res_map["B7"].similarity_score, 0.50)
        self.assertIn("During testing, an operator will confirm", res_map["B7"].matched_text)
        self.assertIn(res_map["B7"].winning_perspective, ["canonical", "full"])

        # B6 (Measurable acceptance criteria): Target explicitly states absence of criteria
        self.assertEqual(res_map["B6"].status, "MISSING")
        self.assertTrue(res_map["B6"].is_explicit_absence)
        self.assertEqual(res_map["B6"].evidence_tier, "EXPLICIT_ABSENCE")
        self.assertIn("No numeric response-time target", res_map["B6"].matched_text)

        # B4 (Interfaces): Detects explicit omission statement regarding protocol & interfaces
        self.assertTrue(res_map["B4"].is_explicit_absence)
        self.assertEqual(res_map["B4"].evidence_tier, "EXPLICIT_ABSENCE")

        # B5 (Safety): Scored below partial threshold, classified as MISSING
        self.assertEqual(res_map["B5"].status, "MISSING")
        self.assertFalse(res_map["B5"].is_explicit_absence)

    def test_pair_a_template_extraction(self):
        """Verifies generic A1-A7 lab report template point extraction."""
        if not os.path.isfile(self.a_template_path):
            self.skipTest(f"Missing test file: {self.a_template_path}")

        template = self.extractor.extract_template(self.a_template_path)
        self.assertEqual(len(template.points), 7, f"Expected 7 points, got {len(template.points)}")

        expected_ids = [f"A{i}" for i in range(1, 8)]
        actual_ids = [p.point_id for p in template.points]
        self.assertEqual(actual_ids, expected_ids)

        expected_topics = {
            "A1": "Aim",
            "A2": "Materials",
            "A3": "Procedure",
            "A4": "Observations",
            "A5": "Analysis",
            "A6": "Result",
            "A7": "Safety",
        }
        for p in template.points:
            self.assertEqual(p.topic_label, expected_topics[p.point_id])
            self.assertTrue(len(p.description) > 10)

    def test_pair_a_target_noise_filtering_and_classification(self):
        """Verifies noise filtering, evidence matching, and explicit absence detection for Pair A."""
        if not (os.path.isfile(self.a_template_path) and os.path.isfile(self.a_target_path)):
            self.skipTest("Pair A test files not available")

        template = self.extractor.extract_template(self.a_template_path)
        chunks = self.extractor.extract_target_chunks(self.a_target_path)

        for chunk in chunks:
            self.assertFalse("Synthetic test document" in chunk)
            self.assertFalse(chunk in ["Aim", "Materials", "Procedure", "Observations", "Analysis", "Result", "Safety"])

        report = self.checker.check_compliance(template, chunks)
        res_map = {r.point_id: r for r in report.results}

        # A1 (Aim): Matches substantive aim statement
        self.assertEqual(res_map["A1"].status, "PARTIAL")
        self.assertIn("The aim was to observe how the temperature", res_map["A1"].matched_text)

        # A2 (Materials): Matches substantive materials statement
        self.assertEqual(res_map["A2"].status, "PARTIAL")
        self.assertIn("A beaker, warm water, a thermometer", res_map["A2"].matched_text)

        # A7 (Safety): Correctly identified as MISSING with explicit absence confirmed
        self.assertEqual(res_map["A7"].status, "MISSING")
        self.assertTrue(res_map["A7"].is_explicit_absence)
        self.assertEqual(res_map["A7"].evidence_tier, "EXPLICIT_ABSENCE")
        self.assertIn("No specific hazards or safety precautions are discussed", res_map["A7"].matched_text)

    def test_safeguard_topic_only_coincidental_match_blocked(self):
        """Verifies that a coincidental keyword match on a short topic label cannot establish compliance."""
        template = ReferenceTemplate(
            title="Topic Safeguard Test",
            points=[
                TemplatePoint(
                    point_id="T-AIM",
                    section_name="General",
                    text="Aim: State the scientific hypothesis and research objective.",
                    topic_label="Aim",
                    description="State the scientific hypothesis and research objective."
                )
            ]
        )
        # Sentence uses "aim" in a completely unrelated ballistic context
        target_chunks = ["The soldier carefully took aim with his rifle before pulling the trigger."]
        report = self.checker.check_compliance(template, target_chunks)

        res = report.results[0]
        self.assertEqual(res.status, "MISSING")
        self.assertEqual(res.evidence_tier, "NONE")
        self.assertIsNone(res.matched_text)
        self.assertLess(res.similarity_score, self.checker.min_evidence_threshold)

    def test_explicit_absence_statement_never_counts_as_compliance(self):
        """Verifies that a statement confirming absence of content classifies as MISSING rather than compliance."""
        template = ReferenceTemplate(
            title="Absence Test",
            points=[
                TemplatePoint(
                    point_id="T-SAFE",
                    section_name="Safety",
                    text="Safety: Document all hazard controls and personal protective equipment.",
                    topic_label="Safety",
                    description="Document all hazard controls and personal protective equipment."
                )
            ]
        )
        target_chunks = ["No safety precautions or hazard controls are documented for this project."]
        report = self.checker.check_compliance(template, target_chunks)

        res = report.results[0]
        self.assertEqual(res.status, "MISSING")
        self.assertTrue(res.is_explicit_absence)
        self.assertEqual(res.evidence_tier, "EXPLICIT_ABSENCE")
        self.assertIn("No safety precautions or hazard controls are documented", res.matched_text)

    def test_unrelated_document_produces_no_evidence(self):
        """Verifies that completely out-of-domain text does not produce false evidence snippets."""
        template = ReferenceTemplate(
            title="Unrelated Doc Test",
            points=[
                TemplatePoint("T-01", "Architecture", "Describe the database clustering and replication schema.", topic_label="Architecture", description="Describe the database clustering and replication schema.")
            ]
        )
        unrelated_chunks = [
            "The corporate cafeteria serves fresh sandwiches and soup at noon.",
            "Please remember to water the office plants on Friday afternoon."
        ]
        report = self.checker.check_compliance(template, unrelated_chunks)

        res = report.results[0]
        self.assertEqual(res.status, "MISSING")
        self.assertEqual(res.evidence_tier, "NONE")
        self.assertIsNone(res.matched_text)

    def test_shortest_window_preference_on_comparable_scores(self):
        """Verifies preference for shorter evidence window when scores are within 0.03 tolerance."""
        template = ReferenceTemplate(
            title="Window Preference Test",
            points=[
                TemplatePoint(
                    point_id="T-01",
                    section_name="Telemetry",
                    text="The system shall transmit heartbeats every 10 seconds.",
                    topic_label="Telemetry",
                    description="The system shall transmit heartbeats every 10 seconds."
                )
            ]
        )
        from src.pdf_extractor import CandidateChunk
        c_single = CandidateChunk("The system shall transmit heartbeats every 10 seconds.", section_heading="Telemetry", window_size=1)
        c_multi = CandidateChunk("The system shall transmit heartbeats every 10 seconds. In addition, the auxiliary logger records system uptime every hour.", section_heading="Telemetry", window_size=2)

        report = self.checker.check_compliance(template, [c_multi, c_single])
        res = report.results[0]
        # Should prefer the clean single sentence over the larger 2-sentence window
        self.assertEqual(res.matched_text, c_single.text)


if __name__ == "__main__":
    unittest.main()



