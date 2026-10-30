"""
Unit and regression tests for V2 Document Parser and Requirement Decomposer (Phase 2.1).
Covers structural extraction, block typing, section hierarchies, table handling,
traceability, and atomic decomposition without inventing criteria.
Cross-platform compatible (macOS and Windows).
"""
import os
import unittest
from pathlib import Path

from src.models.document import BlockType, DocumentBlock, TargetDocument
from src.models.requirement import (
    EvidenceType,
    SubRequirement,
    DecomposedTemplatePoint,
    DecomposedReferenceTemplate,
)
from src.parsers.pdf_parser import TargetPDFParser
from src.parsers.requirement_decomposer import RequirementDecomposer
from src.pdf_extractor import PDFExtractor, TemplatePoint, ReferenceTemplate


class TestV2TargetPDFParser(unittest.TestCase):
    """Verifies TargetPDFParser on Pair A and Pair B synthetic test PDFs."""

    def setUp(self):
        self.parser = TargetPDFParser()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.a_target_path = os.path.join(self.base_dir, "data", "test_packs", "A_target_lab_report.pdf")
        self.b_target_path = os.path.join(self.base_dir, "data", "test_packs", "B_target_technical_document.pdf")

    def test_parse_pair_a_structural_blocks(self):
        """Verifies headings, paragraphs, tables, and disclaimers are correctly typed for Target A."""
        if not os.path.isfile(self.a_target_path):
            self.skipTest(f"Missing test file: {self.a_target_path}")

        doc = self.parser.parse_document(self.a_target_path)
        self.assertIsInstance(doc, TargetDocument)
        self.assertEqual(doc.total_pages, 1)

        # Check headings
        headings = [b.text for b in doc.headings]
        expected_headings = [
            "1. Aim",
            "2. Materials",
            "3. Method",
            "4. Observations",
            "5. Discussion",
            "6. Conclusion"
        ]
        self.assertEqual(headings, expected_headings)

        # Check table row detection (Section 4 observations)
        obs_blocks = [b for b in doc.blocks if b.section_heading == "4. Observations"]
        table_blocks = [b for b in obs_blocks if b.block_type == BlockType.TABLE_ROW]
        self.assertGreaterEqual(len(table_blocks), 1)
        self.assertIn("start 70 degrees C", table_blocks[0].text)

        # Check explicit disclaimer isolation (Section 6 conclusion vs safety absence)
        conc_blocks = [b for b in doc.blocks if b.section_heading == "6. Conclusion"]
        self.assertEqual(len(conc_blocks), 3)  # Heading, positive conclusion, disclaimer
        disclaimer_blocks = [b for b in conc_blocks if b.block_type == BlockType.DISCLAIMER]
        self.assertEqual(len(disclaimer_blocks), 1)
        self.assertTrue(disclaimer_blocks[0].is_explicit_omission)
        self.assertIn("No specific hazards or safety precautions", disclaimer_blocks[0].text)

        # Verify positive conclusion paragraph is distinct from disclaimer
        pos_conc = [b for b in conc_blocks if b.block_type == BlockType.PARAGRAPH]
        self.assertEqual(len(pos_conc), 1)
        self.assertIn("demonstrated a decrease in water temperature", pos_conc[0].text)
        self.assertFalse(pos_conc[0].is_explicit_omission)

    def test_parse_pair_b_structural_blocks(self):
        """Verifies section hierarchy, positive vs omission isolation for Target B."""
        if not os.path.isfile(self.b_target_path):
            self.skipTest(f"Missing test file: {self.b_target_path}")

        doc = self.parser.parse_document(self.b_target_path)
        self.assertIsInstance(doc, TargetDocument)
        self.assertEqual(doc.total_pages, 1)

        headings = [b.text for b in doc.headings]
        expected_headings = [
            "1. Overview",
            "2. Main Capabilities",
            "3. Connections",
            "4. Fault Response",
            "5. Deployment",
            "6. Validation"
        ]
        self.assertEqual(headings, expected_headings)

        # Check Connections section: positive interface statement separated from omission statement
        conn_blocks = [b for b in doc.blocks if b.section_heading == "3. Connections"]
        # Expect Heading, positive paragraph (operators/sensors), and disclaimer (protocol)
        conn_pos = [b for b in conn_blocks if b.block_type == BlockType.PARAGRAPH]
        conn_dis = [b for b in conn_blocks if b.block_type == BlockType.DISCLAIMER]
        self.assertEqual(len(conn_pos), 1)
        self.assertEqual(len(conn_dis), 1)
        self.assertIn("Operators use a web browser", conn_pos[0].text)
        self.assertIn("The document does not specify the communication protocol", conn_dis[0].text)

        # Check Fault Response section: positive fault warning separated from recovery omission
        fault_blocks = [b for b in doc.blocks if b.section_heading == "4. Fault Response"]
        fault_pos = [b for b in fault_blocks if b.block_type == BlockType.PARAGRAPH]
        fault_dis = [b for b in fault_blocks if b.block_type == BlockType.DISCLAIMER]
        self.assertEqual(len(fault_pos), 1)
        self.assertEqual(len(fault_dis), 1)
        self.assertIn("dashboard displays a warning", fault_pos[0].text)
        self.assertIn("does not explain recovery steps", fault_dis[0].text)

        # Check Validation section: pure positive paragraph without synthetic footer notes
        val_blocks = [b for b in doc.blocks if b.section_heading == "6. Validation"]
        val_pos = [b for b in val_blocks if b.block_type == BlockType.PARAGRAPH]
        self.assertEqual(len(val_pos), 1)
        self.assertIn("During testing, an operator will confirm", val_pos[0].text)
        self.assertNotIn("Designed test intent", val_pos[0].text)
        self.assertNotIn("acceptance criteria are absent", val_pos[0].text)

    def test_single_page_and_out_of_bounds(self):
        """Verifies target_page filtering and bounds checking."""
        if not os.path.isfile(self.a_target_path):
            self.skipTest("Missing test file")

        # Page 1 valid
        doc = self.parser.parse_document(self.a_target_path, target_page=1)
        self.assertEqual(doc.target_page, 1)

        # Page 2 invalid (doc only has 1 page)
        with self.assertRaises(ValueError):
            self.parser.parse_document(self.a_target_path, target_page=2)

        # Page 0 invalid
        with self.assertRaises(ValueError):
            self.parser.parse_document(self.a_target_path, target_page=0)


class TestV2RequirementDecomposer(unittest.TestCase):
    """Verifies atomic requirement decomposition and traceability."""

    def setUp(self):
        self.extractor = PDFExtractor()
        self.decomposer = RequirementDecomposer()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.a_template_path = os.path.join(self.base_dir, "data", "test_packs", "A_reference_lab_report_template.pdf")
        self.b_template_path = os.path.join(self.base_dir, "data", "test_packs", "B_reference_technical_template.pdf")

    def test_decompose_template_a(self):
        """Verifies decomposition of Lab Report Template (A1-A7)."""
        if not os.path.isfile(self.a_template_path):
            self.skipTest("Missing Template A")

        tpl = self.extractor.extract_template(self.a_template_path)
        dec = self.decomposer.decompose_template(tpl)

        self.assertEqual(len(dec.points), 7)
        point_map = {p.point_id: p for p in dec.points}

        # A1: Aim -> Purpose (A1.1) and Measure (A1.2)
        self.assertEqual(len(point_map["A1"].sub_requirements), 2)
        self.assertEqual(point_map["A1"].sub_requirements[0].sub_id, "A1.1")
        self.assertEqual(point_map["A1"].sub_requirements[1].sub_id, "A1.2")
        self.assertEqual(point_map["A1"].sub_requirements[0].parent_point_id, "A1")
        self.assertTrue(point_map["A1"].sub_requirements[0].is_mandatory)

        # A2: Materials -> Equipment (A2.1) and Materials/Consumables (A2.2)
        self.assertEqual(len(point_map["A2"].sub_requirements), 2)
        self.assertEqual(point_map["A2"].sub_requirements[0].expected_evidence_type, EvidenceType.LIST_OF_ITEMS)

        # A4: Observations -> Numeric measurements (A4.1)
        self.assertEqual(len(point_map["A4"].sub_requirements), 1)
        self.assertEqual(point_map["A4"].sub_requirements[0].expected_evidence_type, EvidenceType.NUMERIC_TARGET)

        # Verify all sub-requirements retain original text
        for p in dec.points:
            for s in p.sub_requirements:
                self.assertTrue(len(s.original_text) > 0)
                self.assertEqual(s.parent_point_id, p.point_id)

    def test_decompose_template_b(self):
        """Verifies decomposition of Technical Document Template (B1-B7)."""
        if not os.path.isfile(self.b_template_path):
            self.skipTest("Missing Template B")

        tpl = self.extractor.extract_template(self.b_template_path)
        dec = self.decomposer.decompose_template(tpl)

        self.assertEqual(len(dec.points), 7)
        point_map = {p.point_id: p for p in dec.points}

        # B1: Purpose (B1.1) and Problem solved (B1.2)
        self.assertEqual(len(point_map["B1"].sub_requirements), 2)

        # B2: Scope inclusion (B2.1) and Negative boundary exclusion (B2.2)
        self.assertEqual(len(point_map["B2"].sub_requirements), 2)
        self.assertEqual(point_map["B2"].sub_requirements[1].expected_evidence_type, EvidenceType.NEGATIVE_BOUNDARY)

        # B4: Interfaces -> Users (B4.1, mandatory), Gateways (B4.2, mandatory), Protocol (B4.3, optional)
        b4_sub = point_map["B4"].sub_requirements
        self.assertEqual(len(b4_sub), 3)
        self.assertTrue(b4_sub[0].is_mandatory)
        self.assertTrue(b4_sub[1].is_mandatory)
        self.assertFalse(b4_sub[2].is_mandatory)  # Protocol is non-mandatory detail

        # B5: Safety -> Fault detection (B5.1, mandatory) and Recovery steps (B5.2, optional)
        b5_sub = point_map["B5"].sub_requirements
        self.assertEqual(len(b5_sub), 2)
        self.assertTrue(b5_sub[0].is_mandatory)
        self.assertFalse(b5_sub[1].is_mandatory)

        # B6: Acceptance Criteria -> Numeric target (B6.1, mandatory)
        self.assertEqual(len(point_map["B6"].sub_requirements), 1)
        self.assertEqual(point_map["B6"].sub_requirements[0].expected_evidence_type, EvidenceType.NUMERIC_TARGET)

    def test_generic_unknown_requirement_decomposition(self):
        """Verifies fallback heuristic decomposition on an unknown requirement."""
        p_unknown = TemplatePoint(
            point_id="X1",
            section_name="Security",
            text="State all encryption algorithms; describe key management protocols.",
            topic_label="Encryption",
            description="State all encryption algorithms; describe key management protocols.",
            line_number=1,
            raw_line="X1. State all encryption algorithms; describe key management protocols."
        )

        dec_p = self.decomposer.decompose_point(p_unknown)
        self.assertEqual(dec_p.point_id, "X1")
        # Semicolon should safely split into two sub-requirements
        self.assertEqual(len(dec_p.sub_requirements), 2)
        self.assertEqual(dec_p.sub_requirements[0].sub_id, "X1.1")
        self.assertEqual(dec_p.sub_requirements[1].sub_id, "X1.2")
        self.assertTrue(dec_p.sub_requirements[0].needs_review)

        # Non-splittable single requirement
        p_single = TemplatePoint(
            point_id="X2",
            section_name="Performance",
            text="Maintain 99.9% uptime throughout annual operations.",
            topic_label="Uptime",
            description="Maintain 99.9% uptime throughout annual operations.",
            line_number=2,
            raw_line="X2. Maintain 99.9% uptime throughout annual operations."
        )
        dec_single = self.decomposer.decompose_point(p_single)
        self.assertEqual(len(dec_single.sub_requirements), 1)
        self.assertEqual(dec_single.sub_requirements[0].sub_id, "X2.1")
        self.assertFalse(dec_single.sub_requirements[0].needs_review)


if __name__ == "__main__":
    unittest.main()

