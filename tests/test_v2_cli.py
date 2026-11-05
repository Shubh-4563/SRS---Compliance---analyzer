"""
Unit and Integration Tests for V2 CLI Integration (Phase 2.5.4).
Tests:
1. Default CLI execution routes to V2 CompliancePipeline.
2. Explicit --legacy-rules flag routes to V1 legacy pipeline.
3. Argument parsing and forwarding (--template, --target, --page, --top-k, --output-json, --output-md).
4. Page argument correctly forwarded to V2 pipeline.
5. Export of JSON, Markdown, and both reports together with automatic directory creation.
6. Error handling and non-zero exit codes (missing files, invalid page, scanned PDFs, unhandled errors).
7. Non-fallback invariant: V2 failures must never silently fall back to V1.
8. V1 regression tests through CLI argument invocation.
Mocks heavy ML models to ensure zero model downloads or latency during test execution.
"""
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import main as cli_main
from src.pdf_extractor import ScannedPDFError
from src.models.compliance import (
    ParentComplianceStatus,
    DocumentComplianceVerdict,
    ParentPointCompliance,
    DocumentComplianceReport,
)


class TestV2CLI(unittest.TestCase):
    """Test suite for main.py V2 CLI integration."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

        # Create valid mock input files
        self.ref_file = self.dir_path / "reference.pdf"
        self.ref_file.write_bytes(b"%PDF-1.4 Mock Ref")

        self.tgt_file = self.dir_path / "target.pdf"
        self.tgt_file.write_bytes(b"%PDF-1.4 Mock Tgt")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_mock_report(
        self,
        target_page=None,
        verdict=DocumentComplianceVerdict.COMPLIANT,
    ) -> DocumentComplianceReport:
        p1 = ParentPointCompliance(
            point_id="A1",
            section_name="Aim",
            topic_label="Purpose",
            description="State experiment purpose.",
            raw_text="A1. State purpose.",
            status=ParentComplianceStatus.FULLY_COMPLIANT,
            mandatory_fulfilled=1,
            mandatory_total=1,
            optional_fulfilled=0,
            optional_total=0,
            sub_results=[],
            findings=["Met."]
        )
        return DocumentComplianceReport(
            target_document_path=str(self.tgt_file),
            target_document_name="target.pdf",
            template_title="Mock Reference Template",
            template_path=str(self.ref_file),
            total_parent_points=1,
            total_sub_requirements=1,
            parent_points=[p1],
            mandatory_coverage_pct=100.0,
            total_mandatory=1,
            mandatory_fulfilled=1,
            fulfilled_count=1,
            verdict=verdict,
            summary="Compliance audit completed successfully.",
            total_pages=3,
            target_page=target_page,
        )

    # --------------------------------------------------------------------------
    # 1. Default Execution & Routing Tests
    # --------------------------------------------------------------------------

    @patch("main.run_v2_pipeline")
    @patch("main.run_legacy_pipeline")
    def test_default_execution_selects_v2(self, mock_v1, mock_v2):
        """Verifies that running without --legacy-rules selects the V2 pipeline by default."""
        mock_v2.return_value = 0
        args = ["--template", str(self.ref_file), "--target", str(self.tgt_file)]

        exit_code = cli_main.main(args)

        self.assertEqual(exit_code, 0)
        mock_v2.assert_called_once()
        mock_v1.assert_not_called()

    @patch("main.run_v2_pipeline")
    @patch("main.run_legacy_pipeline")
    def test_legacy_rules_flag_selects_v1(self, mock_v1, mock_v2):
        """Verifies that passing --legacy-rules explicitly routes to the V1 pipeline."""
        mock_v1.return_value = None
        args = ["--legacy-rules", "--template", str(self.ref_file), "--target", str(self.tgt_file)]

        exit_code = cli_main.main(args)

        self.assertEqual(exit_code, 0)
        mock_v1.assert_called_once()
        mock_v2.assert_not_called()

    # --------------------------------------------------------------------------
    # 2. Argument Parsing and Forwarding Tests
    # --------------------------------------------------------------------------

    @patch("main.CompliancePipeline")
    def test_page_and_top_k_forwarded_correctly(self, mock_pipeline_cls):
        """Verifies that --page and --top-k arguments are passed to the V2 pipeline."""
        mock_pipeline_inst = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline_inst
        mock_pipeline_inst.run.return_value = self._make_mock_report(target_page=2)

        args = [
            "--template", str(self.ref_file),
            "--target", str(self.tgt_file),
            "--page", "2",
            "--top-k", "7",
            "--output-json", str(self.dir_path / "out.json"),
            "--output-md", str(self.dir_path / "out.md"),
        ]

        exit_code = cli_main.main(args)

        self.assertEqual(exit_code, 0)
        mock_pipeline_inst.run.assert_called_once_with(
            reference_path=str(self.ref_file),
            target_path=str(self.tgt_file),
            target_page=2,
            top_k=7,
        )

    # --------------------------------------------------------------------------
    # 3. Report Export & Directory Creation Tests
    # --------------------------------------------------------------------------

    @patch("main.CompliancePipeline")
    def test_json_and_markdown_export_with_directory_creation(self, mock_pipeline_cls):
        """Verifies that JSON and Markdown reports are saved and nested directories created."""
        mock_pipeline_inst = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline_inst
        mock_pipeline_inst.run.return_value = self._make_mock_report()

        nested_out_dir = self.dir_path / "nested" / "reports"
        json_file = nested_out_dir / "audit.json"
        md_file = nested_out_dir / "audit.md"

        args = [
            "-r", str(self.ref_file),
            "-s", str(self.tgt_file),
            "-j", str(json_file),
            "-m", str(md_file),
        ]

        exit_code = cli_main.main(args)

        self.assertEqual(exit_code, 0)
        self.assertTrue(json_file.exists())
        self.assertTrue(md_file.exists())

        # Check JSON content
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["metadata"]["schema_version"], "2.0")
        self.assertEqual(data["executive_summary"]["verdict"], "COMPLIANT")

        # Check Markdown content (no tables)
        with open(md_file, "r", encoding="utf-8") as f:
            md_content = f.read()
        self.assertIn("# SRS & Technical Document Compliance Audit Report", md_content)
        self.assertNotIn("|---", md_content)

    # --------------------------------------------------------------------------
    # 4. Error Handling & Non-Zero Exit Codes
    # --------------------------------------------------------------------------

    def test_missing_input_file_returns_code_1(self):
        """Verifies non-zero exit code 1 when input files are missing."""
        stderr_buf = io.StringIO()
        with patch("sys.stderr", stderr_buf):
            exit_code = cli_main.main([
                "--template", str(self.dir_path / "non_existent_ref.pdf"),
                "--target", str(self.tgt_file),
            ])
        self.assertEqual(exit_code, 1)
        self.assertIn("File not found", stderr_buf.getvalue())

    def test_scanned_pdf_error_returns_code_2(self):
        """Verifies exit code 2 when a scanned/image-only PDF is detected."""
        with patch("main.CompliancePipeline") as mock_pipeline_cls:
            mock_inst = MagicMock()
            mock_pipeline_cls.return_value = mock_inst
            mock_inst.run.side_effect = ScannedPDFError(
                filepath=str(self.tgt_file),
                char_count=5,
                page_count=1
            )
            stderr_buf = io.StringIO()
            with patch("sys.stderr", stderr_buf):
                exit_code = cli_main.main([
                    "--template", str(self.ref_file),
                    "--target", str(self.tgt_file),
                ])

            self.assertEqual(exit_code, 2)
            self.assertIn("SCANNED / IMAGE-ONLY PDF DETECTED", stderr_buf.getvalue())

    def test_invalid_page_number_returns_code_1(self):
        """Verifies exit code 1 when an invalid target_page is requested."""
        with patch("main.CompliancePipeline") as mock_pipeline_cls:
            mock_inst = MagicMock()
            mock_pipeline_cls.return_value = mock_inst
            mock_inst.run.side_effect = ValueError("Invalid target_page=99. Target has only 3 page(s).")

            stderr_buf = io.StringIO()
            with patch("sys.stderr", stderr_buf):
                exit_code = cli_main.main([
                    "--template", str(self.ref_file),
                    "--target", str(self.tgt_file),
                    "--page", "99",
                ])

            self.assertEqual(exit_code, 1)
            self.assertIn("Invalid input or document", stderr_buf.getvalue())

    def test_v2_failure_never_silently_falls_back_to_v1(self):
        """Verifies that if V2 pipeline fails, it reports error and exits without calling V1."""
        with patch("main.CompliancePipeline") as mock_pipeline_cls:
            mock_inst = MagicMock()
            mock_pipeline_cls.return_value = mock_inst
            mock_inst.run.side_effect = RuntimeError("Inference memory fault")

            with patch("main.run_legacy_pipeline") as mock_v1:
                stderr_buf = io.StringIO()
                with patch("sys.stderr", stderr_buf):
                    exit_code = cli_main.main([
                        "--template", str(self.ref_file),
                        "--target", str(self.tgt_file),
                    ])

                self.assertEqual(exit_code, 1)
                self.assertIn("Compliance audit pipeline failed", stderr_buf.getvalue())
                # Must NEVER call legacy pipeline on V2 error
                mock_v1.assert_not_called()

    # --------------------------------------------------------------------------
    # 5. Terminal Output Quality
    # --------------------------------------------------------------------------

    @patch("main.CompliancePipeline")
    def test_terminal_summary_output_format(self, mock_pipeline_cls):
        """Verifies that a concise summary is printed to stdout without dumping verbose candidate evaluations."""
        mock_inst = MagicMock()
        mock_pipeline_cls.return_value = mock_inst
        mock_inst.run.return_value = self._make_mock_report()

        stdout_buf = io.StringIO()
        with patch("sys.stdout", stdout_buf):
            exit_code = cli_main.main([
                "--template", str(self.ref_file),
                "--target", str(self.tgt_file),
            ])

        self.assertEqual(exit_code, 0)
        output = stdout_buf.getvalue()
        self.assertIn("V2 COMPLIANCE AUDIT REPORT SUMMARY", output)
        self.assertIn("Overall Verdict    : COMPLIANT", output)
        self.assertIn("Mandatory Coverage : 100.0% (1/1)", output)
        # Should NOT dump candidate evaluations or NLI probability tables
        self.assertNotIn("Candidate Block `", output)


if __name__ == "__main__":
    unittest.main()

