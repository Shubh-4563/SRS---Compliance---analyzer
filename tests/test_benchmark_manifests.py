"""
Integration tests for production declarative benchmark manifests (Phase 2.6.2.3).
Verifies that:
- Pair A manifest (pair_a_lab_report.json) conforms to Layers A, B, and C.
- Pair B manifest (pair_b_srs.json) conforms to Layers A, B, and C.
- Ground truth annotations match verified pipeline evaluations.
- Physical document integrity and excerpt text grounding hold true.
"""
import copy
import json
import unittest
from pathlib import Path

from src.benchmarks.manifest_validator import (
    load_and_validate_manifest,
    validate_schema,
    validate_cross_field_rules,
    validate_document_integrity,
    BenchmarkIntegrityError,
)
from src.models.benchmark import (
    BenchmarkManifest,
    BenchmarkDocumentComplianceVerdict,
    BenchmarkVerificationStatus,
    BenchmarkParentComplianceStatus,
)


class TestBenchmarkManifests(unittest.TestCase):
    """Test suite validating production benchmark manifests against live test pack PDFs."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.benchmarks_dir = cls.project_root / "data" / "benchmarks"
        cls.pair_a_path = cls.benchmarks_dir / "pair_a_lab_report.json"
        cls.pair_b_path = cls.benchmarks_dir / "pair_b_srs.json"

    def test_pair_a_manifest_exists(self):
        """Verifies that the Pair A benchmark manifest file exists."""
        self.assertTrue(self.pair_a_path.is_file(), f"Missing Pair A manifest at: {self.pair_a_path}")

    def test_pair_b_manifest_exists(self):
        """Verifies that the Pair B benchmark manifest file exists."""
        self.assertTrue(self.pair_b_path.is_file(), f"Missing Pair B manifest at: {self.pair_b_path}")

    def test_pair_a_manifest_e2e_validation(self):
        """Validates Pair A manifest through all three validation layers (A, B, C)."""
        manifest = load_and_validate_manifest(self.pair_a_path, base_dir=self.project_root)
        self.assertIsInstance(manifest, BenchmarkManifest)

        # Metadata checks
        self.assertEqual(manifest.schema_version, "2.0.0")
        self.assertEqual(manifest.metadata.pair_id, "PAIR_A")
        self.assertEqual(manifest.metadata.domain, "Laboratory Science")
        self.assertEqual(manifest.metadata.target_document.total_pages, 1)

        # Parent requirements structure
        self.assertEqual(len(manifest.parent_requirements), 7)
        parent_ids = [p.parent_id for p in manifest.parent_requirements]
        self.assertEqual(parent_ids, [f"A{i}" for i in range(1, 8)])

        # Atomic sub-requirements
        all_subreqs = [sr for p in manifest.parent_requirements for sr in p.sub_requirements]
        self.assertEqual(len(all_subreqs), 10)
        self.assertTrue(all(sr.is_mandatory for sr in all_subreqs))

        # Expected document compliance
        comp = manifest.expected_document_compliance
        self.assertEqual(comp.mandatory_total, 10)
        self.assertEqual(comp.mandatory_fulfilled, 9)
        self.assertEqual(comp.mandatory_coverage_pct, 90.0)
        self.assertEqual(comp.optional_total, 0)
        self.assertEqual(comp.optional_fulfilled, 0)
        self.assertIsNone(comp.optional_coverage_pct)
        self.assertEqual(comp.verdict, BenchmarkDocumentComplianceVerdict.NON_COMPLIANT)

        # Status counts
        counts = comp.expected_status_counts
        self.assertEqual(counts.fulfilled, 9)
        self.assertEqual(counts.explicit_omission, 1)
        self.assertEqual(counts.contradicted, 0)
        self.assertEqual(counts.missing, 0)
        self.assertEqual(counts.needs_review, 0)

        # Safety section non-compliance check
        a7 = [p for p in manifest.parent_requirements if p.parent_id == "A7"][0]
        self.assertEqual(a7.expected_compliance_status, BenchmarkParentComplianceStatus.NON_COMPLIANT)
        self.assertEqual(a7.sub_requirements[0].ground_truth.verification.expected_status, BenchmarkVerificationStatus.EXPLICIT_OMISSION)

    def test_pair_b_manifest_e2e_validation(self):
        """Validates Pair B manifest through all three validation layers (A, B, C)."""
        manifest = load_and_validate_manifest(self.pair_b_path, base_dir=self.project_root)
        self.assertIsInstance(manifest, BenchmarkManifest)

        # Metadata checks
        self.assertEqual(manifest.schema_version, "2.0.0")
        self.assertEqual(manifest.metadata.pair_id, "PAIR_B")
        self.assertEqual(manifest.metadata.domain, "Software Engineering")
        self.assertEqual(manifest.metadata.target_document.total_pages, 1)

        # Parent requirements structure
        self.assertEqual(len(manifest.parent_requirements), 7)
        parent_ids = [p.parent_id for p in manifest.parent_requirements]
        self.assertEqual(parent_ids, [f"B{i}" for i in range(1, 8)])

        # Atomic sub-requirements
        all_subreqs = [sr for p in manifest.parent_requirements for sr in p.sub_requirements]
        self.assertEqual(len(all_subreqs), 12)

        m_subreqs = [sr for sr in all_subreqs if sr.is_mandatory]
        o_subreqs = [sr for sr in all_subreqs if not sr.is_mandatory]
        self.assertEqual(len(m_subreqs), 10)
        self.assertEqual(len(o_subreqs), 2)

        # Expected document compliance
        comp = manifest.expected_document_compliance
        self.assertEqual(comp.mandatory_total, 10)
        self.assertEqual(comp.mandatory_fulfilled, 8)
        self.assertEqual(comp.mandatory_coverage_pct, 80.0)
        self.assertEqual(comp.optional_total, 2)
        self.assertEqual(comp.optional_fulfilled, 0)
        self.assertEqual(comp.optional_coverage_pct, 0.0)
        self.assertEqual(comp.verdict, BenchmarkDocumentComplianceVerdict.CONDITIONALLY_COMPLIANT)

        # Status counts
        counts = comp.expected_status_counts
        self.assertEqual(counts.fulfilled, 8)
        self.assertEqual(counts.explicit_omission, 3)
        self.assertEqual(counts.contradicted, 0)
        self.assertEqual(counts.missing, 0)
        self.assertEqual(counts.needs_review, 1)

        # B5 safety parent has NEEDS_REVIEW
        b5 = [p for p in manifest.parent_requirements if p.parent_id == "B5"][0]
        self.assertEqual(b5.expected_compliance_status, BenchmarkParentComplianceStatus.NEEDS_REVIEW)
        self.assertEqual(b5.sub_requirements[0].ground_truth.verification.expected_status, BenchmarkVerificationStatus.NEEDS_REVIEW)
        self.assertEqual(b5.sub_requirements[1].ground_truth.verification.expected_status, BenchmarkVerificationStatus.EXPLICIT_OMISSION)

        # B6 acceptance criteria parent is NON_COMPLIANT
        b6 = [p for p in manifest.parent_requirements if p.parent_id == "B6"][0]
        self.assertEqual(b6.expected_compliance_status, BenchmarkParentComplianceStatus.NON_COMPLIANT)
        self.assertEqual(b6.sub_requirements[0].ground_truth.verification.expected_status, BenchmarkVerificationStatus.EXPLICIT_OMISSION)

    def test_pair_a_tampered_excerpt_fails_integrity(self):
        """Verifies that tampering with an excerpt in Pair A triggers Layer C BenchmarkIntegrityError."""
        with open(self.pair_a_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        raw["parent_requirements"][0]["sub_requirements"][0]["ground_truth"]["retrieval"]["expected_evidence"][0]["excerpt"] = (
            "This corrupted excerpt does not exist in the cooling report."
        )

        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkIntegrityError) as ctx:
            validate_document_integrity(manifest, base_dir=self.project_root)
        self.assertIn("evidence excerpt not found in block", str(ctx.exception))

    def test_pair_b_tampered_sha256_fails_integrity(self):
        """Verifies that altering the declared target SHA-256 triggers Layer C BenchmarkIntegrityError."""
        with open(self.pair_b_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        raw["metadata"]["target_document"]["sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"

        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkIntegrityError) as ctx:
            validate_document_integrity(manifest, base_dir=self.project_root)
        self.assertIn("Target PDF SHA-256 mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

