"""
Unit and integration tests for Three-Layer Benchmark Manifest Validator (Phase 2.6.2.2 - Step 3).
Covers:
- Layer A: JSON Schema Draft 2020-12 validation.
- Layer B: Python cross-field logic, invariants, parent aggregation, and document verdicts.
- Layer C: Physical document integrity, SHA-256 digests, and block text grounding.
"""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any

from src.benchmarks.manifest_validator import (
    validate_schema,
    validate_cross_field_rules,
    validate_document_integrity,
    load_and_validate_manifest,
    BenchmarkSchemaError,
    BenchmarkCrossFieldError,
    BenchmarkIntegrityError,
)
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


class TestBenchmarkManifestValidator(unittest.TestCase):
    """Comprehensive test suite for the three-layer manifest validator."""

    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent.parent
        cls.ref_pdf_path = cls.base_dir / "data" / "test_packs" / "A_reference_lab_report_template.pdf"
        cls.tgt_pdf_path = cls.base_dir / "data" / "test_packs" / "A_target_lab_report.pdf"

        # Verified hashes for Pair A
        cls.ref_sha = "b8256074931289363371bd6f4b5948015a7df5784217f2c066167b0dbfa7de42"
        cls.tgt_sha = "c8dfb2b85907c5ebba2c7c915621a0d988cc81314effd43143c2cf7b276dee50"

    def _get_valid_raw_manifest(self) -> Dict[str, Any]:
        """Constructs a valid raw dictionary conforming to Draft 2020-12 schema."""
        return {
            "schema_version": "2.0.0",
            "metadata": {
                "pair_id": "PAIR_VALID",
                "domain": "Laboratory Science",
                "title": "Valid Manifest Title",
                "description": "Valid manifest description",
                "reference_document": {
                    "path": "data/test_packs/A_reference_lab_report_template.pdf",
                    "filename": "A_reference_lab_report_template.pdf",
                    "sha256": self.ref_sha
                },
                "target_document": {
                    "path": "data/test_packs/A_target_lab_report.pdf",
                    "filename": "A_target_lab_report.pdf",
                    "sha256": self.tgt_sha,
                    "total_pages": 1
                },
                "tags": ["test", "laboratory"]
            },
            "parent_requirements": [
                {
                    "parent_id": "A1",
                    "section_name": "1. Aim",
                    "topic_label": "Purpose and Aim",
                    "description": "State the purpose of the experiment.",
                    "is_mandatory": True,
                    "expected_compliance_status": "FULLY_COMPLIANT",
                    "sub_requirements": [
                        {
                            "sub_id": "A1.1",
                            "claim_text": "The document states the purpose or aim of the experiment.",
                            "expected_evidence_type": "FACTUAL_STATEMENT",
                            "is_mandatory": True,
                            "decomposition_rationale": "Aim statement.",
                            "ground_truth": {
                                "retrieval": {
                                    "evidence_presence": "EXISTS",
                                    "expected_evidence": [
                                        {
                                            "block_id": "p1_b3",
                                            "page": 1,
                                            "excerpt": "The aim was to observe how the temperature of warm water changes"
                                        }
                                    ]
                                },
                                "verification": {
                                    "expected_status": "FULFILLED",
                                    "expected_rationale_notes": "Aim statement verified in overview."
                                }
                            }
                        }
                    ]
                }
            ],
            "expected_document_compliance": {
                "verdict": "COMPLIANT",
                "mandatory_coverage_pct": 100.0,
                "optional_coverage_pct": None,
                "mandatory_fulfilled": 1,
                "mandatory_total": 1,
                "optional_fulfilled": 0,
                "optional_total": 0,
                "expected_status_counts": {
                    "FULFILLED": 1,
                    "EXPLICIT_OMISSION": 0,
                    "CONTRADICTED": 0,
                    "MISSING": 0,
                    "NEEDS_REVIEW": 0
                }
            }
        }

    # =======================================================================
    # Layer A: JSON Schema Validation Tests
    # =======================================================================

    def test_layer_a_valid_manifest_passes(self):
        raw = self._get_valid_raw_manifest()
        validate_schema(raw)

    def test_layer_a_missing_required_field_fails(self):
        raw = self._get_valid_raw_manifest()
        del raw["metadata"]["pair_id"]
        with self.assertRaises(BenchmarkSchemaError) as ctx:
            validate_schema(raw)
        self.assertIn("pair_id", str(ctx.exception))

    def test_layer_a_invalid_enum_fails(self):
        raw = self._get_valid_raw_manifest()
        raw["parent_requirements"][0]["sub_requirements"][0]["ground_truth"]["verification"]["expected_status"] = "INVALID_STATUS"
        with self.assertRaises(BenchmarkSchemaError) as ctx:
            validate_schema(raw)
        self.assertIn("INVALID_STATUS", str(ctx.exception))

    def test_layer_a_invalid_sha256_format_fails(self):
        raw = self._get_valid_raw_manifest()
        raw["metadata"]["reference_document"]["sha256"] = "NOT_A_VALID_HEX"
        with self.assertRaises(BenchmarkSchemaError) as ctx:
            validate_schema(raw)
        self.assertIn("sha256", str(ctx.exception))

    def test_layer_a_mandatory_total_zero_fails(self):
        raw = self._get_valid_raw_manifest()
        raw["expected_document_compliance"]["mandatory_total"] = 0
        with self.assertRaises(BenchmarkSchemaError) as ctx:
            validate_schema(raw)
        self.assertIn("mandatory_total", str(ctx.exception))

    # =======================================================================
    # Layer B: Python Cross-Field Validation Tests
    # =======================================================================

    def test_layer_b_duplicate_parent_id_fails(self):
        raw = self._get_valid_raw_manifest()
        # Add duplicate parent A1
        dup_parent = copy.deepcopy(raw["parent_requirements"][0])
        dup_parent["sub_requirements"][0]["sub_id"] = "A1.2"
        raw["parent_requirements"].append(dup_parent)
        raw["expected_document_compliance"]["mandatory_total"] = 2
        raw["expected_document_compliance"]["mandatory_fulfilled"] = 2
        raw["expected_document_compliance"]["expected_status_counts"]["FULFILLED"] = 2

        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkCrossFieldError) as ctx:
            validate_cross_field_rules(manifest)
        self.assertIn("Duplicate parent ID 'A1'", str(ctx.exception))

    def test_layer_b_duplicate_sub_id_fails(self):
        raw = self._get_valid_raw_manifest()
        # Add duplicate sub_id A1.1 in parent A1
        dup_sr = copy.deepcopy(raw["parent_requirements"][0]["sub_requirements"][0])
        raw["parent_requirements"][0]["sub_requirements"].append(dup_sr)
        raw["expected_document_compliance"]["mandatory_total"] = 2
        raw["expected_document_compliance"]["mandatory_fulfilled"] = 2
        raw["expected_document_compliance"]["expected_status_counts"]["FULFILLED"] = 2

        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkCrossFieldError) as ctx:
            validate_cross_field_rules(manifest)
        self.assertIn("Duplicate sub-requirement ID 'A1.1'", str(ctx.exception))

    def test_layer_b_sub_id_hierarchy_mismatch_fails(self):
        raw = self._get_valid_raw_manifest()
        raw["parent_requirements"][0]["sub_requirements"][0]["sub_id"] = "B1.1"  # Under parent A1
        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkCrossFieldError) as ctx:
            validate_cross_field_rules(manifest)
        self.assertIn("does not follow parent ID prefix 'A1.'", str(ctx.exception))

    def test_layer_b_empty_parent_subrequirements_fails(self):
        raw = self._get_valid_raw_manifest()
        raw["parent_requirements"][0]["sub_requirements"] = []
        # Construct directly without schema validation to test Layer B
        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkCrossFieldError) as ctx:
            validate_cross_field_rules(manifest)
        self.assertIn("contains zero sub-requirements", str(ctx.exception))

    def test_layer_b_parent_mandatory_inconsistency_fails(self):
        raw = self._get_valid_raw_manifest()
        # Parent declared False, but child is True
        raw["parent_requirements"][0]["is_mandatory"] = False
        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkCrossFieldError) as ctx:
            validate_cross_field_rules(manifest)
        self.assertIn("Parent 'A1' has is_mandatory=False, but child requirements have is_mandatory=True", str(ctx.exception))

    def test_layer_b_zero_mandatory_atomic_requirements_fails(self):
        raw = self._get_valid_raw_manifest()
        raw["parent_requirements"][0]["is_mandatory"] = False
        raw["parent_requirements"][0]["sub_requirements"][0]["is_mandatory"] = False
        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkCrossFieldError) as ctx:
            validate_cross_field_rules(manifest)
        self.assertIn("must contain at least one mandatory atomic sub-requirement", str(ctx.exception))

    def test_layer_b_incompatible_evidence_presence_exists_with_missing(self):
        raw = self._get_valid_raw_manifest()
        raw["parent_requirements"][0]["sub_requirements"][0]["ground_truth"]["retrieval"]["evidence_presence"] = "EXISTS"
        raw["parent_requirements"][0]["sub_requirements"][0]["ground_truth"]["verification"]["expected_status"] = "MISSING"
        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkCrossFieldError) as ctx:
            validate_cross_field_rules(manifest)
        self.assertIn("incompatible expected_status='MISSING'", str(ctx.exception))

    def test_layer_b_exists_with_empty_evidence_fails(self):
        raw = self._get_valid_raw_manifest()
        raw["parent_requirements"][0]["sub_requirements"][0]["ground_truth"]["retrieval"]["expected_evidence"] = []
        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkCrossFieldError) as ctx:
            validate_cross_field_rules(manifest)
        self.assertIn("EXISTS requires at least one evidence annotation", str(ctx.exception))

    def test_layer_b_explicit_omission_with_non_omission_status_fails(self):
        raw = self._get_valid_raw_manifest()
        raw["parent_requirements"][0]["sub_requirements"][0]["ground_truth"]["retrieval"]["evidence_presence"] = "EXPLICIT_OMISSION"
        raw["parent_requirements"][0]["sub_requirements"][0]["ground_truth"]["verification"]["expected_status"] = "FULFILLED"
        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkCrossFieldError) as ctx:
            validate_cross_field_rules(manifest)
        self.assertIn("incompatible expected_status='FULFILLED'", str(ctx.exception))

    def test_layer_b_explicit_omission_with_empty_evidence_fails(self):
        raw = self._get_valid_raw_manifest()
        raw["parent_requirements"][0]["sub_requirements"][0]["ground_truth"]["retrieval"]["evidence_presence"] = "EXPLICIT_OMISSION"
        raw["parent_requirements"][0]["sub_requirements"][0]["ground_truth"]["verification"]["expected_status"] = "EXPLICIT_OMISSION"
        raw["parent_requirements"][0]["sub_requirements"][0]["ground_truth"]["retrieval"]["expected_evidence"] = []
        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkCrossFieldError) as ctx:
            validate_cross_field_rules(manifest)
        self.assertIn("EXPLICIT_OMISSION requires at least one disclaimer evidence annotation", str(ctx.exception))

    def test_layer_b_absent_with_non_empty_evidence_fails(self):
        raw = self._get_valid_raw_manifest()
        raw["parent_requirements"][0]["sub_requirements"][0]["ground_truth"]["retrieval"]["evidence_presence"] = "ABSENT"
        raw["parent_requirements"][0]["sub_requirements"][0]["ground_truth"]["verification"]["expected_status"] = "MISSING"
        # Non-empty evidence:
        raw["parent_requirements"][0]["sub_requirements"][0]["ground_truth"]["retrieval"]["expected_evidence"] = [
            {"block_id": "p1_b1", "page": 1}
        ]
        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkCrossFieldError) as ctx:
            validate_cross_field_rules(manifest)
        self.assertIn("ABSENT requires expected_evidence to be strictly empty", str(ctx.exception))

    def test_layer_b_needs_review_with_exists_and_absent(self):
        # 1. NEEDS_REVIEW with EXISTS and non-empty evidence passes
        raw1 = self._get_valid_raw_manifest()
        raw1["parent_requirements"][0]["expected_compliance_status"] = "NEEDS_REVIEW"
        raw1["parent_requirements"][0]["sub_requirements"][0]["ground_truth"]["retrieval"]["evidence_presence"] = "EXISTS"
        raw1["parent_requirements"][0]["sub_requirements"][0]["ground_truth"]["verification"]["expected_status"] = "NEEDS_REVIEW"
        raw1["expected_document_compliance"]["verdict"] = "CONDITIONALLY_COMPLIANT"
        raw1["expected_document_compliance"]["mandatory_fulfilled"] = 0
        raw1["expected_document_compliance"]["mandatory_coverage_pct"] = 0.0
        raw1["expected_document_compliance"]["expected_status_counts"]["FULFILLED"] = 0
        raw1["expected_document_compliance"]["expected_status_counts"]["NEEDS_REVIEW"] = 1

        manifest1 = BenchmarkManifest.from_dict(raw1)
        validate_cross_field_rules(manifest1)  # Should pass without error

        # 2. NEEDS_REVIEW with ABSENT and empty evidence passes
        raw2 = copy.deepcopy(raw1)
        raw2["parent_requirements"][0]["sub_requirements"][0]["ground_truth"]["retrieval"]["evidence_presence"] = "ABSENT"
        raw2["parent_requirements"][0]["sub_requirements"][0]["ground_truth"]["retrieval"]["expected_evidence"] = []
        manifest2 = BenchmarkManifest.from_dict(raw2)
        validate_cross_field_rules(manifest2)  # Should pass without error

    def test_layer_b_parent_aggregation_mandatory_rules(self):
        # 1. PARTIALLY_COMPLIANT mismatch
        raw = self._get_valid_raw_manifest()
        # Add A1.2 with MISSING
        sr2 = copy.deepcopy(raw["parent_requirements"][0]["sub_requirements"][0])
        sr2["sub_id"] = "A1.2"
        sr2["ground_truth"]["retrieval"]["evidence_presence"] = "ABSENT"
        sr2["ground_truth"]["retrieval"]["expected_evidence"] = []
        sr2["ground_truth"]["verification"]["expected_status"] = "MISSING"
        raw["parent_requirements"][0]["sub_requirements"].append(sr2)

        # Parent has 1 FULFILLED, 1 MISSING -> Should be PARTIALLY_COMPLIANT
        raw["parent_requirements"][0]["expected_compliance_status"] = "FULLY_COMPLIANT"  # Wrong!
        raw["expected_document_compliance"]["mandatory_total"] = 2
        raw["expected_document_compliance"]["mandatory_fulfilled"] = 1
        raw["expected_document_compliance"]["mandatory_coverage_pct"] = 50.0
        raw["expected_document_compliance"]["verdict"] = "NON_COMPLIANT"
        raw["expected_document_compliance"]["expected_status_counts"]["FULFILLED"] = 1
        raw["expected_document_compliance"]["expected_status_counts"]["MISSING"] = 1

        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkCrossFieldError) as ctx:
            validate_cross_field_rules(manifest)
        self.assertIn("calculated aggregation rule yielded 'PARTIALLY_COMPLIANT'", str(ctx.exception))

    def test_layer_b_optional_only_parent_rules(self):
        # Test optional-only parent waterfall:
        # FULFILLED + NEEDS_REVIEW + MISSING -> Must be NEEDS_REVIEW
        raw = self._get_valid_raw_manifest()

        # Add optional-only parent OPT1
        opt_parent = {
            "parent_id": "OPT1",
            "section_name": "Optional Section",
            "topic_label": "Guidelines",
            "description": "Optional recommendations",
            "is_mandatory": False,
            "expected_compliance_status": "NEEDS_REVIEW",
            "sub_requirements": [
                {
                    "sub_id": "OPT1.1",
                    "claim_text": "Recommendation 1",
                    "expected_evidence_type": "FACTUAL_STATEMENT",
                    "is_mandatory": False,
                    "decomposition_rationale": "Rationale",
                    "ground_truth": {
                        "retrieval": {"evidence_presence": "EXISTS", "expected_evidence": [{"block_id": "p1_b3", "page": 1}]},
                        "verification": {"expected_status": "FULFILLED", "expected_rationale_notes": "Note"}
                    }
                },
                {
                    "sub_id": "OPT1.2",
                    "claim_text": "Recommendation 2",
                    "expected_evidence_type": "FACTUAL_STATEMENT",
                    "is_mandatory": False,
                    "decomposition_rationale": "Rationale",
                    "ground_truth": {
                        "retrieval": {"evidence_presence": "EXISTS", "expected_evidence": [{"block_id": "p1_b3", "page": 1}]},
                        "verification": {"expected_status": "NEEDS_REVIEW", "expected_rationale_notes": "Note"}
                    }
                },
                {
                    "sub_id": "OPT1.3",
                    "claim_text": "Recommendation 3",
                    "expected_evidence_type": "FACTUAL_STATEMENT",
                    "is_mandatory": False,
                    "decomposition_rationale": "Rationale",
                    "ground_truth": {
                        "retrieval": {"evidence_presence": "ABSENT", "expected_evidence": []},
                        "verification": {"expected_status": "MISSING", "expected_rationale_notes": "Note"}
                    }
                }
            ]
        }
        raw["parent_requirements"].append(opt_parent)
        raw["expected_document_compliance"]["optional_total"] = 3
        raw["expected_document_compliance"]["optional_fulfilled"] = 1
        raw["expected_document_compliance"]["optional_coverage_pct"] = 33.3
        raw["expected_document_compliance"]["expected_status_counts"]["FULFILLED"] = 2
        raw["expected_document_compliance"]["expected_status_counts"]["NEEDS_REVIEW"] = 1
        raw["expected_document_compliance"]["expected_status_counts"]["MISSING"] = 1

        # Optional NEEDS_REVIEW must NOT affect mandatory coverage or verdict!
        manifest = BenchmarkManifest.from_dict(raw)
        validate_cross_field_rules(manifest)  # Successfully passes!

        # Now test that setting OPT1 to PARTIALLY_COMPLIANT fails (NEEDS_REVIEW takes priority)
        raw["parent_requirements"][1]["expected_compliance_status"] = "PARTIALLY_COMPLIANT"
        manifest_bad = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkCrossFieldError) as ctx:
            validate_cross_field_rules(manifest_bad)
        self.assertIn("calculated aggregation rule yielded 'NEEDS_REVIEW'", str(ctx.exception))

    def test_layer_b_document_verdict_consistency_rules(self):
        raw = self._get_valid_raw_manifest()
        # 100% coverage, 0 review, 0 contra -> Should be COMPLIANT, but manifest declares NON_COMPLIANT
        raw["expected_document_compliance"]["verdict"] = "NON_COMPLIANT"
        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkCrossFieldError) as ctx:
            validate_cross_field_rules(manifest)
        self.assertIn("does not match calculated verdict='COMPLIANT'", str(ctx.exception))

    def test_layer_b_coverage_calculation_mismatch_fails(self):
        raw = self._get_valid_raw_manifest()
        raw["expected_document_compliance"]["mandatory_coverage_pct"] = 99.0  # Actually 100.0
        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkCrossFieldError) as ctx:
            validate_cross_field_rules(manifest)
        self.assertIn("mandatory_coverage_pct=99.0 does not match recalculated value", str(ctx.exception))

    # =======================================================================
    # Layer C: Document Integrity & Evidence Grounding Tests
    # =======================================================================

    def test_layer_c_valid_document_integrity_passes(self):
        raw = self._get_valid_raw_manifest()
        manifest = BenchmarkManifest.from_dict(raw)
        # Validate against actual repository files
        validate_document_integrity(manifest, base_dir=self.base_dir)

    def test_layer_c_missing_reference_pdf_fails(self):
        raw = self._get_valid_raw_manifest()
        raw["metadata"]["reference_document"]["path"] = "data/test_packs/non_existent_ref.pdf"
        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkIntegrityError) as ctx:
            validate_document_integrity(manifest, base_dir=self.base_dir)
        self.assertIn("Reference template PDF not found", str(ctx.exception))

    def test_layer_c_missing_target_pdf_fails(self):
        raw = self._get_valid_raw_manifest()
        raw["metadata"]["target_document"]["path"] = "data/test_packs/non_existent_tgt.pdf"
        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkIntegrityError) as ctx:
            validate_document_integrity(manifest, base_dir=self.base_dir)
        self.assertIn("Target document PDF not found", str(ctx.exception))

    def test_layer_c_incorrect_reference_sha256_fails(self):
        raw = self._get_valid_raw_manifest()
        raw["metadata"]["reference_document"]["sha256"] = "0" * 64
        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkIntegrityError) as ctx:
            validate_document_integrity(manifest, base_dir=self.base_dir)
        self.assertIn("Reference PDF SHA-256 mismatch", str(ctx.exception))

    def test_layer_c_incorrect_target_sha256_fails(self):
        raw = self._get_valid_raw_manifest()
        raw["metadata"]["target_document"]["sha256"] = "0" * 64
        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkIntegrityError) as ctx:
            validate_document_integrity(manifest, base_dir=self.base_dir)
        self.assertIn("Target PDF SHA-256 mismatch", str(ctx.exception))

    def test_layer_c_incorrect_target_page_count_fails(self):
        raw = self._get_valid_raw_manifest()
        raw["metadata"]["target_document"]["total_pages"] = 99  # Actual is 1
        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkIntegrityError) as ctx:
            validate_document_integrity(manifest, base_dir=self.base_dir)
        self.assertIn("total_pages mismatch", str(ctx.exception))

    def test_layer_c_unknown_evidence_block_id_fails(self):
        raw = self._get_valid_raw_manifest()
        raw["parent_requirements"][0]["sub_requirements"][0]["ground_truth"]["retrieval"]["expected_evidence"][0]["block_id"] = "p1_b999"
        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkIntegrityError) as ctx:
            validate_document_integrity(manifest, base_dir=self.base_dir)
        self.assertIn("references unknown block_id 'p1_b999'", str(ctx.exception))

    def test_layer_c_incorrect_evidence_page_fails(self):
        raw = self._get_valid_raw_manifest()
        # p1_b3 is on page 1, set page to 2
        raw["parent_requirements"][0]["sub_requirements"][0]["ground_truth"]["retrieval"]["expected_evidence"][0]["page"] = 2
        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkIntegrityError) as ctx:
            validate_document_integrity(manifest, base_dir=self.base_dir)
        self.assertIn("references block_id 'p1_b3' on page 2, but the block is actually located on page 1", str(ctx.exception))

    def test_layer_c_evidence_excerpt_not_found_fails(self):
        raw = self._get_valid_raw_manifest()
        raw["parent_requirements"][0]["sub_requirements"][0]["ground_truth"]["retrieval"]["expected_evidence"][0]["excerpt"] = "Completely non-existent text in this block"
        manifest = BenchmarkManifest.from_dict(raw)
        with self.assertRaises(BenchmarkIntegrityError) as ctx:
            validate_document_integrity(manifest, base_dir=self.base_dir)
        self.assertIn("evidence excerpt not found in block 'p1_b3'", str(ctx.exception))

    def test_layer_c_evidence_excerpt_normalized_whitespace_passes(self):
        raw = self._get_valid_raw_manifest()
        # Change casing and whitespace of excerpt
        raw["parent_requirements"][0]["sub_requirements"][0]["ground_truth"]["retrieval"]["expected_evidence"][0]["excerpt"] = "THE   AIM  WAS   TO OBSERVE"
        manifest = BenchmarkManifest.from_dict(raw)
        # Should pass because validator normalizes whitespace and lowers case
        validate_document_integrity(manifest, base_dir=self.base_dir)

    # =======================================================================
    # End-to-End Validator Entry Point Tests
    # =======================================================================

    def test_load_and_validate_manifest_e2e_success(self):
        raw = self._get_valid_raw_manifest()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(raw, f)
            temp_path = f.name

        try:
            validated = load_and_validate_manifest(
                path=temp_path,
                base_dir=self.base_dir,
                check_document_integrity=True
            )
            self.assertIsInstance(validated, BenchmarkManifest)
            self.assertEqual(validated.metadata.pair_id, "PAIR_VALID")
            self.assertEqual(validated.expected_document_compliance.verdict, BenchmarkDocumentComplianceVerdict.COMPLIANT)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()

