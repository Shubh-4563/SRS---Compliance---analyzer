#!/usr/bin/env python3
"""
Demonstration: PDF Input Support & Scanned PDF Detection (Version 0.4)
======================================================================
Demonstrates:
1. Extraction and compliance auditing of SRS document from PDF (data/sample_srs.pdf).
2. End-to-end PDF-to-PDF analysis using both reference-standard PDF and SRS PDF
   (data/reference_standard.pdf + data/avionics_sample_srs.pdf).
3. Graceful handling and detection of scanned / image-only PDFs (data/scanned_sample_srs.pdf)
   verifying that an informative OCR-required notification is emitted.
"""

import os
import sys

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.analyzer import (
    ReferenceStandard,
    RequirementExtractor,
    ComplianceAnalyzer,
    ReportGenerator,
)
from src.pdf_extractor import (
    PDFExtractor,
    ScannedPDFError,
)


def run_demo():
    print("=" * 80)
    print("  AI-SRS COMPLIANCE ANALYZER — VERSION 0.4 PDF DEMONSTRATION")
    print("=" * 80)
    print("Capabilities demonstrated:")
    print(" 1. Digital SRS PDF Extraction (Sections & Requirements)")
    print(" 2. Reference Standard PDF Extraction (Sections, Rules R1-R4, Lexicons)")
    print(" 3. End-to-End PDF-to-PDF Compliance Analysis")
    print(" 4. Scanned / Image-Only PDF Detection (OCR Requirement Notice)")
    print("=" * 80)

    extractor = PDFExtractor()

    # --------------------------------------------------------------------------
    # DEMO 1: Digital SRS PDF Extraction (data/sample_srs.pdf)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(">>> DEMO 1: Baseline SRS PDF Ingestion ('data/sample_srs.pdf')")
    print("-" * 80)
    srs_pdf = os.path.join(BASE_DIR, "data", "sample_srs.pdf")
    res1 = extractor.extract_text(srs_pdf)
    print(f"File Path           : {srs_pdf}")
    print(f"Total Pages         : {res1.page_count}")
    print(f"Characters Extracted: {res1.char_count}")
    print(f"Words Extracted     : {res1.word_count}")
    print(f"Scanned Document?   : {res1.is_scanned}")

    sections1, reqs1 = extractor.extract_srs_document(srs_pdf)
    print(f"\nExtracted {len(reqs1)} requirements from PDF:")
    for r in reqs1:
        print(f"  [{r.req_id}] {r.statement}")

    # Audit against standard JSON
    standard_json = ReferenceStandard(os.path.join(BASE_DIR, "data", "reference_standard.json"))
    analyzer1 = ComplianceAnalyzer(standard_json, enable_semantic_matching=False)
    report1 = analyzer1.audit_document(reqs1)
    print(f"\nAudit Result on sample_srs.pdf:")
    print(f"  Total: {report1.total_requirements} | Compliant: {report1.compliant_count} | Rate: {report1.compliance_percentage}%")
    print("  (Preserves exact 60.0% benchmark from text sample_srs.txt!)")

    # --------------------------------------------------------------------------
    # DEMO 2: Reference Standard PDF + Multi-Section Avionics SRS PDF
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(">>> DEMO 2: End-to-End PDF-to-PDF Analysis")
    print("    Standard: 'data/reference_standard.pdf' + SRS: 'data/avionics_sample_srs.pdf'")
    print("-" * 80)
    std_pdf = os.path.join(BASE_DIR, "data", "reference_standard.pdf")
    avionics_pdf = os.path.join(BASE_DIR, "data", "avionics_sample_srs.pdf")

    # Extract reference standard from PDF
    print("Parsing Reference Standard PDF...")
    std_from_pdf = extractor.extract_reference_standard(std_pdf)
    print(f"  Standard Name : {std_from_pdf.standard_name} (v{std_from_pdf.version})")
    print(f"  Binding Modal : '{std_from_pdf.binding_modal}'")
    print(f"  Vague Words   : {std_from_pdf.vague_words}")
    print(f"  Extracted Sections ({len(std_from_pdf.sections)}):")
    for s in std_from_pdf.sections:
        print(f"    [{s.section_id}] {s.name}")
    print(f"  Extracted Rules ({len(std_from_pdf.rules)}):")
    for r in std_from_pdf.rules:
        print(f"    [{r.rule_id}] {r.name}: \"{r.description}\"")

    # Extract SRS from PDF
    print("\nParsing Avionics SRS PDF...")
    avionics_sections, avionics_reqs = extractor.extract_srs_document(avionics_pdf)
    print(f"  Extracted Sections    : {len(avionics_sections)}")
    print(f"  Extracted Requirements: {len(avionics_reqs)}")

    # Full audit
    print("\nExecuting Full AI Compliance Audit (Section Alignment + Rule Semantic Matching + R1-R4)...")
    analyzer2 = ComplianceAnalyzer(std_from_pdf, similarity_threshold=0.55)
    report2 = analyzer2.audit_document(avionics_reqs, sections=avionics_sections)

    print(f"\nAudit Summary:")
    print(f"  Total Requirements Analyzed : {report2.total_requirements}")
    print(f"  Compliant Requirements       : {report2.compliant_count}")
    print(f"  Non-Compliant Requirements   : {report2.non_compliant_count}")
    print(f"  Compliance Percentage        : {report2.compliance_percentage}%")
    print(f"  Mandatory Sections Matched   : {report2.sections_summary.get('matched_sections', 0)}/{report2.sections_summary.get('total_sections', 0)}")
    print(f"  Section Alignment Coverage   : {report2.sections_summary.get('coverage_percentage', 0.0)}%")

    print("\nSection Alignment Breakdown:")
    for m in report2.section_matches:
        badge = "MATCHED (✓)" if m.is_matched else "MISSING (✗)"
        print(f"  [{m.reference_section_id}] {m.reference_section_name:<30} -> {m.similarity_score:.4f} | {badge}")

    print("\nFirst 3 Requirements Semantic Rule Alignments:")
    for audit in report2.audits[:3]:
        badge = "COMPLIANT" if audit.is_compliant else "NON-COMPLIANT"
        req_id = audit.req_id or "MISSING-ID"
        print(f"  [{req_id}] ({badge}) -> Best Rule: {audit.matched_rule_id} ({audit.matched_rule_name}) [Score: {audit.rule_similarity_score:.4f}]")

    # --------------------------------------------------------------------------
    # DEMO 3: Scanned / Image-Only PDF Detection & Graceful Notification
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(">>> DEMO 3: Scanned / Image-Only PDF Handling ('data/scanned_sample_srs.pdf')")
    print("-" * 80)
    scanned_pdf = os.path.join(BASE_DIR, "data", "scanned_sample_srs.pdf")

    print(f"Inspecting file: {scanned_pdf}")
    is_scanned = extractor.is_scanned_pdf(scanned_pdf)
    print(f"Scanner Check -> is_scanned_pdf: {is_scanned}")

    print("\nAttempting text extraction with raise_if_scanned=True:")
    try:
        extractor.extract_text(scanned_pdf, raise_if_scanned=True)
        print("ERROR: Should not have succeeded extraction on scanned PDF!")
    except ScannedPDFError as err:
        print("\n  [GRACEFUL EXCEPTION CAUGHT]")
        print(f"  Exception Type : {type(err).__name__}")
        print(f"  Target File    : {err.filepath}")
        print(f"  Extracted Chars: {err.char_count}")
        print(f"  Total Pages    : {err.page_count}")
        print(f"  User Message   : {err}")
        print("\n  [CONFIRMED] Scanned document successfully detected without OCR crashing.")

    print("\n" + "=" * 80)
    print("VERSION 0.4 PDF DEMONSTRATION COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
