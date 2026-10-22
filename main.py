#!/usr/bin/env python3
"""
SRS Compliance Analyzer - Version 0.4
======================================
CLI entry point to execute requirement extraction (from TXT, MD, or PDF),
semantic section alignment, semantic requirement-to-rule matching
(sentence-transformers / all-MiniLM-L6-v2), and deterministic compliance checks (R1 - R4).
Supports PDF reference standards and SRS documents with graceful scanned PDF detection.
"""

import os
import sys
import json
import argparse

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.analyzer import (
    ReferenceStandard,
    RequirementExtractor,
    ComplianceAnalyzer,
    ReportGenerator,
    SemanticSectionMatcher,
)
from src.pdf_extractor import (
    ScannedPDFError,
    PDFExtractor,
)


def run_pipeline(
    srs_path: str,
    standard_path: str,
    output_json: str = "compliance_report.json",
    output_md: str = "compliance_report.md",
    threshold: float = 0.55
):
    print("=" * 72)
    print("        AI-SRS COMPLIANCE ANALYZER — PIPELINE (VERSION 0.4)")
    print("=" * 72)
    print(f"Loading Reference Standard : {standard_path}")
    print(f"Loading SRS Document       : {srs_path}")
    print(f"Semantic Alignment Engine  : sentence-transformers (all-MiniLM-L6-v2)")
    print(f"Section & Rule Matching    : Enabled (Cosine Similarity)")
    print(f"PDF Input Support          : Enabled (pypdf digital extraction)")
    print(f"Similarity Threshold       : {threshold}")
    print("-" * 72)

    if not os.path.exists(srs_path):
        print(f"Error: SRS file '{srs_path}' not found.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(standard_path):
        print(f"Error: Standard file '{standard_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # Stage 1: Load Reference Standard (Rules R1-R4 + Mandatory Sections from JSON or PDF)
    try:
        standard = ReferenceStandard(standard_path)
    except ScannedPDFError as err:
        print("\n" + "=" * 72, file=sys.stderr)
        print("  [ALERT] SCANNED / IMAGE-ONLY REFERENCE STANDARD PDF DETECTED", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        print(f"File      : {err.filepath}", file=sys.stderr)
        print(f"Details   : Text extraction returned only {err.char_count} character(s) across {err.page_count} page(s).", file=sys.stderr)
        print("Verdict   : Little or no extractable text found in this standard document.", file=sys.stderr)
        print("Action    : Optical Character Recognition (OCR) is required to process this document.", file=sys.stderr)
        print("Note      : OCR is not implemented in Version 0.4. Please provide a digital text-based PDF or JSON standard.", file=sys.stderr)
        print("=" * 72 + "\n", file=sys.stderr)
        sys.exit(2)

    # Stage 2: Document Parsing (Sections + Requirements from TXT, MD, or PDF)
    try:
        extractor = RequirementExtractor()
        requirements = extractor.extract_from_file(srs_path)
        sections = extractor.extract_sections_from_file(srs_path)
    except ScannedPDFError as err:
        print("\n" + "=" * 72, file=sys.stderr)
        print("  [ALERT] SCANNED / IMAGE-ONLY SRS PDF DETECTED", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        print(f"File      : {err.filepath}", file=sys.stderr)
        print(f"Details   : Text extraction returned only {err.char_count} character(s) across {err.page_count} page(s).", file=sys.stderr)
        print("Verdict   : Little or no extractable text found in this SRS document.", file=sys.stderr)
        print("Action    : Optical Character Recognition (OCR) is required to process this document.", file=sys.stderr)
        print("Note      : OCR is not implemented in Version 0.4. Please provide a digital text-based PDF or text/markdown document.", file=sys.stderr)
        print("=" * 72 + "\n", file=sys.stderr)
        sys.exit(2)

    print(f"Document Parsing Completed  : {len(sections)} sections, {len(requirements)} requirements extracted.")
    print("-" * 72)

    # Stage 3: Semantic Section Alignment & Requirement-to-Rule Matching & Checks (R1 - R4)
    analyzer = ComplianceAnalyzer(standard, similarity_threshold=threshold)
    report = analyzer.audit_document(requirements, sections=sections)

    # Stage 4: Generate & Display Reports
    terminal_output = ReportGenerator.format_terminal(report)
    print(terminal_output)

    # Save structured JSON report
    json_path = os.path.join(BASE_DIR, output_json)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ReportGenerator.to_json(report), f, indent=2)

    # Save human-readable Markdown report
    md_path = os.path.join(BASE_DIR, output_md)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(ReportGenerator.format_markdown(report))

    print(f"\n[Artifact] Structured JSON report saved to: {json_path}")
    print(f"[Artifact] Markdown report saved to        : {md_path}")
    print("=" * 72)


def run_demonstration(mode: str = "pdf"):
    """
    Runs interactive demonstrations of AI compliance and PDF capabilities:
    - 'pdf': Version 0.4 PDF input support, PDF-to-PDF analysis, and scanned detection
    - 'rules': Version 0.3 semantic matching of rules against requirements
    - 'sections': Version 0.2 semantic matching of document sections
    - 'all': Runs all demonstrations sequentially
    """
    if mode in ("pdf", "all"):
        print("\n>>> Launching Version 0.4 PDF Input & Scanned Detection Demonstration...")
        from demo_pdf_support import run_demo as run_pdf_demo
        run_pdf_demo()

    if mode in ("rules", "all"):
        print("\n>>> Launching Version 0.3 Requirement-to-Rule Semantic Demonstration...")
        from demo_rule_matching import run_demo as run_rule_demo
        run_rule_demo()

    if mode in ("sections", "all"):
        print("\n>>> Launching Version 0.2 Section Alignment Semantic Demonstration...")
        from demo_semantic_matching import run_demo as run_section_demo
        run_section_demo()


def main():
    parser = argparse.ArgumentParser(
        description="Verify an SRS document (TXT, MD, or PDF) against standard compliance rules (R1-R4), semantic section alignment, and semantic rule matching (Version 0.4)."
    )
    default_srs = os.path.join(BASE_DIR, "data", "sample_srs.txt")
    default_standard = os.path.join(BASE_DIR, "data", "reference_standard.json")

    parser.add_argument(
        "--srs", "-s",
        default=default_srs,
        help=f"Path to SRS file (.txt, .md, or .pdf) (default: {default_srs})"
    )
    parser.add_argument(
        "--standard", "-r",
        default=default_standard,
        help=f"Path to reference standard file (.json or .pdf) (default: {default_standard})"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.55,
        help="Cosine similarity threshold for section semantic matching (default: 0.55)"
    )
    parser.add_argument(
        "--output-json", "-j",
        default="compliance_report.json",
        help="Filename for exported JSON audit report (default: compliance_report.json)"
    )
    parser.add_argument(
        "--output-md", "-m",
        default="compliance_report.md",
        help="Filename for exported Markdown audit report (default: compliance_report.md)"
    )
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run Version 0.4 PDF input support and scanned PDF detection demonstration"
    )
    parser.add_argument(
        "--demo-rules",
        action="store_true",
        help="Run Version 0.3 requirement-to-rule semantic similarity demonstration"
    )
    parser.add_argument(
        "--demo-sections",
        action="store_true",
        help="Run Version 0.2 section alignment semantic similarity demonstration"
    )
    parser.add_argument(
        "--demo-all",
        action="store_true",
        help="Run all demonstrations (PDF support + rule matching + section alignment)"
    )

    args = parser.parse_args()

    if args.demo_all:
        run_demonstration(mode="all")
        return
    elif args.demo_sections:
        run_demonstration(mode="sections")
        return
    elif args.demo_rules:
        run_demonstration(mode="rules")
        return
    elif args.demo:
        run_demonstration(mode="pdf")
        return

    run_pipeline(
        srs_path=args.srs,
        standard_path=args.standard,
        output_json=args.output_json,
        output_md=args.output_md,
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()


