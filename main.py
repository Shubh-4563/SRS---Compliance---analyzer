#!/usr/bin/env python3
"""
AI-SRS Compliance Analyzer
===========================
CLI entry point to verify target documents (entire PDF or single page via --page N)
against a reference template PDF.

Default Engine (V2):
- Candidate Retrieval: Dual-Engine Hybrid RRF (BM25 lexical + MiniLM dense embeddings).
- Evidence Verification: NLI cross-encoder (cross-encoder/nli-deberta-v3-small) with
  explicit omission isolation, numerical token validation, and conflict detection.
- Compliance Aggregation: Deterministic rollup of atomic verification results to
  parent points and document-level verdicts.
- Audit Reporting: Export to structured JSON and clean, table-free Markdown reports.

Legacy Engine (V1):
- Selectable via --legacy-rules for v0.1-v0.4 quality rules (R1-R4) and cosine similarity checks.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# V2 Pipeline & Reporting imports
from src.pipeline import CompliancePipeline
from src.reporting.report_formatter import ComplianceReportFormatter
from src.models.compliance import DocumentComplianceReport

# V1 Legacy imports
from src.analyzer import (
    ReferenceStandard,
    RequirementExtractor,
    ComplianceAnalyzer,
    ReportGenerator,
    SemanticSectionMatcher,
    TemplatePoint,
    ReferenceTemplate,
    TemplateContentChecker,
)
from src.pdf_extractor import (
    ScannedPDFError,
    PDFExtractor,
)


def run_v2_pipeline(
    template_path: str,
    target_path: str,
    page: Optional[int] = None,
    top_k: int = 5,
    output_json: Optional[str] = "compliance_report.json",
    output_md: Optional[str] = "compliance_report.md",
    pipeline: Optional[CompliancePipeline] = None,
    formatter: Optional[ComplianceReportFormatter] = None,
    debug: bool = False,
) -> int:
    """
    Executes Phase 2.5 V2 Reference-Guided Compliance Verification Pipeline:
    1. Validates reference and target document inputs.
    2. Decomposes reference template into atomic verifiable sub-requirements.
    3. Indexes target document into hybrid retrieval engine (BM25 + Dense + RRF).
    4. Evaluates evidence with NLI cross-encoder verifier (DeBERTa-v3).
    5. Aggregates findings into hierarchical compliance report.
    6. Exports structured JSON and Markdown audit reports (no tables) and displays terminal summary.
    """
    print("=" * 80)
    print("      AI-SRS COMPLIANCE ANALYZER — V2 REFERENCE-GUIDED VERIFICATION")
    print("=" * 80)
    print(f"Reference Template : {template_path}")
    print(f"Target Document    : {target_path}")
    if page is not None:
        print(f"Target Scope       : Single Page (Page {page})")
    else:
        print("Target Scope       : Entire Document")
    print("Retrieval Engine   : Hybrid RRF (BM25 Lexical + MiniLM Dense)")
    print("Verification Engine: NLI Cross-Encoder (cross-encoder/nli-deberta-v3-small)")
    print(f"Top-K Evidence     : {top_k} candidate block(s) per requirement")
    print("-" * 80)

    formatter = formatter or ComplianceReportFormatter()

    try:
        pipeline = pipeline or CompliancePipeline()
        report = pipeline.run(
            reference_path=template_path,
            target_path=target_path,
            target_page=page,
            top_k=top_k,
        )
    except FileNotFoundError as err:
        print(f"\n[Error] File not found: {err}", file=sys.stderr)
        return 1
    except ScannedPDFError as err:
        print("\n" + "=" * 80, file=sys.stderr)
        print("  [ALERT] SCANNED / IMAGE-ONLY PDF DETECTED", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(f"File    : {err.filepath}", file=sys.stderr)
        print(f"Details : {err.char_count} character(s) across {err.page_count} page(s).", file=sys.stderr)
        print("Action  : OCR is required. Please provide a digital text-based PDF.", file=sys.stderr)
        print("=" * 80 + "\n", file=sys.stderr)
        return 2
    except ValueError as err:
        print(f"\n[Error] Invalid input or document: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        if debug:
            raise
        print(f"\n[Error] Compliance audit pipeline failed: {err}", file=sys.stderr)
        return 1

    # Export reports if paths specified
    saved_json_path = None
    saved_md_path = None

    if output_json:
        try:
            target_json = Path(output_json)
            if not target_json.is_absolute():
                target_json = Path.cwd() / target_json
            saved_json_path = str(formatter.export_json(report, target_json))
        except Exception as err:
            print(f"[Warning] Failed to export JSON report to '{output_json}': {err}", file=sys.stderr)

    if output_md:
        try:
            target_md = Path(output_md)
            if not target_md.is_absolute():
                target_md = Path.cwd() / target_md
            saved_md_path = str(formatter.export_markdown(report, target_md))
        except Exception as err:
            print(f"[Warning] Failed to export Markdown report to '{output_md}': {err}", file=sys.stderr)

    # Print terminal summary
    terminal_output = formatter.to_terminal(
        report,
        json_path=saved_json_path,
        md_path=saved_md_path,
    )
    print(terminal_output)

    return 0


def run_template_pipeline(
    template_path: str,
    target_path: str,
    page: int = None,
    covered_threshold: float = 0.70,
    partial_threshold: float = 0.50,
    min_evidence_threshold: float = 0.38,
    output_json: str = "compliance_report.json",
    output_md: str = "compliance_report.md"
):
    """
    Executes Phase 1.1 Reference-Template Content Compliance Checker:
    1. Extracts required sections and points from reference PDF (or JSON).
    2. Extracts paragraph-constrained candidate chunks & sliding windows from target PDF.
    3. Evaluates semantic coverage per required point using multi-perspective embeddings and all-MiniLM-L6-v2.
    4. Classifies each point as COVERED, PARTIAL, or MISSING with explicit absence & evidence safeguards.
    5. Emits readable terminal, JSON, and Markdown reports.
    """
    print("=" * 76)
    print("      AI-SRS COMPLIANCE ANALYZER — TEMPLATE CONTENT CHECKER (PHASE 1.1)")
    print("=" * 76)
    print(f"Reference Template : {template_path}")
    print(f"Target Document    : {target_path}")
    if page is not None:
        print(f"Target Scope       : Single Page (Page {page})")
    else:
        print("Target Scope       : Entire Document")
    print(f"Model Engine       : sentence-transformers (all-MiniLM-L6-v2)")
    print(f"Covered Threshold  : >= {covered_threshold:.2f} (Cosine Similarity)")
    print(f"Partial Threshold  : >= {partial_threshold:.2f} (Cosine Similarity)")
    print(f"Min Evidence Thresh: >= {min_evidence_threshold:.2f} (Cosine Similarity)")
    print("-" * 76)

    if not os.path.exists(target_path):
        print(f"Error: Target file '{target_path}' not found.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(template_path):
        print(f"Error: Template file '{template_path}' not found.", file=sys.stderr)
        sys.exit(1)

    pdf_extractor = PDFExtractor()

    # Stage 1: Ingest Reference Template (PDF or JSON standard)
    template: ReferenceTemplate
    try:
        if template_path.lower().endswith(".pdf"):
            template = pdf_extractor.extract_template(template_path)
        else:
            std = ReferenceStandard(template_path)
            points = []
            for s in std.sections:
                text_val = f"{s.name}: {s.description}" if s.description else s.name
                points.append(
                    TemplatePoint(
                        point_id=s.section_id,
                        section_name=s.name,
                        text=text_val
                    )
                )
            for r in std.rules:
                text_val = f"{r.name}: {r.description}" if r.description else r.name
                points.append(
                    TemplatePoint(
                        point_id=r.rule_id,
                        section_name="Quality Rules",
                        text=text_val
                    )
                )
            template = ReferenceTemplate(
                title=std.standard_name,
                points=points,
                sections=[s.name for s in std.sections]
            )
    except ScannedPDFError as err:
        print("\n" + "=" * 76, file=sys.stderr)
        print("  [ALERT] SCANNED / IMAGE-ONLY REFERENCE TEMPLATE DETECTED", file=sys.stderr)
        print("=" * 76, file=sys.stderr)
        print(f"File    : {err.filepath}", file=sys.stderr)
        print(f"Details : {err.char_count} character(s) across {err.page_count} page(s).", file=sys.stderr)
        print("Action  : OCR is required. Please provide a digital text-based template PDF.", file=sys.stderr)
        print("=" * 76 + "\n", file=sys.stderr)
        sys.exit(2)
    except Exception as err:
        print(f"Error loading reference template '{template_path}': {err}", file=sys.stderr)
        sys.exit(1)

    print(f"Template Ingestion Completed : '{template.title}' with {len(template.points)} required point(s).")

    # Stage 2: Extract Target Candidate Chunks
    total_pages = 1
    target_chunks = []
    try:
        if target_path.lower().endswith(".pdf"):
            full_res = pdf_extractor.extract_text(target_path, raise_if_scanned=False)
            total_pages = full_res.page_count
            target_chunks = pdf_extractor.extract_target_chunks(target_path, page_number=page)
        else:
            with open(target_path, "r", encoding="utf-8") as f:
                content = f.read()
            target_chunks = [l.strip() for l in content.splitlines() if len(l.strip()) >= 15]
    except ScannedPDFError as err:
        print("\n" + "=" * 76, file=sys.stderr)
        print("  [ALERT] SCANNED / IMAGE-ONLY TARGET PDF DETECTED", file=sys.stderr)
        print("=" * 76, file=sys.stderr)
        print(f"File    : {err.filepath}", file=sys.stderr)
        print(f"Details : {err.char_count} character(s) across {err.page_count} page(s).", file=sys.stderr)
        print("Action  : OCR is required for scanned documents. Please provide digital text.", file=sys.stderr)
        print("=" * 76 + "\n", file=sys.stderr)
        sys.exit(2)
    except ValueError as err:
        print(f"Error extracting target: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"Error reading target file '{target_path}': {err}", file=sys.stderr)
        sys.exit(1)

    print(f"Target Ingestion Completed   : {len(target_chunks)} candidate statement chunk(s) extracted.")
    print("-" * 76)

    # Stage 3: Semantic Comparison & Content Compliance Checking
    checker = TemplateContentChecker(
        covered_threshold=covered_threshold,
        partial_threshold=partial_threshold,
        min_evidence_threshold=min_evidence_threshold
    )
    report = checker.check_compliance(
        template=template,
        target_chunks=target_chunks,
        target_name=os.path.basename(target_path),
        target_page=page,
        total_pages=total_pages
    )

    # Stage 4: Generate & Display Reports
    terminal_output = ReportGenerator.format_template_terminal(report)
    print(terminal_output)

    # Save structured JSON report
    json_path = os.path.join(BASE_DIR, output_json)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ReportGenerator.to_template_json(report), f, indent=2)

    # Save human-readable Markdown report
    md_path = os.path.join(BASE_DIR, output_md)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(ReportGenerator.format_template_markdown(report))

    print(f"\n[Artifact] Structured JSON report saved to : {json_path}")
    print(f"[Artifact] Markdown report saved to        : {md_path}")
    print("=" * 76)


def run_legacy_pipeline(
    srs_path: str,
    standard_path: str,
    output_json: str = "compliance_report.json",
    output_md: str = "compliance_report.md",
    threshold: float = 0.55
):
    """Legacy v0.1-v0.4 pipeline checking deterministic R1-R4 rules and section alignment."""
    print("=" * 72)
    print("    AI-SRS COMPLIANCE ANALYZER — LEGACY QUALITY RULES (v0.1-v0.4)")
    print("=" * 72)
    print(f"Loading Reference Standard : {standard_path}")
    print(f"Loading SRS Document       : {srs_path}")
    print("-" * 72)

    if not os.path.exists(srs_path) or not os.path.exists(standard_path):
        print("Error: Input files not found.", file=sys.stderr)
        sys.exit(1)

    try:
        standard = ReferenceStandard(standard_path)
    except ScannedPDFError as err:
        print(f"Scanned PDF Error: {err}", file=sys.stderr)
        sys.exit(2)

    try:
        extractor = RequirementExtractor()
        requirements = extractor.extract_from_file(srs_path)
        sections = extractor.extract_sections_from_file(srs_path)
    except ScannedPDFError as err:
        print(f"Scanned PDF Error: {err}", file=sys.stderr)
        sys.exit(2)

    analyzer = ComplianceAnalyzer(standard, similarity_threshold=threshold)
    report = analyzer.audit_document(requirements, sections=sections)

    terminal_output = ReportGenerator.format_terminal(report)
    print(terminal_output)

    json_path = os.path.join(BASE_DIR, output_json)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ReportGenerator.to_json(report), f, indent=2)

    md_path = os.path.join(BASE_DIR, output_md)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(ReportGenerator.format_markdown(report))

    print(f"[Artifact] Saved to: {json_path} and {md_path}")


def run_demonstration(mode: str = "pdf"):
    """
    Runs interactive demonstrations of AI compliance and PDF capabilities:
    - 'pdf': Version 0.4 PDF input support, PDF-to-PDF analysis, and scanned detection
    - 'rules': Version 0.3 semantic matching of rules against requirements
    - 'sections': Version 0.2 section alignment demonstration
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


def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Verify a target document (PDF) against a reference template PDF using V2 Hybrid Retrieval and NLI Evidence Verification."
    )
    default_template = os.path.join(BASE_DIR, "data", "reference_standard.pdf")
    default_target = os.path.join(BASE_DIR, "data", "sample_srs.pdf")

    parser.add_argument(
        "--template", "--standard", "-r",
        dest="template",
        default=default_template,
        help=f"Path to reference template PDF or standard file (default: {default_template})"
    )
    parser.add_argument(
        "--target", "--srs", "-s",
        dest="target",
        default=default_target,
        help=f"Path to target document (.pdf, .txt, .md) (default: {default_target})"
    )
    parser.add_argument(
        "--page", "-p",
        type=int,
        default=None,
        help="Specific page number (1-based index) of target PDF to check. Default: all pages."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of candidate evidence blocks to retrieve per requirement in V2 pipeline (default: 5)"
    )
    parser.add_argument(
        "--output-json", "-j",
        default="compliance_report.json",
        help="Filename or path for exported JSON audit report (default: compliance_report.json)"
    )
    parser.add_argument(
        "--output-md", "-m",
        default="compliance_report.md",
        help="Filename or path for exported Markdown audit report (default: compliance_report.md)"
    )
    parser.add_argument(
        "--legacy-rules",
        action="store_true",
        help="Run legacy v0.1-v0.4 quality rules audit (R1-R4) instead of V2 pipeline"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.55,
        help="Cosine similarity threshold for legacy section matching (default: 0.55)"
    )
    parser.add_argument(
        "--covered-threshold",
        type=float,
        default=0.70,
        help="Cosine similarity threshold for legacy COVERED classification (default: 0.70)"
    )
    parser.add_argument(
        "--partial-threshold",
        type=float,
        default=0.50,
        help="Cosine similarity threshold for legacy PARTIAL classification (default: 0.50)"
    )
    parser.add_argument(
        "--min-evidence-threshold",
        type=float,
        default=0.38,
        help="Minimum cosine similarity threshold for legacy evidence selection (default: 0.38)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Display full traceback on unhandled pipeline exceptions"
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
    return parser


def main(args_list: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(args_list)

    if args.demo_all:
        run_demonstration(mode="all")
        return 0
    elif args.demo_sections:
        run_demonstration(mode="sections")
        return 0
    elif args.demo_rules:
        run_demonstration(mode="rules")
        return 0
    elif args.demo:
        run_demonstration(mode="pdf")
        return 0

    if args.legacy_rules:
        run_legacy_pipeline(
            srs_path=args.target,
            standard_path=args.template,
            output_json=args.output_json,
            output_md=args.output_md,
            threshold=args.threshold,
        )
        return 0
    else:
        exit_code = run_v2_pipeline(
            template_path=args.template,
            target_path=args.target,
            page=args.page,
            top_k=args.top_k,
            output_json=args.output_json,
            output_md=args.output_md,
            debug=args.debug,
        )
        return exit_code


if __name__ == "__main__":
    sys.exit(main())
