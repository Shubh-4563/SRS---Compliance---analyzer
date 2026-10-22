#!/usr/bin/env python3
"""
Run script for the SRS Compliance and Quality Analysis Proof of Concept (POC).
Demonstrates end-to-end execution on sample avionics-style requirements.
"""

import os
import json
from srs_compliance_analyzer import SRSComplianceAnalyzer, format_markdown_report


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ref_file = os.path.join(base_dir, "data", "reference_standard.json")
    doc_file = os.path.join(base_dir, "data", "sample_srs.md")

    print("=" * 70)
    print("AI-POWERED SRS COMPLIANCE & QUALITY ANALYZER - PROTOTYPE RUN")
    print("=" * 70)
    print(f"Loading Reference Standard: {ref_file}")
    print(f"Loading Document Under Review: {doc_file}")

    with open(doc_file, "r", encoding="utf-8") as f:
        doc_text = f.read()

    # Initialize analyzer with arbitrary reference standard
    analyzer = SRSComplianceAnalyzer(ref_file)

    # Execute full audit
    audit_results = analyzer.run_full_audit(doc_text)

    # Generate Markdown Report
    report_md = format_markdown_report(audit_results)

    # Save outputs
    json_output_path = os.path.join(base_dir, "compliance_report.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    md_output_path = os.path.join(base_dir, "compliance_report.md")
    with open(md_output_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n" + report_md)
    print("=" * 70)
    print(f"Audit Complete! Saved structured JSON to: {json_output_path}")
    print(f"Saved Markdown Report to: {md_output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
