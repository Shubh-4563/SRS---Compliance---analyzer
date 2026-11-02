"""
Automated Evaluation Script for V2 Coverage Aggregation and Compliance Scoring (Phase 2.4).
Runs full end-to-end verification and compliance aggregation on Pair A and Pair B benchmarks.

Usage:
    python3 scripts/evaluate_compliance.py
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Ensure offline mode for test/benchmark reproducibility
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.pdf_extractor import PDFExtractor
from src.parsers.pdf_parser import TargetPDFParser
from src.parsers.requirement_decomposer import RequirementDecomposer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.verification.nli_verifier import NLIVerifier
from src.verification.evidence_verifier import EvidenceVerifier
from src.verification.compliance_aggregator import ComplianceAggregator
from src.models.verification import VerificationStatus, SubRequirementVerificationResult
from src.models.compliance import DocumentComplianceReport, ParentPointCompliance


def evaluate_compliance_pair(
    pair_name: str,
    template_path: Path,
    target_path: Path,
    extractor: PDFExtractor,
    parser: TargetPDFParser,
    decomposer: RequirementDecomposer,
    retriever: HybridRetriever,
    verifier: EvidenceVerifier,
    aggregator: ComplianceAggregator
) -> DocumentComplianceReport:
    print(f"\n==================================================")
    print(f"COMPLIANCE EVALUATION: PAIR {pair_name}")
    print(f"Template: {template_path.name}")
    print(f"Target:   {target_path.name}")
    print(f"==================================================")

    # 1. Parse and decompose template
    tmpl = extractor.extract_template(str(template_path))
    dec_template = decomposer.decompose_template(tmpl)

    # 2. Parse and index target document
    target_doc = parser.parse_document(str(target_path))
    retriever.index_document(target_doc, include_headings=False)

    # 3. Verify all sub-requirements using Top-5 candidates
    verification_results: List[SubRequirementVerificationResult] = []
    for pt in dec_template.points:
        for sr in pt.sub_requirements:
            candidates = retriever.retrieve_candidates_for_subrequirement(sr, top_k=5)
            res = verifier.verify_subrequirement(sr, candidates)
            verification_results.append(res)

    # 4. Aggregate into DocumentComplianceReport
    report = aggregator.aggregate(
        template=dec_template,
        verification_results=verification_results,
        target_document_path=str(target_path),
        target_document_name=target_path.name,
        total_pages=target_doc.total_pages
    )

    # 5. Print formatted results (NO TABLES)
    print(f"\nDocument Compliance Verdict: {report.verdict.value}")
    print(f"Mandatory Coverage: {f'{report.mandatory_coverage_pct:.1f}% ({report.mandatory_fulfilled}/{report.total_mandatory})' if report.mandatory_coverage_pct is not None else 'N/A'}")
    print(f"Optional Coverage:  {f'{report.optional_coverage_pct:.1f}% ({report.optional_fulfilled}/{report.total_optional})' if report.optional_coverage_pct is not None else 'N/A'}")
    print(f"Total Parent Points: {report.total_parent_points}")
    print(f"Total Sub-requirements: {report.total_sub_requirements}")

    print("\n--- Parent Point Compliance Breakdown ---")
    for p in report.parent_points:
        print(f"\nParent Point {p.point_id}: {p.topic_label or p.section_name}")
        print(f"  Status: {p.status.value}")
        print(f"  Mandatory: {p.mandatory_fulfilled}/{p.mandatory_total} fulfilled")
        if p.optional_total > 0:
            print(f"  Optional:  {p.optional_fulfilled}/{p.optional_total} fulfilled")
        print(f"  Findings:")
        for f in p.findings:
            print(f"    - {f}")

    print("\n--- Atomic Verification Status Counts ---")
    print(f"  FULFILLED:         {report.fulfilled_count}")
    print(f"  EXPLICIT_OMISSION: {report.explicit_omission_count}")
    print(f"  CONTRADICTED:      {report.contradicted_count}")
    print(f"  MISSING:           {report.missing_count}")
    print(f"  NEEDS_REVIEW:      {report.needs_review_count}")

    print("\n--- Items Requiring Human Review ---")
    if report.items_requiring_review:
        for item in report.items_requiring_review:
            print(f"  * [{item.sub_id}] ({item.parent_point_id}): {item.rationale}")
    else:
        print("  None. All items resolved deterministically.")

    return report


def main():
    extractor = PDFExtractor()
    parser = TargetPDFParser()
    decomposer = RequirementDecomposer()
    retriever = HybridRetriever(rrf_k=60)
    nli = NLIVerifier(local_files_only=True)
    verifier = EvidenceVerifier(nli)
    aggregator = ComplianceAggregator()

    pack_dir = BASE_DIR / "data" / "test_packs"

    # Pair A
    tmpl_a = pack_dir / "A_reference_lab_report_template.pdf"
    target_a = pack_dir / "A_target_lab_report.pdf"
    report_a = evaluate_compliance_pair(
        "A", tmpl_a, target_a,
        extractor, parser, decomposer, retriever, verifier, aggregator
    )

    # Pair B
    tmpl_b = pack_dir / "B_reference_technical_template.pdf"
    target_b = pack_dir / "B_target_technical_document.pdf"
    report_b = evaluate_compliance_pair(
        "B", tmpl_b, target_b,
        extractor, parser, decomposer, retriever, verifier, aggregator
    )


if __name__ == "__main__":
    main()

