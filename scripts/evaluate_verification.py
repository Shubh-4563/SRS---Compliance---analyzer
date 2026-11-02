"""
Automated Evaluation Script for V2 NLI-based Evidence Verification (Phase 2.3).
Evaluates Top-5 retrieved candidates for all atomic sub-requirements in Pair A and Pair B.

Usage:
    python3 scripts/evaluate_verification.py
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
from src.models.verification import VerificationStatus, SubRequirementVerificationResult


def evaluate_verification_pair(
    pair_name: str,
    template_path: Path,
    target_path: Path,
    extractor: PDFExtractor,
    parser: TargetPDFParser,
    decomposer: RequirementDecomposer,
    retriever: HybridRetriever,
    verifier: EvidenceVerifier
) -> List[SubRequirementVerificationResult]:
    print(f"\n==================================================")
    print(f"VERIFICATION BENCHMARK: PAIR {pair_name}")
    print(f"Template: {template_path.name}")
    print(f"Target:   {target_path.name}")
    print(f"==================================================")

    tmpl = extractor.extract_template(str(template_path))
    dec = decomposer.decompose_template(tmpl)
    doc = parser.parse_document(str(target_path))
    retriever.index_document(doc, include_headings=False)

    results: List[SubRequirementVerificationResult] = []

    for pt in dec.points:
        print(f"\n--- Parent Point {pt.point_id}: {pt.topic_label} ---")
        for sr in pt.sub_requirements:
            # Retrieve Top-5 candidates as approved
            candidates = retriever.retrieve_candidates_for_subrequirement(sr, top_k=5)

            # Verify using NLI EvidenceVerifier
            res = verifier.verify_subrequirement(sr, candidates)
            results.append(res)

            best_block = res.best_evidence.candidate.block_id if res.best_evidence else "None"
            best_lbl = res.best_evidence.predicted_label if res.best_evidence else "None"

            print(f"[{sr.sub_id}] ({sr.expected_evidence_type.value})")
            print(f"  Claim:      {sr.claim_text}")
            print(f"  Status:     {res.status.value} (Confidence: {res.confidence:.4f})")
            print(f"  Best Block: {best_block} ({best_lbl})")
            if res.best_evidence:
                print(f"  NLI Probs:  contra={res.best_evidence.contradiction_prob:.4f}, ent={res.best_evidence.entailment_prob:.4f}, neu={res.best_evidence.neutral_prob:.4f}")
                print(f"  Evidence:   {res.best_evidence.premise_text[:80]}...")
            print(f"  Rationale:  {res.rationale}")

    return results


def summarize_results(name: str, results: List[SubRequirementVerificationResult]):
    total = len(results)
    fulfilled = sum(1 for r in results if r.status == VerificationStatus.FULFILLED)
    omitted = sum(1 for r in results if r.status == VerificationStatus.EXPLICIT_OMISSION)
    contradicted = sum(1 for r in results if r.status == VerificationStatus.CONTRADICTED)
    missing = sum(1 for r in results if r.status == VerificationStatus.MISSING)
    review = sum(1 for r in results if r.status == VerificationStatus.NEEDS_REVIEW)

    print(f"\n--- PAIR {name} VERIFICATION SUMMARY ---")
    print(f"Total Sub-requirements: {total}")
    print(f"  FULFILLED:         {fulfilled} ({(fulfilled/total)*100:.1f}%)")
    print(f"  EXPLICIT_OMISSION: {omitted} ({(omitted/total)*100:.1f}%)")
    print(f"  CONTRADICTED:      {contradicted} ({(contradicted/total)*100:.1f}%)")
    print(f"  MISSING:           {missing} ({(missing/total)*100:.1f}%)")
    print(f"  NEEDS_REVIEW:      {review} ({(review/total)*100:.1f}%)")


def main():
    extractor = PDFExtractor()
    parser = TargetPDFParser()
    decomposer = RequirementDecomposer()
    retriever = HybridRetriever(rrf_k=60)
    nli = NLIVerifier(local_files_only=True)
    verifier = EvidenceVerifier(nli)

    a_tmpl = BASE_DIR / "data" / "test_packs" / "A_reference_lab_report_template.pdf"
    a_tgt = BASE_DIR / "data" / "test_packs" / "A_target_lab_report.pdf"

    b_tmpl = BASE_DIR / "data" / "test_packs" / "B_reference_technical_template.pdf"
    b_tgt = BASE_DIR / "data" / "test_packs" / "B_target_technical_document.pdf"

    results_a = evaluate_verification_pair("A", a_tmpl, a_tgt, extractor, parser, decomposer, retriever, verifier)
    summarize_results("A", results_a)

    results_b = evaluate_verification_pair("B", b_tmpl, b_tgt, extractor, parser, decomposer, retriever, verifier)
    summarize_results("B", results_b)


if __name__ == "__main__":
    main()

