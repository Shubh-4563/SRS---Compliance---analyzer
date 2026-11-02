"""
Automated Evaluation Script for V2 Hybrid Candidate Retrieval (Phase 2.2).
Computes strict exact ground-truth block matching for Top-3 and Top-5 recall
across benchmark test packs Pair A and Pair B.

Usage:
    python3 scripts/evaluate_retrieval.py
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.pdf_extractor import PDFExtractor
from src.parsers.pdf_parser import TargetPDFParser
from src.parsers.requirement_decomposer import RequirementDecomposer
from src.retrieval.hybrid_retriever import HybridRetriever


# Ground-truth evidence mapping for Benchmark Pair A
# Target A Blocks:
# p1_b1: Title (Measuring the Cooling of Warm Water)
# p1_b3: 1. Aim
# p1_b5: 2. Materials
# p1_b7: 3. Method
# p1_b9: 4. Observations (table row)
# p1_b11: 5. Discussion
# p1_b13: 6. Conclusion
# p1_b14: Safety disclaimer (explicit omission)
GROUND_TRUTH_A: Dict[str, Set[str]] = {
    "A1.1": {"p1_b3"},   # Purpose/aim
    "A1.2": {"p1_b3"},   # What is demonstrated/measured
    "A2.1": {"p1_b5"},   # Equipment/instruments
    "A2.2": {"p1_b5"},   # Materials/reagents
    "A3.1": {"p1_b7"},   # Procedure/ordered steps
    "A4.1": {"p1_b9"},   # Measured numerical values
    "A5.1": {"p1_b11"},  # Meaning of observations / trends
    "A5.2": {"p1_b11"},  # Comparison with theory/room temp
    "A6.1": {"p1_b13"},  # Outcome and whether aim achieved
    "A7.1": {"p1_b14"},  # Safety disclaimer / omission
}

# Ground-truth evidence mapping for Benchmark Pair B
# Target B Blocks:
# p1_b1: Title (Equipment Monitoring Dashboard)
# p1_b3: 1. Overview
# p1_b5: 2. Main Capabilities
# p1_b7: 3. Connections (Users & gateway)
# p1_b8: Connections disclaimer (explicit omission of protocols)
# p1_b10: 4. Fault Response (Sensor failure warning)
# p1_b11: Fault Response disclaimer (explicit omission of recovery)
# p1_b13: 5. Deployment (Quick response)
# p1_b14: Deployment disclaimer (explicit omission of numeric targets)
# p1_b16: 6. Validation (Testing operator steps)
GROUND_TRUTH_B: Dict[str, Set[str]] = {
    "B1.1": {"p1_b3"},   # Purpose / intended use
    "B1.2": {"p1_b3"},   # Problem it solves
    "B2.1": {"p1_b5", "p1_b3"},  # Included components / capabilities
    "B2.2": {"p1_b3"},   # Outside scope
    "B3.1": {"p1_b5"},   # Core functions / actions
    "B4.1": {"p1_b7"},   # System users / browser
    "B4.2": {"p1_b7"},   # External hardware / gateway
    "B4.3": {"p1_b8"},   # Interface protocol disclaimer
    "B5.1": {"p1_b10"},  # Hazards/faults detection & warning
    "B5.2": {"p1_b11"},  # Recovery steps disclaimer
    "B6.1": {"p1_b14"},  # Acceptance criteria disclaimer
    "B7.1": {"p1_b16"},  # Testing / verification
}


def evaluate_pair(
    pair_name: str,
    template_path: Path,
    target_path: Path,
    ground_truth: Dict[str, Set[str]],
    retriever: HybridRetriever,
    extractor: PDFExtractor,
    parser: TargetPDFParser,
    decomposer: RequirementDecomposer,
) -> Tuple[int, int, int]:
    print(f"\n==================================================")
    print(f"EVALUATING PAIR {pair_name}")
    print(f"Template: {template_path.name}")
    print(f"Target:   {target_path.name}")
    print(f"==================================================")

    tmpl = extractor.extract_template(str(template_path))
    dec = decomposer.decompose_template(tmpl)
    doc = parser.parse_document(str(target_path))
    retriever.index_document(doc, include_headings=False)

    total_subreqs = 0
    top3_hits = 0
    top5_hits = 0

    for pt in dec.points:
        for sr in pt.sub_requirements:
            total_subreqs += 1
            t3 = retriever.retrieve_candidates_for_subrequirement(sr, top_k=3)
            t5 = retriever.retrieve_candidates_for_subrequirement(sr, top_k=5)

            t3_ids = [c.block_id for c in t3]
            t5_ids = [c.block_id for c in t5]

            expected = ground_truth.get(sr.sub_id, set())

            hit_3 = any(bid in expected for bid in t3_ids)
            hit_5 = any(bid in expected for bid in t5_ids)

            if hit_3:
                top3_hits += 1
            if hit_5:
                top5_hits += 1

            status_3 = "PASS (Hit)" if hit_3 else "FAIL (Miss)"
            status_5 = "PASS (Hit)" if hit_5 else "FAIL (Miss)"

            print(f"[{sr.sub_id}] {sr.topic_label}")
            print(f"  Claim:    {sr.claim_text}")
            print(f"  Expected: {sorted(list(expected))}")
            print(f"  Top-3:    {t3_ids} -> {status_3}")
            print(f"  Top-5:    {t5_ids} -> {status_5}")

    top3_pct = (top3_hits / total_subreqs) * 100 if total_subreqs else 0
    top5_pct = (top5_hits / total_subreqs) * 100 if total_subreqs else 0
    print(f"--- Pair {pair_name} Summary ---")
    print(f"Total Sub-requirements: {total_subreqs}")
    print(f"Top-3 Recall (Strict):  {top3_hits}/{total_subreqs} ({top3_pct:.1f}%)")
    print(f"Top-5 Recall (Strict):  {top5_hits}/{total_subreqs} ({top5_pct:.1f}%)")

    return total_subreqs, top3_hits, top5_hits


def main():
    extractor = PDFExtractor()
    parser = TargetPDFParser()
    decomposer = RequirementDecomposer()
    retriever = HybridRetriever(rrf_k=60)

    a_tmpl = BASE_DIR / "data" / "test_packs" / "A_reference_lab_report_template.pdf"
    a_tgt = BASE_DIR / "data" / "test_packs" / "A_target_lab_report.pdf"

    b_tmpl = BASE_DIR / "data" / "test_packs" / "B_reference_technical_template.pdf"
    b_tgt = BASE_DIR / "data" / "test_packs" / "B_target_technical_document.pdf"

    tot_a, t3_a, t5_a = evaluate_pair("A", a_tmpl, a_tgt, GROUND_TRUTH_A, retriever, extractor, parser, decomposer)
    tot_b, t3_b, t5_b = evaluate_pair("B", b_tmpl, b_tgt, GROUND_TRUTH_B, retriever, extractor, parser, decomposer)

    total = tot_a + tot_b
    total_t3 = t3_a + t3_b
    total_t5 = t5_a + t5_b

    print("\n==================================================")
    print("OVERALL REPRODUCIBILITY SUMMARY")
    print("==================================================")
    print(f"Pair A Strict Top-3 Recall: {t3_a}/{tot_a} ({(t3_a/tot_a)*100:.1f}%) | Top-5: {t5_a}/{tot_a} ({(t5_a/tot_a)*100:.1f}%)")
    print(f"Pair B Strict Top-3 Recall: {t3_b}/{tot_b} ({(t3_b/tot_b)*100:.1f}%) | Top-5: {t5_b}/{tot_b} ({(t5_b/tot_b)*100:.1f}%)")
    print(f"Combined Strict Top-3 Recall: {total_t3}/{total} ({(total_t3/total)*100:.1f}%)")
    print(f"Combined Strict Top-5 Recall: {total_t5}/{total} ({(total_t5/total)*100:.1f}%)")


if __name__ == "__main__":
    main()

