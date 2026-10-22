#!/usr/bin/env python3
"""
Demonstration: Semantic Rule Matching using sentence-transformers (all-MiniLM-L6-v2)
====================================================================================
Version 0.3 AI Capability Demonstration for SRS Compliance Analyzer.

This script demonstrates:
1. Controlled demonstration across 4 distinct categories:
   - Compliant requirement statements
   - Non-compliant requirement statements
   - Paraphrased rule statements
   - Irrelevant / out-of-domain text
2. Semantic similarity scoring between individual requirements and reference rules (R1 - R4).
3. Candidate score distribution across all rules for each test statement.
4. Live rule alignment on the benchmark sample requirements (data/sample_srs.txt).
"""

import os
import sys
import sentence_transformers
# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.analyzer import (
    ReferenceStandard,
    RequirementExtractor,
    SemanticRuleMatcher,
)


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def run_demo():
    print_header("AI-SRS COMPLIANCE ANALYZER — VERSION 0.3 DEMONSTRATION")
    print("Model  : sentence-transformers (all-MiniLM-L6-v2)")
    print("Vector : 384-dimensional dense semantic embeddings")
    print("Metric : Cosine Similarity ∈ [-1.0, 1.0]")
    print("Goal   : Semantic matching between individual reference rules and SRS requirements")

    matcher = SemanticRuleMatcher(model_name="all-MiniLM-L6-v2")
    standard_path = os.path.join(BASE_DIR, "data", "reference_standard.json")
    standard = ReferenceStandard(standard_path)

    print("\nLoaded Reference Rules:")
    for r in standard.rules:
        print(f"  [{r.rule_id}] {r.name}: \"{r.description}\"")

    # --------------------------------------------------------------------------
    # CONTROLLED TEST CASES
    # --------------------------------------------------------------------------
    test_cases = [
        # 1. Compliant Examples
        (
            "COMPLIANT",
            "The navigation unit shall record pilot inputs and compute aircraft trajectory.",
            "Valid mandatory action using binding 'shall' modal."
        ),
        (
            "COMPLIANT",
            "The flight display unit shall refresh the barometric altitude reading at a frequency of at least 20 Hz with a maximum end-to-end latency of 50 ms.",
            "Valid quantitative metric with explicit latency & frequency bounds."
        ),
        # 2. Non-Compliant Examples
        (
            "NON-COMPLIANT",
            "The system should update the ground speed display quickly whenever wind gust changes occur.",
            "Non-binding modal 'should' (R2) and vague term 'quickly' (R4)."
        ),
        (
            "NON-COMPLIANT",
            "The user interface should be easy, intuitive, and display information quickly.",
            "Subjective qualitative terms 'easy' and 'quickly' violating R4."
        ),
        (
            "NON-COMPLIANT",
            "The software module shall initialize memory buffers upon cold boot.",
            "Missing unique alphanumeric tracking identifier tag (violating R1)."
        ),
        # 3. Paraphrased Examples (Semantic Alignment to Rules)
        (
            "PARAPHRASED",
            "Every specification item requires a unique alphanumeric identifier tag.",
            "Paraphrase of Rule R1 (Unique Requirement ID)."
        ),
        (
            "PARAPHRASED",
            "The software is strictly mandated to execute deterministic operational actions.",
            "Paraphrase of Rule R2 (Clear Behaviour)."
        ),
        (
            "PARAPHRASED",
            "All response latencies and timing bounds must be quantitatively measurable.",
            "Paraphrase of Rule R3 (Measurable Criteria)."
        ),
        (
            "PARAPHRASED",
            "Ambiguous subjective terms like quickly, fast, or user-friendly must be avoided.",
            "Paraphrase of Rule R4 (Avoid Vague Language)."
        ),
        # 4. Irrelevant / Out-of-Domain Examples
        (
            "IRRELEVANT",
            "The corporate cafeteria serves fresh sandwiches and hot soup between noon and 1:30 PM.",
            "Food/catering service description unrelated to software requirements."
        ),
        (
            "IRRELEVANT",
            "Employee parking permits must be displayed on the front windshield in garage B.",
            "Facility parking policy completely out-of-domain."
        ),
    ]

    print_header("CONTROLLED DEMONSTRATION: REQUIREMENT ↔ RULE SEMANTIC MATCHING")
    print(f"{'Category':<15} | {'Requirement Text':<50} | {'Best Rule Match':<24} | {'Score':<7}")
    print("-" * 105)

    for category, text, note in test_cases:
        res = matcher.match_requirement_to_rules(text, standard.rules)
        short_text = text[:48] + ".." if len(text) > 50 else text
        match_str = f"{res.best_rule_id} ({res.best_rule_name})"
        print(f"{category:<15} | {short_text:<50} | {match_str:<24} | {res.similarity_score:<7.4f}")

    print("\nDetailed Score Distributions Across All Rules (R1 - R4):")
    print("-" * 105)
    for category, text, note in test_cases:
        res = matcher.match_requirement_to_rules(text, standard.rules)
        scores_str = " | ".join([f"{rid}: {score:.4f}" for rid, score in res.candidate_scores.items()])
        print(f"[{category}] \"{text[:60]}...\"")
        print(f"  Note : {note}")
        print(f"  Best : {res.best_rule_id} ({res.best_rule_name}) [Score: {res.similarity_score:.4f}]")
        print(f"  All  : {scores_str}")
        print()

    # --------------------------------------------------------------------------
    # LIVE AUDIT DEMONSTRATION ON BENCHMARK REQUIREMENTS
    # --------------------------------------------------------------------------
    print_header("LIVE AUDIT: SEMANTIC RULE MATCHING ON 'data/sample_srs.txt'")
    sample_path = os.path.join(BASE_DIR, "data", "sample_srs.txt")
    extractor = RequirementExtractor()
    reqs = extractor.extract_from_file(sample_path)

    for req in reqs:
        res = matcher.match_requirement_to_rules(req.statement, standard.rules)
        scores_str = ", ".join([f"{rid}={s:.3f}" for rid, s in res.candidate_scores.items()])
        print(f"[{req.req_id}] \"{req.statement}\"")
        print(f"  -> Best Matching Rule : {res.best_rule_id} ({res.best_rule_name})")
        print(f"  -> Similarity Score    : {res.similarity_score:.4f}")
        print(f"  -> Rule Score Spectrum : {scores_str}")
        print()

    print("=" * 80)
    print("VERSION 0.3 DEMONSTRATION COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
