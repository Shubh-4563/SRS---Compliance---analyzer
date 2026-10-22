#!/usr/bin/env python3
"""
Demonstration: Semantic Section Matching using sentence-transformers (all-MiniLM-L6-v2)
=======================================================================================
Version 0.2 AI Capability Demonstration for SRS Compliance Analyzer.

This script demonstrates:
1. Pairwise semantic similarity scoring between standard sections and diverse phrasings.
2. Robustness to synonyms and paraphrasing where exact-match / keyword search fails.
3. Cross-similarity matrix showing semantic selectivity.
4. Negative / out-of-domain detection (rejecting irrelevant headings).
5. End-to-end section alignment on the sample avionics SRS document.
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
    SemanticSectionMatcher,
)


def print_header(title: str):
    print("\n" + "=" * 76)
    print(f"  {title}")
    print("=" * 76)


def run_demo():
    print_header("AI-SRS COMPLIANCE ANALYZER — VERSION 0.2 DEMONSTRATION")
    print("Model  : sentence-transformers (all-MiniLM-L6-v2)")
    print("Vector : 384-dimensional dense semantic embeddings")
    print("Metric : Cosine Similarity ∈ [-1.0, 1.0]")
    print("Goal   : Semantic alignment of reference-standard sections with SRS sections")

    matcher = SemanticSectionMatcher(model_name="all-MiniLM-L6-v2", threshold=0.55)

    # --------------------------------------------------------------------------
    # DEMO 1: Paraphrased Section Headings (Synonym Robustness)
    # --------------------------------------------------------------------------
    print_header("DEMO 1: SEMANTIC SIMILARITY ON PARAPHRASED SECTION HEADINGS")
    print("Showing how all-MiniLM-L6-v2 matches sections despite different vocabulary:\n")

    test_pairs = [
        (
            "Scope and System Overview",
            "System Scope and Architectural Overview",
            "Scope vs. system boundaries & architecture"
        ),
        (
            "Functional Requirements",
            "System Functional and Operational Specifications",
            "Functional requirements vs. functional/operational specifications"
        ),
        (
            "Interface Requirements",
            "External Interface and Communications Specifications",
            "Interface requirements vs. communications & external interface"
        ),
        (
            "Safety and Failure Handling",
            "Safety Hazard and Fault Management Procedures",
            "Safety handling vs. fault & hazard management"
        ),
    ]

    print(f"{'Standard Section Name':<28} | {'Candidate Section Heading':<54} | {'Score':<7} | {'Verdict'}")
    print("-" * 102)
    for ref_name, candidate, desc in test_pairs:
        score = matcher.compute_similarity(ref_name, candidate)
        verdict = "MATCHED (✓)" if score >= 0.55 else "REJECTED (✗)"
        print(f"{ref_name:<28} | {candidate:<54} | {score:<7.4f} | {verdict}")

    # --------------------------------------------------------------------------
    # DEMO 2: Cross-Similarity Matrix (Selectivity & Discrimination)
    # --------------------------------------------------------------------------
    print_header("DEMO 2: CROSS-SIMILARITY MATRIX (DISCRIMINATION TEST)")
    print("Evaluating all reference sections against all candidate section headings:\n")

    ref_sections = [
        "Scope and System Overview",
        "Functional Requirements",
        "Interface Requirements",
        "Safety and Failure Handling"
    ]

    candidates = [
        "System Scope and Architecture",
        "Functional & Operational Specifications",
        "External Interface Protocols",
        "Safety Hazard & Fault Management"
    ]

    # Print table header
    header_str = f"{'Reference Standard Section':<30} | " + " | ".join([f"Cand {i+1}" for i in range(len(candidates))])
    print(header_str)
    print("-" * len(header_str))

    for ref in ref_sections:
        scores = [matcher.compute_similarity(ref, cand) for cand in candidates]
        row_scores = " | ".join([f" {s:.4f} " for s in scores])
        print(f"{ref:<30} | {row_scores}")

    print("\nLegend for Candidates:")
    for idx, cand in enumerate(candidates, start=1):
        print(f"  Cand {idx}: \"{cand}\"")
    print("Notice: The diagonal values (true semantic matches) clearly dominate each row.")

    # --------------------------------------------------------------------------
    # DEMO 3: Negative / Out-of-Domain Rejection
    # --------------------------------------------------------------------------
    print_header("DEMO 3: NEGATIVE MATCHING / IRRELEVANT SECTION REJECTION")
    print("Verifying that unrelated non-engineering sections receive low similarity scores:\n")

    negative_pairs = [
        ("Safety and Failure Handling", "Marketing Strategy & Commercial Pricing Plans"),
        ("Interface Requirements", "Human Resources & Employee Benefit Guidelines"),
        ("Functional Requirements", "Cafeteria Lunch Menu and Dietary Accommodations"),
        ("Scope and System Overview", "Legal Disclaimer and Trademark Registration"),
    ]

    print(f"{'Standard Section Name':<28} | {'Irrelevant Candidate':<46} | {'Score':<7} | {'Verdict'}")
    print("-" * 92)
    for ref_name, candidate in negative_pairs:
        score = matcher.compute_similarity(ref_name, candidate)
        verdict = "REJECTED (✓)" if score < 0.55 else "FALSE MATCH (✗)"
        print(f"{ref_name:<28} | {candidate:<46} | {score:<7.4f} | {verdict}")

    # --------------------------------------------------------------------------
    # DEMO 4: Live Document Section Alignment on avionics_sample_srs.md
    # --------------------------------------------------------------------------
    print_header("DEMO 4: LIVE SECTION ALIGNMENT ON 'data/avionics_sample_srs.md'")

    sample_path = os.path.join(BASE_DIR, "data", "avionics_sample_srs.md")
    standard_path = os.path.join(BASE_DIR, "data", "reference_standard.json")

    standard = ReferenceStandard(standard_path)
    extractor = RequirementExtractor()
    doc_sections = extractor.extract_sections_from_file(sample_path)

    print(f"Detected Document Sections ({len(doc_sections)}):")
    for s in doc_sections:
        print(f"  - Line {s.line_number}: \"{s.title}\" (clean: \"{s.clean_title}\")")

    print("\nRunning Semantic Alignment with all-MiniLM-L6-v2 (Threshold: 0.55):\n")
    matches = matcher.match_sections(standard.sections, doc_sections, threshold=0.55)

    for m in matches:
        status_tag = "✅ MATCHED" if m.is_matched else "❌ MISSING"
        print(f"Section {m.reference_section_id}: '{m.reference_section_name}' → {status_tag}")
        if m.is_matched:
            print(f"    Matched to: \"{m.matched_document_section}\" | Similarity Score: {m.similarity_score:.4f}")
        else:
            top_cand = max(m.candidate_scores.items(), key=lambda x: x[1]) if m.candidate_scores else ("None", 0.0)
            print(f"    Highest Candidate: \"{top_cand[0]}\" | Similarity Score: {top_cand[1]:.4f} (< 0.55)")
            print(f"    Finding: Omitted mandatory section detected automatically!")
        print()

    print("=" * 76)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY.")
    print("=" * 76)


if __name__ == "__main__":
    run_demo()
