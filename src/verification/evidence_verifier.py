"""
V2 Atomic Requirement Evidence Verifier (Phase 2.3).
Synthesizes candidate NLI evaluations, handles explicit omissions, validates numeric targets,
detects evidence conflicts, and assigns verifiable compliance states with strict provenance.
"""
import re
from typing import List, Optional, Tuple, Dict, Any, Set

from src.models.requirement import EvidenceType, SubRequirement
from src.models.retrieval import CandidateEvidence
from src.models.verification import (
    VerificationStatus,
    EvidenceEvaluation,
    SubRequirementVerificationResult,
)
from src.verification.nli_verifier import NLIVerifier


# Decision thresholds derived from empirical benchmark distributions
ENTAILMENT_THRESHOLD = 0.50     # Dominant entailment required for fulfillment
CONTRADICTION_THRESHOLD = 0.50  # Dominant contradiction required for disproof/omission
CONFLICT_MARGIN = 0.40          # Score at which competing signals trigger review

# Stop words to filter out when assessing topical relevance between disclaimers and sub-requirements
GENERIC_STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "these", "those",
    "state", "states", "describe", "describes", "provide", "provides", "include", "includes",
    "system", "systems", "document", "documents", "experiment", "experiments", "report", "reports",
    "identify", "identifies", "explain", "explains", "which", "during", "about",
    "were", "been", "have", "used", "will", "show", "shows", "also", "into", "over",
    "main", "other", "some", "such", "than", "more", "less", "each", "both", "their", "there",
    "what", "when", "where", "how", "well", "such"
}


class EvidenceVerifier:
    """
    Verifies an atomic SubRequirement against its top retrieved candidate evidence blocks.
    Enforces all V2 verification rules:
    - Retrieval similarity is never proof of compliance.
    - Explicit omissions are isolated and topic-matched.
    - Contradiction is distinguished from missing evidence.
    - Numeric targets must contain concrete measurable values.
    - Conflicting evidence flags NEEDS_REVIEW.
    """

    def __init__(self, nli_verifier: Optional[NLIVerifier] = None):
        self.nli_verifier = nli_verifier or NLIVerifier()

    @staticmethod
    def _is_topically_relevant(text: str, sub_req: SubRequirement) -> bool:
        """
        Determines whether an omission disclaimer specifically pertains to
        the sub-requirement's specific topic rather than an unrelated section.
        """
        t_lower = text.lower()
        topic_context = f"{sub_req.topic_label or ''} {sub_req.claim_text or ''}".lower()

        # Extract domain tokens (length >= 4, non-stopwords)
        domain_tokens = set(re.findall(r"\b[a-z]{4,}\b", topic_context)) - GENERIC_STOP_WORDS
        if not domain_tokens:
            return True

        for token in domain_tokens:
            if token in t_lower:
                return True
            if len(token) >= 5 and (token[:4] in t_lower or token[:-1] in t_lower):
                return True

        return False

    def verify_subrequirement(
        self,
        sub_req: SubRequirement,
        candidates: List[CandidateEvidence]
    ) -> SubRequirementVerificationResult:
        """
        Evaluates top candidates (e.g. Top-5) and determines the verification status.

        Args:
            sub_req: The atomic SubRequirement to verify.
            candidates: Retrieved CandidateEvidence blocks sorted by RRF rank.

        Returns:
            SubRequirementVerificationResult populated with status, confidence, and provenance.
        """
        if not candidates:
            return SubRequirementVerificationResult(
                sub_id=sub_req.sub_id,
                parent_point_id=sub_req.parent_point_id,
                claim_text=sub_req.claim_text,
                expected_evidence_type=sub_req.expected_evidence_type,
                status=VerificationStatus.MISSING,
                confidence=1.0,
                rationale="No candidate evidence blocks were retrieved from the target document.",
                source_page=sub_req.source_page,
                section_name=sub_req.section_name,
                topic_label=sub_req.topic_label,
                original_text=sub_req.original_text
            )

        # 1. Run NLI evaluations on all candidates
        evaluations = self.nli_verifier.evaluate_candidates(candidates, sub_req)

        # 2. Separate candidates by role
        topical_omissions = [
            e for e in evaluations
            if e.candidate.is_explicit_omission and self._is_topically_relevant(e.candidate.text, sub_req)
        ]
        regular_evals = [e for e in evaluations if not e.candidate.is_explicit_omission]

        # 3. Identify valid positive evidence
        # For NUMERIC_TARGET: evidence is only valid positive if it meets entailment threshold AND contains numeric values
        if sub_req.expected_evidence_type == EvidenceType.NUMERIC_TARGET:
            valid_positive_evals = [
                e for e in regular_evals
                if e.entailment_prob >= ENTAILMENT_THRESHOLD and e.has_numeric_content
            ]
        else:
            valid_positive_evals = [
                e for e in regular_evals
                if e.entailment_prob >= ENTAILMENT_THRESHOLD
            ]

        best_positive = max(valid_positive_evals, key=lambda e: e.entailment_prob) if valid_positive_evals else None

        # 4. Handle Explicit Omissions on this topic (Rules 1 & 2)
        if topical_omissions:
            best_om = max(topical_omissions, key=lambda e: e.contradiction_prob)
            if best_om.contradiction_prob >= CONTRADICTION_THRESHOLD:
                if not best_positive:
                    # Rule 1: Explicit omission with no valid positive evidence -> EXPLICIT_OMISSION
                    return SubRequirementVerificationResult(
                        sub_id=sub_req.sub_id,
                        parent_point_id=sub_req.parent_point_id,
                        claim_text=sub_req.claim_text,
                        expected_evidence_type=sub_req.expected_evidence_type,
                        status=VerificationStatus.EXPLICIT_OMISSION,
                        confidence=best_om.contradiction_prob,
                        best_evidence=best_om,
                        all_evaluations=evaluations,
                        rationale=f"Target document explicitly disclaims or omits this requirement in block {best_om.candidate.block_id}.",
                        source_page=sub_req.source_page,
                        section_name=sub_req.section_name,
                        topic_label=sub_req.topic_label,
                        original_text=sub_req.original_text
                    )
                else:
                    # Rule 2: Positive evidence covering only part of requirement alongside an explicit disclaimer -> NEEDS_REVIEW
                    return SubRequirementVerificationResult(
                        sub_id=sub_req.sub_id,
                        parent_point_id=sub_req.parent_point_id,
                        claim_text=sub_req.claim_text,
                        expected_evidence_type=sub_req.expected_evidence_type,
                        status=VerificationStatus.NEEDS_REVIEW,
                        confidence=min(best_positive.entailment_prob, best_om.contradiction_prob),
                        best_evidence=best_positive,
                        all_evaluations=evaluations,
                        rationale=(
                            f"Partial evidence conflict on topic: block {best_positive.candidate.block_id} provides positive evidence "
                            f"(entailment={best_positive.entailment_prob:.2f}), but block {best_om.candidate.block_id} "
                            f"explicitly disclaims part of the requirement (contradiction={best_om.contradiction_prob:.2f})."
                        ),
                        needs_review=True,
                        source_page=sub_req.source_page,
                        section_name=sub_req.section_name,
                        topic_label=sub_req.topic_label,
                        original_text=sub_req.original_text
                    )

        # 5. Handle Valid Positive Evidence (Rules 3 & 4)
        if best_positive:
            # Rule 3: Check for genuine contradictory evidence on the same claim
            topical_contra = [
                e for e in regular_evals
                if e.candidate.block_id != best_positive.candidate.block_id
                and e.contradiction_prob >= CONFLICT_MARGIN
                and self._is_topically_relevant(e.candidate.text, sub_req)
            ]
            if topical_contra:
                worst_contra = max(topical_contra, key=lambda e: e.contradiction_prob)
                return SubRequirementVerificationResult(
                    sub_id=sub_req.sub_id,
                    parent_point_id=sub_req.parent_point_id,
                    claim_text=sub_req.claim_text,
                    expected_evidence_type=sub_req.expected_evidence_type,
                    status=VerificationStatus.NEEDS_REVIEW,
                    confidence=min(best_positive.entailment_prob, worst_contra.contradiction_prob),
                    best_evidence=best_positive,
                    all_evaluations=evaluations,
                    rationale=(
                        f"Conflicting evidence detected on same claim: block {best_positive.candidate.block_id} supports "
                        f"(entailment={best_positive.entailment_prob:.2f}), but block {worst_contra.candidate.block_id} contradicts "
                        f"(contradiction={worst_contra.contradiction_prob:.2f})."
                    ),
                    needs_review=True,
                    source_page=sub_req.source_page,
                    section_name=sub_req.section_name,
                    topic_label=sub_req.topic_label,
                    original_text=sub_req.original_text
                )

            # Rule 4: Uncontested valid positive evidence -> FULFILLED
            if sub_req.expected_evidence_type == EvidenceType.NUMERIC_TARGET:
                rationale_text = (
                    f"Satisfied by numeric evidence block {best_positive.candidate.block_id} with "
                    f"entailment probability {best_positive.entailment_prob:.2f} (values: {', '.join(best_positive.numeric_tokens[:4])})."
                )
            else:
                rationale_text = (
                    f"Satisfied by evidence block {best_positive.candidate.block_id} with "
                    f"entailment probability {best_positive.entailment_prob:.2f}."
                )

            return SubRequirementVerificationResult(
                sub_id=sub_req.sub_id,
                parent_point_id=sub_req.parent_point_id,
                claim_text=sub_req.claim_text,
                expected_evidence_type=sub_req.expected_evidence_type,
                status=VerificationStatus.FULFILLED,
                confidence=best_positive.entailment_prob,
                best_evidence=best_positive,
                all_evaluations=evaluations,
                rationale=rationale_text,
                source_page=sub_req.source_page,
                section_name=sub_req.section_name,
                topic_label=sub_req.topic_label,
                original_text=sub_req.original_text
            )

        # 6. Check for NUMERIC_TARGET that discusses the topic but lacks concrete values
        if sub_req.expected_evidence_type == EvidenceType.NUMERIC_TARGET:
            qualitative_match = [
                e for e in regular_evals
                if e.entailment_prob >= ENTAILMENT_THRESHOLD and not e.has_numeric_content
            ]
            if qualitative_match:
                best_qual = max(qualitative_match, key=lambda e: e.entailment_prob)
                return SubRequirementVerificationResult(
                    sub_id=sub_req.sub_id,
                    parent_point_id=sub_req.parent_point_id,
                    claim_text=sub_req.claim_text,
                    expected_evidence_type=sub_req.expected_evidence_type,
                    status=VerificationStatus.NEEDS_REVIEW,
                    confidence=best_qual.entailment_prob,
                    best_evidence=best_qual,
                    all_evaluations=evaluations,
                    rationale=(
                        f"Block {best_qual.candidate.block_id} discusses the required topic "
                        f"(entailment={best_qual.entailment_prob:.2f}), but lacks concrete measurable numeric values, units, or thresholds."
                    ),
                    needs_review=True,
                    source_page=sub_req.source_page,
                    section_name=sub_req.section_name,
                    topic_label=sub_req.topic_label,
                    original_text=sub_req.original_text
                )

        # 7. Check for topical contradiction on non-disclaimer blocks
        topical_contra_only = [
            e for e in regular_evals
            if e.contradiction_prob >= CONTRADICTION_THRESHOLD
            and self._is_topically_relevant(e.candidate.text, sub_req)
        ]
        if topical_contra_only:
            worst_contra = max(topical_contra_only, key=lambda e: e.contradiction_prob)
            return SubRequirementVerificationResult(
                sub_id=sub_req.sub_id,
                parent_point_id=sub_req.parent_point_id,
                claim_text=sub_req.claim_text,
                expected_evidence_type=sub_req.expected_evidence_type,
                status=VerificationStatus.CONTRADICTED,
                confidence=worst_contra.contradiction_prob,
                best_evidence=worst_contra,
                all_evaluations=evaluations,
                rationale=f"Retrieved evidence in block {worst_contra.candidate.block_id} directly contradicts requirement claim on same topic.",
                source_page=sub_req.source_page,
                section_name=sub_req.section_name,
                topic_label=sub_req.topic_label,
                original_text=sub_req.original_text
            )

        # 8. No candidate entails and no topical contradiction/omission -> Missing Evidence
        best_cand = max(evaluations, key=lambda e: e.entailment_prob) if evaluations else None
        avg_neutral = sum(e.neutral_prob for e in evaluations) / len(evaluations)
        return SubRequirementVerificationResult(
            sub_id=sub_req.sub_id,
            parent_point_id=sub_req.parent_point_id,
            claim_text=sub_req.claim_text,
            expected_evidence_type=sub_req.expected_evidence_type,
            status=VerificationStatus.MISSING,
            confidence=avg_neutral,
            best_evidence=best_cand,
            all_evaluations=evaluations,
            rationale=(
                f"No retrieved candidate evidence logically entails the requirement claim "
                f"(maximum entailment={best_cand.entailment_prob if best_cand else 0.0:.2f}, average neutral={avg_neutral:.2f})."
            ),
            source_page=sub_req.source_page,
            section_name=sub_req.section_name,
            topic_label=sub_req.topic_label,
            original_text=sub_req.original_text
        )
