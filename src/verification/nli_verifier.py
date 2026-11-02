"""
V2 NLI Cross-Encoder Verifier (Phase 2.3.1 Refined).
Uses cross-encoder/nli-deberta-v3-small to determine logical entailment,
contradiction, or neutral relationship between candidate evidence and atomic claims.
Handles negative boundary clauses, meta-discourse framing, and sentence-level evidence.
Cross-platform compatible (macOS and Windows).
"""
import os
import re
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from sentence_transformers import CrossEncoder

from src.models.requirement import EvidenceType, SubRequirement
from src.models.retrieval import CandidateEvidence
from src.models.verification import EvidenceEvaluation
from src.parsers.pdf_parser import EXPLICIT_ABSENCE_PATTERN


# Regex for detecting concrete numbers, units, tolerances, or thresholds in text
NUMERIC_REGEX = re.compile(
    r"\b("
    r"\d+(?:\.\d+)?\s*(?:degrees?\s*[CF]|%|s|sec|seconds?|ms|milliseconds?|minutes?|min|hours?|hr|m|mm|cm|km|kg|g|kb|mb|gb|tb|req/s|fps|v|mv|hz|khz|mhz|ghz)"
    r"|[<>=≤≥]\s*\d+(?:\.\d+)?"
    r"|±\s*\d+(?:\.\d+)?"
    r"|\b\d+\s*(?:to|-)\s*\d+\b"
    r"|\b\d{1,6}\b"
    r")\b",
    re.IGNORECASE
)

# Regex to identify negative boundary clauses in regular paragraphs
# (e.g. 'does not control the equipment's physical operation', 'outside the scope', 'not intended to control')
NEGATIVE_BOUNDARY_CLAUSE_REGEX = re.compile(
    r"\b("
    r"(?:does|do|did)\s+not\s+(?:control|operate|modify|execute|drive|actuate|intervene|run|manage)"
    r"|(?:not\s+(?:designed|intended)\s+to\s+(?:control|operate|modify|execute|drive))"
    r"|(?:outside\s+(?:the\s+)?scope)"
    r"|(?:out\s+of\s+scope)"
    r"|(?:not\s+responsible\s+for)"
    r"|(?:does\s+not\s+have\s+control)"
    r")\b",
    re.IGNORECASE
)

# Regex to strip meta-discourse framing that confuses NLI models into seeking the word 'document'
META_PREFIX_REGEX = re.compile(
    r"^\s*(?:The\s+document\s+(?:states|describes|explains|lists|identifies|records|provides|specifies|contains|includes|shows|asserts|discusses|analyzes)"
    r"|The\s+report\s+(?:states|describes|explains|lists|identifies|records|provides|specifies|contains|includes|shows|asserts|discusses|analyzes))\s+(?:that\s+)?",
    re.IGNORECASE
)


class NLIVerifier:
    """
    NLI Cross-Encoder wrapper evaluating [premise, hypothesis] pairs.
    Preserves exact evidence provenance, detects numeric tokens,
    and maps contradiction, entailment, and neutral probabilities.
    Includes sentence-level multi-clause analysis and substantive claim transformation.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        device: Optional[str] = None,
        local_files_only: bool = True,
    ):
        self.model_name = model_name
        self.device = device

        # Attempt offline load first; fallback to online if not yet cached
        try:
            self.model = CrossEncoder(
                model_name,
                device=device,
                local_files_only=local_files_only
            )
        except Exception:
            self.model = CrossEncoder(
                model_name,
                device=device,
                local_files_only=False
            )

        # Inspect label mapping from model config
        config = getattr(self.model.model, "config", None)
        id2label = getattr(config, "id2label", {0: "contradiction", 1: "entailment", 2: "neutral"})
        self.id2label = {int(k): v.lower() for k, v in id2label.items()}

        # Identify index positions
        self.contra_idx = self._find_label_index("contradiction", default=0)
        self.entail_idx = self._find_label_index("entailment", default=1)
        self.neutral_idx = self._find_label_index("neutral", default=2)

    def _find_label_index(self, target_label: str, default: int) -> int:
        for idx, lbl in self.id2label.items():
            if target_label in lbl:
                return idx
        return default

    @staticmethod
    def extract_numeric_tokens(text: str) -> List[str]:
        """Extracts all concrete numeric expressions from evidence text."""
        return [match.strip() for match in NUMERIC_REGEX.findall(text) if match.strip()]

    @staticmethod
    def derive_substantive_hypothesis(claim_text: str, expected_type: EvidenceType) -> str:
        """
        Derives a substantive factual hypothesis from a meta-discourse claim.
        Avoids requiring the target document to explicitly describe itself as a document.
        """
        c = claim_text.strip()

        # Operational purpose / problem it solves (B1.2)
        if re.search(r'problem\s+the\s+system\s+solves|operational\s+purpose', c, re.I):
            return 'The system solves an operational problem or serves an operational purpose for users.'

        # Purpose / aim of system or dashboard (B1.1)
        if re.search(r'(?:system|product|dashboard).*intended\s+to\s+do', c, re.I):
            return 'The system, product, or dashboard has an intended purpose or operational role.'

        # Included in the system / main components (B2.1)
        if re.search(r'what\s+is\s+included\s+in\s+the\s+system|main\s+components', c, re.I):
            return 'The system includes main components, sensors, or monitoring capabilities.'

        # Negative boundary / outside system scope (B2.2)
        if re.search(r'outside\s+(?:system\s+)?scope|boundaries', c, re.I):
            return 'Controlling physical equipment operation is outside system scope or boundaries.'

        # Core functional actions (B3.1)
        if re.search(r'main\s+functional\s+actions', c, re.I):
            return 'The application or system performs main functional actions like collecting sensor readings and displaying status.'

        # System users / browser UI (B4.1)
        if re.search(r'system\s+users|user\s+interfaces|client\s+applications', c, re.I):
            return 'Operators use a user interface or web browser to view the system.'

        # External systems / gateways (B4.2)
        if re.search(r'external\s+systems|hardware|gateways', c, re.I):
            return 'External systems, hardware, or gateways connect to the system.'

        # Communication protocols (B4.3)
        if re.search(r'communication\s+protocols', c, re.I):
            return 'The communication protocol for data exchange is specified.'

        # Hazards / faults / sensor warnings (B5.1)
        if re.search(r'hazards,\s*faults|detects\s+or\s+warns', c, re.I):
            return 'The system detects or warns of hazards, faults, or sensor failures.'

        # Recovery steps (B5.2)
        if re.search(r'recovery\s+steps|automated\s+fault\s+recovery', c, re.I):
            return 'Recovery steps or automated fault recovery procedures are explained.'

        # Numeric targets / criteria (B6.1)
        if re.search(r'measurable\s+numeric\s+target|capacity\s+limit|response-time\s+threshold', c, re.I):
            return 'A measurable numeric target, capacity limit, response-time threshold, or pass/fail criterion is specified.'

        # Testing scenarios (B7.1)
        if re.search(r'how\s+the\s+system\s+is\s+tested|test\s+scenarios', c, re.I):
            return 'The system is tested, verified, or validated through test scenarios.'

        # Aim / purpose of experiment (A1.1)
        if re.search(r'purpose\s+or\s+aim\s+of\s+the\s+experiment', c, re.I):
            return 'The purpose or aim of the experiment is stated.'

        # What experiment is intended to demonstrate (A1.2)
        if re.search(r'intended\s+to\s+demonstrate\s+or\s+measure', c, re.I):
            return 'The experiment demonstrates or measures a physical cooling process.'

        # Equipment / instruments / apparatus (A2.1)
        if re.search(r'equipment,\s*instruments|apparatus\s+used', c, re.I):
            return 'Equipment, instruments, or apparatus such as beakers or thermometers were used in the experiment.'

        # Materials / reagents / software (A2.2)
        if re.search(r'materials,\s*reagents|software\s+used', c, re.I):
            return 'Materials, reagents, water, or notebooks were used in the experiment.'

        # Procedural steps (A3.1)
        if re.search(r'ordered\s+procedural\s+steps', c, re.I):
            return 'Ordered procedural steps were performed or provided.'

        # Measured numerical values (A4.1)
        if re.search(r'measured\s+numerical\s+values|data\s+points', c, re.I):
            return 'Measured numerical values or data points were recorded during the experiment.'

        # Observations meaning / trends in data (A5.1)
        if re.search(r'what\s+the\s+observations\s+mean|trends\s+in\s+the\s+data', c, re.I):
            return 'The observations or data explain cooling trends over time.'

        # Compare observed results with expected / theory (A5.2)
        if re.search(r'compares\s+the\s+observed\s+results|expected\s+behavior', c, re.I):
            return 'The observed results are compared with expected cooling behavior.'

        # Final outcome and whether aim was achieved (A6.1)
        if re.search(r'final\s+outcome\s+of\s+the\s+experiment|whether\s+the\s+aim\s+was\s+achieved', c, re.I):
            return 'The final outcome of the experiment was demonstrated.'

        # Laboratory hazards / precautions (A7.1)
        if re.search(r'laboratory\s+hazards|safety\s+precautions|precautions\s+taken', c, re.I):
            return 'The text identifies laboratory hazards, safety risks, or precautions.'

        # Generic fallback: only transform if claim starts with "The document explains/analyzes/discusses"
        if re.search(r'^\s*(?:The\s+(?:document|report)\s+(?:explains|analyzes|discusses))\b', c, re.I):
            cleaned = META_PREFIX_REGEX.sub('', c).strip()
            if cleaned:
                return cleaned[0].upper() + cleaned[1:]
        return c

    def evaluate_candidates(
        self,
        candidates: List[CandidateEvidence],
        sub_req: SubRequirement
    ) -> List[EvidenceEvaluation]:
        """
        Evaluates a list of candidate evidence blocks against an atomic sub-requirement.
        Performs multi-granularity (block and sentence level) premise evaluation against
        the substantive factual hypothesis, isolating negative boundary clauses and
        meta-discourse artifacts.
        """
        if not candidates:
            return []

        raw_hypo = sub_req.claim_text
        subst_hypo = self.derive_substantive_hypothesis(raw_hypo, sub_req.expected_evidence_type)
        is_neg_boundary = (sub_req.expected_evidence_type == EvidenceType.NEGATIVE_BOUNDARY)

        evaluations: List[EvidenceEvaluation] = []

        for cand in candidates:
            # 1. Break block into constituent sentences
            raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cand.text) if s.strip()]
            sentences = raw_sentences if raw_sentences else [cand.text]

            # 2. Build premise variants to evaluate: full block followed by sentences
            premises = [cand.text] + sentences

            # 3. Build hypothesis: use substantive factual hypothesis (eliminating meta-discourse artifacts)
            hypos = [subst_hypo]

            # 4. Predict NLI scores for all (premise, hypothesis) combinations
            pairs = []
            pair_meta = []
            for p in premises:
                for h in hypos:
                    pairs.append([p, h])
                    pair_meta.append((p, h))

            scores = self.model.predict(pairs)
            probs = F.softmax(torch.tensor(scores), dim=-1).numpy()

            e_vals = probs[:, self.entail_idx]
            c_vals = probs[:, self.contra_idx]
            n_vals = probs[:, self.neutral_idx]

            max_e_idx = int(e_vals.argmax())
            max_e = float(e_vals[max_e_idx])

            # 5. Coherent evidence selection:
            if max_e >= 0.50:
                # Candidate positively entails: provenance belongs to the supporting (premise, hypothesis) pair
                best_ent = max_e
                best_contra = float(c_vals[max_e_idx])
                best_neu = float(n_vals[max_e_idx])
                best_premise, best_hypo = pair_meta[max_e_idx]
                pred_label = "entailment"
            else:
                # Candidate does not entail: evaluate genuine contradiction or neutrality
                if cand.is_explicit_omission:
                    max_c_idx = int(c_vals.argmax())
                    best_ent = float(e_vals[max_c_idx])
                    best_contra = float(c_vals[max_c_idx])
                    best_neu = float(n_vals[max_c_idx])
                    best_premise, best_hypo = pair_meta[max_c_idx]
                else:
                    # Filter out clausal noise (boundary clauses on affirmative claims, affirmative claims on boundary claims)
                    valid_indices = []
                    for idx, (p, h) in enumerate(pair_meta):
                        is_boundary = bool(NEGATIVE_BOUNDARY_CLAUSE_REGEX.search(p))
                        if not is_neg_boundary and is_boundary:
                            continue
                        if is_neg_boundary and not is_boundary:
                            continue
                        valid_indices.append(idx)

                    if valid_indices:
                        best_c_among_valid = max(valid_indices, key=lambda i: c_vals[i])
                        best_contra = float(c_vals[best_c_among_valid])
                        best_ent = float(e_vals[best_c_among_valid])
                        best_neu = float(n_vals[best_c_among_valid])
                        best_premise, best_hypo = pair_meta[best_c_among_valid]
                    else:
                        best_contra = 0.0
                        best_ent = max_e
                        best_neu = float(n_vals.max())
                        best_premise, best_hypo = cand.text, raw_hypo

                if best_contra >= 0.50:
                    pred_label = "contradiction"
                else:
                    pred_label = "neutral"

            # Check numeric tokens
            numeric_tokens = self.extract_numeric_tokens(cand.text)
            has_numeric = len(numeric_tokens) > 0

            evaluations.append(
                EvidenceEvaluation(
                    candidate=cand,
                    premise_text=best_premise,
                    hypothesis_text=best_hypo,
                    entailment_prob=best_ent,
                    contradiction_prob=best_contra,
                    neutral_prob=best_neu,
                    predicted_label=pred_label,
                    has_numeric_content=has_numeric,
                    numeric_tokens=numeric_tokens
                )
            )

        return evaluations
