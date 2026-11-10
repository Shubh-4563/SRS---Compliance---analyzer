"""
V2 Benchmark Evaluation Models (Phase 2.6.3).
Defines strongly typed evaluation data structures for comparing actual pipeline
predictions against declared benchmark manifest ground truth:
- Retrieval metrics (Hit@1, Hit@3, Hit@5, MRR, Precision@K, Recall@K)
- Verification metrics (Accuracy, Macro-F1, Confusion Matrix, Per-status Precision/Recall)
- Aggregation & Compliance metrics (Parent status accuracy, Verdict exact match, Coverage diff)
- Diagnostic discrepancies for root-cause error analysis
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union


@dataclass
class SubRequirementRetrievalMetric:
    """Retrieval performance evaluation for an individual atomic sub-requirement."""
    sub_id: str
    evidence_presence: str
    expected_blocks: List[str]
    retrieved_blocks: List[str]
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    first_relevant_rank: Optional[int]
    reciprocal_rank: float
    precision_at_k: float
    recall_at_k: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sub_id": self.sub_id,
            "evidence_presence": self.evidence_presence,
            "expected_blocks": list(self.expected_blocks),
            "retrieved_blocks": list(self.retrieved_blocks),
            "hit_at_1": self.hit_at_1,
            "hit_at_3": self.hit_at_3,
            "hit_at_5": self.hit_at_5,
            "first_relevant_rank": self.first_relevant_rank,
            "reciprocal_rank": round(self.reciprocal_rank, 4),
            "precision_at_k": round(self.precision_at_k, 4),
            "recall_at_k": round(self.recall_at_k, 4),
        }


@dataclass
class SubRequirementVerificationMetric:
    """Verification classification evaluation for an individual atomic sub-requirement."""
    sub_id: str
    parent_point_id: str
    expected_status: str
    actual_status: str
    is_match: bool
    confidence: float
    best_evidence_block: Optional[str]
    actual_rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sub_id": self.sub_id,
            "parent_point_id": self.parent_point_id,
            "expected_status": self.expected_status,
            "actual_status": self.actual_status,
            "is_match": self.is_match,
            "confidence": round(self.confidence, 4),
            "best_evidence_block": self.best_evidence_block,
            "actual_rationale": self.actual_rationale,
        }


@dataclass
class ParentComplianceMetric:
    """Parent requirement compliance status evaluation."""
    parent_id: str
    expected_status: str
    actual_status: str
    is_match: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "expected_status": self.expected_status,
            "actual_status": self.actual_status,
            "is_match": self.is_match,
        }


@dataclass
class AggregateRetrievalMetrics:
    """Macro-level retrieval metrics aggregated across all evaluated sub-requirements."""
    evaluated_count: int
    hit_at_1_pct: float
    hit_at_3_pct: float
    hit_at_5_pct: float
    mrr: float
    mean_precision_at_k: float
    mean_recall_at_k: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluated_count": self.evaluated_count,
            "hit_at_1_pct": round(self.hit_at_1_pct, 1),
            "hit_at_3_pct": round(self.hit_at_3_pct, 1),
            "hit_at_5_pct": round(self.hit_at_5_pct, 1),
            "mrr": round(self.mrr, 4),
            "mean_precision_at_k": round(self.mean_precision_at_k, 4),
            "mean_recall_at_k": round(self.mean_recall_at_k, 4),
        }


@dataclass
class StatusClassMetric:
    """Classification metrics for a single verification status category."""
    support: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "support": self.support,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class AggregateVerificationMetrics:
    """Macro-level verification metrics aggregated across all atomic sub-requirements."""
    total_sub_requirements: int
    correct_predictions: int
    accuracy_pct: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_status: Dict[str, StatusClassMetric] = field(default_factory=dict)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_sub_requirements": self.total_sub_requirements,
            "correct_predictions": self.correct_predictions,
            "accuracy_pct": round(self.accuracy_pct, 1),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "per_status": {k: v.to_dict() for k, v in self.per_status.items()},
            "confusion_matrix": self.confusion_matrix,
        }


@dataclass
class AggregateComplianceMetrics:
    """Document and parent section level rollup metrics."""
    total_parents: int
    correct_parents: int
    parent_accuracy_pct: float
    expected_verdict: str
    actual_verdict: str
    verdict_matches: bool
    expected_mandatory_coverage_pct: float
    actual_mandatory_coverage_pct: float
    mandatory_coverage_diff: float
    expected_optional_coverage_pct: Optional[float]
    actual_optional_coverage_pct: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_parents": self.total_parents,
            "correct_parents": self.correct_parents,
            "parent_accuracy_pct": round(self.parent_accuracy_pct, 1),
            "expected_verdict": self.expected_verdict,
            "actual_verdict": self.actual_verdict,
            "verdict_matches": self.verdict_matches,
            "expected_mandatory_coverage_pct": round(self.expected_mandatory_coverage_pct, 1),
            "actual_mandatory_coverage_pct": round(self.actual_mandatory_coverage_pct, 1),
            "mandatory_coverage_diff": round(self.mandatory_coverage_diff, 1),
            "expected_optional_coverage_pct": (
                round(self.expected_optional_coverage_pct, 1)
                if self.expected_optional_coverage_pct is not None
                else None
            ),
            "actual_optional_coverage_pct": (
                round(self.actual_optional_coverage_pct, 1)
                if self.actual_optional_coverage_pct is not None
                else None
            ),
        }


@dataclass
class DiscrepancyItem:
    """Individual mismatch or error identified during benchmark evaluation."""
    sub_id: str
    category: str
    description: str
    expected: str
    actual: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sub_id": self.sub_id,
            "category": self.category,
            "description": self.description,
            "expected": self.expected,
            "actual": self.actual,
            "details": self.details,
        }


@dataclass
class BenchmarkEvaluationReport:
    """Complete benchmark evaluation report comparing pipeline execution against manifest."""
    pair_id: str
    domain: str
    title: str
    manifest_path: str
    reference_path: str
    target_path: str
    top_k: int
    retrieval_metrics: AggregateRetrievalMetrics
    verification_metrics: AggregateVerificationMetrics
    compliance_metrics: AggregateComplianceMetrics
    subreq_retrieval_metrics: List[SubRequirementRetrievalMetric] = field(default_factory=list)
    subreq_verification_metrics: List[SubRequirementVerificationMetric] = field(default_factory=list)
    parent_metrics: List[ParentComplianceMetric] = field(default_factory=list)
    discrepancies: List[DiscrepancyItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "domain": self.domain,
            "title": self.title,
            "manifest_path": self.manifest_path,
            "reference_path": self.reference_path,
            "target_path": self.target_path,
            "top_k": self.top_k,
            "retrieval_metrics": self.retrieval_metrics.to_dict(),
            "verification_metrics": self.verification_metrics.to_dict(),
            "compliance_metrics": self.compliance_metrics.to_dict(),
            "subreq_retrieval_metrics": [m.to_dict() for m in self.subreq_retrieval_metrics],
            "subreq_verification_metrics": [m.to_dict() for m in self.subreq_verification_metrics],
            "parent_metrics": [m.to_dict() for m in self.parent_metrics],
            "discrepancies": [d.to_dict() for d in self.discrepancies],
            "has_discrepancies": len(self.discrepancies) > 0,
        }

    def format_terminal_summary(self) -> str:
        """Emits a concise, ANSI-friendly terminal summary without markdown tables."""
        lines = []
        lines.append("================================================================================")
        lines.append(f"BENCHMARK ACCURACY EVALUATION: {self.pair_id}")
        lines.append(f"Title:  {self.title}")
        lines.append(f"Domain: {self.domain}")
        lines.append(f"Target: {self.target_path}")
        lines.append("================================================================================")

        # Retrieval metrics
        r = self.retrieval_metrics
        lines.append("\n--- Candidate Retrieval Metrics ---")
        lines.append(f"  Evaluated Sub-requirements: {r.evaluated_count}")
        lines.append(f"  Hit@1 Rate:                 {r.hit_at_1_pct:.1f}%")
        lines.append(f"  Hit@3 Rate:                 {r.hit_at_3_pct:.1f}%")
        lines.append(f"  Hit@5 Rate:                 {r.hit_at_5_pct:.1f}%")
        lines.append(f"  Mean Reciprocal Rank (MRR): {r.mrr:.4f}")
        lines.append(f"  Mean Precision@{self.top_k}:         {r.mean_precision_at_k:.4f}")
        lines.append(f"  Mean Recall@{self.top_k}:            {r.mean_recall_at_k:.4f}")

        # Verification metrics
        v = self.verification_metrics
        lines.append("\n--- Evidence Verification Metrics ---")
        lines.append(f"  Atomic Verification Accuracy: {v.accuracy_pct:.1f}% ({v.correct_predictions}/{v.total_sub_requirements})")
        lines.append(f"  Macro Precision:              {v.macro_precision:.4f}")
        lines.append(f"  Macro Recall:                 {v.macro_recall:.4f}")
        lines.append(f"  Macro F1 Score:               {v.macro_f1:.4f}")

        lines.append("\n  Per-Status Breakdown:")
        for status_name, sm in v.per_status.items():
            if sm.support > 0:
                lines.append(f"    * {status_name}: F1={sm.f1:.4f} (P={sm.precision:.4f}, R={sm.recall:.4f}, Support={sm.support})")

        # Compliance metrics
        c = self.compliance_metrics
        lines.append("\n--- Compliance Aggregation Metrics ---")
        lines.append(f"  Parent Section Accuracy:  {c.parent_accuracy_pct:.1f}% ({c.correct_parents}/{c.total_parents})")
        lines.append(f"  Document Verdict Match:   {'PASS' if c.verdict_matches else 'FAIL'} (Expected: {c.expected_verdict}, Actual: {c.actual_verdict})")
        lines.append(f"  Mandatory Coverage Pct:   Actual {c.actual_mandatory_coverage_pct:.1f}% vs Expected {c.expected_mandatory_coverage_pct:.1f}% (Diff: {c.mandatory_coverage_diff:.1f}%)")
        if c.expected_optional_coverage_pct is not None:
            actual_opt = f"{c.actual_optional_coverage_pct:.1f}%" if c.actual_optional_coverage_pct is not None else "N/A"
            lines.append(f"  Optional Coverage Pct:    Actual {actual_opt} vs Expected {c.expected_optional_coverage_pct:.1f}%")

        # Discrepancies
        lines.append("\n--- Diagnostic Discrepancies ---")
        if not self.discrepancies:
            lines.append("  None. 100% agreement between pipeline predictions and ground truth.")
        else:
            lines.append(f"  Found {len(self.discrepancies)} discrepancy item(s):")
            for disc in self.discrepancies:
                lines.append(f"  * [{disc.sub_id}] ({disc.category}): {disc.description}")
                lines.append(f"      Expected: {disc.expected}")
                lines.append(f"      Actual:   {disc.actual}")

        lines.append("================================================================================")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Emits a structured GitHub-Flavored Markdown report without tables."""
        lines = []
        lines.append(f"# Benchmark Evaluation Report: {self.pair_id}")
        lines.append(f"**Title**: {self.title}")
        lines.append(f"**Domain**: {self.domain}")
        lines.append(f"**Target Document**: `{self.target_path}`")
        lines.append(f"**Manifest File**: `{self.manifest_path}`")
        lines.append(f"**Retrieval Candidate Depth**: Top-{self.top_k}")

        lines.append("\n## 1. Candidate Retrieval Evaluation")
        r = self.retrieval_metrics
        lines.append(f"* Sub-requirements Evaluated: {r.evaluated_count}")
        lines.append(f"* Hit@1 Success Rate: {r.hit_at_1_pct:.1f}%")
        lines.append(f"* Hit@3 Success Rate: {r.hit_at_3_pct:.1f}%")
        lines.append(f"* Hit@5 Success Rate: {r.hit_at_5_pct:.1f}%")
        lines.append(f"* Mean Reciprocal Rank (MRR): {r.mrr:.4f}")
        lines.append(f"* Mean Precision@{self.top_k}: {r.mean_precision_at_k:.4f}")
        lines.append(f"* Mean Recall@{self.top_k}: {r.mean_recall_at_k:.4f}")

        lines.append("\n## 2. Evidence Verification Evaluation")
        v = self.verification_metrics
        lines.append(f"* Atomic Verification Accuracy: {v.accuracy_pct:.1f}% ({v.correct_predictions}/{v.total_sub_requirements})")
        lines.append(f"* Macro Precision: {v.macro_precision:.4f}")
        lines.append(f"* Macro Recall: {v.macro_recall:.4f}")
        lines.append(f"* Macro F1 Score: {v.macro_f1:.4f}")

        lines.append("\n### Per-Status Classification Metrics")
        for status_name, sm in v.per_status.items():
            if sm.support > 0:
                lines.append(f"* **{status_name}**:")
                lines.append(f"  * Ground Truth Support: {sm.support}")
                lines.append(f"  * Precision: {sm.precision:.4f}")
                lines.append(f"  * Recall: {sm.recall:.4f}")
                lines.append(f"  * F1 Score: {sm.f1:.4f}")

        lines.append("\n## 3. Compliance Aggregation & Scoring Evaluation")
        c = self.compliance_metrics
        lines.append(f"* Parent Section Status Accuracy: {c.parent_accuracy_pct:.1f}% ({c.correct_parents}/{c.total_parents})")
        lines.append(f"* Overall Document Verdict: {'MATCH' if c.verdict_matches else 'MISMATCH'}")
        lines.append(f"  * Expected: `{c.expected_verdict}`")
        lines.append(f"  * Actual: `{c.actual_verdict}`")
        lines.append(f"* Mandatory Coverage Percentage: Actual `{c.actual_mandatory_coverage_pct:.1f}%` vs Expected `{c.expected_mandatory_coverage_pct:.1f}%` (Diff: {c.mandatory_coverage_diff:.1f}%)")
        if c.expected_optional_coverage_pct is not None:
            actual_opt = f"`{c.actual_optional_coverage_pct:.1f}%`" if c.actual_optional_coverage_pct is not None else "`N/A`"
            lines.append(f"* Optional Coverage Percentage: Actual {actual_opt} vs Expected `{c.expected_optional_coverage_pct:.1f}%`")

        lines.append("\n## 4. Diagnostic Discrepancies")
        if not self.discrepancies:
            lines.append("* Zero discrepancies detected. Perfect alignment between model predictions and declared ground truth.")
        else:
            for disc in self.discrepancies:
                lines.append(f"* **[{disc.sub_id}] {disc.category}**: {disc.description}")
                lines.append(f"  * Expected: `{disc.expected}`")
                lines.append(f"  * Actual: `{disc.actual}`")

        return "\n".join(lines)
