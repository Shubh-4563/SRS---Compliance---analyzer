"""
V2 Automated Benchmark Evaluation Runner & Accuracy CLI (Phase 2.6.3).
Executes the live compliance pipeline against validated declarative benchmark manifests,
evaluating:
- Candidate retrieval performance (Hit@1, Hit@3, Hit@5, MRR, Precision@K, Recall@K)
- Evidence verification accuracy (Accuracy, Macro-F1, per-status metrics, confusion matrix)
- Hierarchical compliance aggregation (Parent status accuracy, Verdict exact match, Coverage diff)
- Root-cause discrepancy diagnostics
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set

from src.benchmarks.manifest_validator import load_and_validate_manifest
from src.models.benchmark import (
    BenchmarkManifest,
    BenchmarkEvidencePresence,
    BenchmarkVerificationStatus,
)
from src.models.benchmark_evaluation import (
    SubRequirementRetrievalMetric,
    SubRequirementVerificationMetric,
    ParentComplianceMetric,
    AggregateRetrievalMetrics,
    StatusClassMetric,
    AggregateVerificationMetrics,
    AggregateComplianceMetrics,
    DiscrepancyItem,
    BenchmarkEvaluationReport,
)
from src.models.compliance import DocumentComplianceReport
from src.pipeline import CompliancePipeline


class BenchmarkRunner:
    """
    Automated benchmark evaluation engine comparing live pipeline outputs
    against validated declarative benchmark ground-truth manifests.
    """

    def __init__(self, pipeline: Optional[CompliancePipeline] = None):
        """
        Initializes the BenchmarkRunner with an optional injected CompliancePipeline.
        If pipeline is not provided, it will be lazily created on demand.
        """
        self._pipeline = pipeline

    @property
    def pipeline(self) -> CompliancePipeline:
        """Returns injected or lazily-initialized CompliancePipeline."""
        if self._pipeline is None:
            self._pipeline = CompliancePipeline()
        return self._pipeline

    def compute_metrics(
        self,
        manifest: BenchmarkManifest,
        report: DocumentComplianceReport,
        top_k: int = 5,
        manifest_path: str = "",
    ) -> BenchmarkEvaluationReport:
        """
        Calculates retrieval, verification, and aggregation metrics by comparing
        the pipeline report against the benchmark manifest ground truth.
        """
        # Map ground-truth sub-requirements
        gt_subreq_map = {
            sr.sub_id: sr
            for parent in manifest.parent_requirements
            for sr in parent.sub_requirements
        }

        # Map actual sub-requirement verification results
        actual_subreq_map = {}
        for p in report.parent_points:
            for res in p.sub_results:
                actual_subreq_map[res.sub_id] = res

        discrepancies: List[DiscrepancyItem] = []

        # -------------------------------------------------------------------
        # 1. Retrieval Metrics Evaluation
        # -------------------------------------------------------------------
        subreq_retrieval_metrics: List[SubRequirementRetrievalMetric] = []
        evaluated_retrieval_count = 0
        hit_1_count = 0
        hit_3_count = 0
        hit_5_count = 0
        total_rr = 0.0
        total_precision_at_k = 0.0
        total_recall_at_k = 0.0

        for sub_id, gt_sr in gt_subreq_map.items():
            presence = gt_sr.ground_truth.retrieval.evidence_presence
            expected_blocks = [ev.block_id for ev in gt_sr.ground_truth.retrieval.expected_evidence]
            actual_res = actual_subreq_map.get(sub_id)

            retrieved_blocks: List[str] = []
            if actual_res and actual_res.all_evaluations:
                retrieved_blocks = [e.candidate.block_id for e in actual_res.all_evaluations]

            # Retrieval is scored when ground truth specifies expected evidence
            if presence in (BenchmarkEvidencePresence.EXISTS, BenchmarkEvidencePresence.EXPLICIT_OMISSION):
                evaluated_retrieval_count += 1
                expected_set = set(expected_blocks)

                h1 = len(set(retrieved_blocks[:1]) & expected_set) > 0
                h3 = len(set(retrieved_blocks[:3]) & expected_set) > 0
                h5 = len(set(retrieved_blocks[:5]) & expected_set) > 0

                if h1:
                    hit_1_count += 1
                if h3:
                    hit_3_count += 1
                if h5:
                    hit_5_count += 1

                first_rank = None
                for idx, b_id in enumerate(retrieved_blocks, start=1):
                    if b_id in expected_set:
                        first_rank = idx
                        break

                rr = (1.0 / first_rank) if first_rank is not None else 0.0
                total_rr += rr

                top_k_set = set(retrieved_blocks[:top_k])
                p_k = len(top_k_set & expected_set) / max(1, len(top_k_set))
                r_k = len(top_k_set & expected_set) / max(1, len(expected_set))
                total_precision_at_k += p_k
                total_recall_at_k += r_k

                if not h5:
                    discrepancies.append(DiscrepancyItem(
                        sub_id=sub_id,
                        category="RETRIEVAL_MISS",
                        description=f"Expected evidence block(s) {expected_blocks} not found in top-{top_k} candidates.",
                        expected=str(expected_blocks),
                        actual=str(retrieved_blocks[:top_k]),
                        details={"presence": presence.value, "first_rank": first_rank}
                    ))

                subreq_retrieval_metrics.append(SubRequirementRetrievalMetric(
                    sub_id=sub_id,
                    evidence_presence=presence.value,
                    expected_blocks=expected_blocks,
                    retrieved_blocks=retrieved_blocks[:top_k],
                    hit_at_1=h1,
                    hit_at_3=h3,
                    hit_at_5=h5,
                    first_relevant_rank=first_rank,
                    reciprocal_rank=rr,
                    precision_at_k=p_k,
                    recall_at_k=r_k,
                ))
            else:
                # For ABSENT, no evidence was expected
                subreq_retrieval_metrics.append(SubRequirementRetrievalMetric(
                    sub_id=sub_id,
                    evidence_presence=presence.value,
                    expected_blocks=[],
                    retrieved_blocks=retrieved_blocks[:top_k],
                    hit_at_1=True,
                    hit_at_3=True,
                    hit_at_5=True,
                    first_relevant_rank=None,
                    reciprocal_rank=1.0,
                    precision_at_k=1.0,
                    recall_at_k=1.0,
                ))

        if evaluated_retrieval_count > 0:
            agg_retrieval = AggregateRetrievalMetrics(
                evaluated_count=evaluated_retrieval_count,
                hit_at_1_pct=(hit_1_count / evaluated_retrieval_count) * 100.0,
                hit_at_3_pct=(hit_3_count / evaluated_retrieval_count) * 100.0,
                hit_at_5_pct=(hit_5_count / evaluated_retrieval_count) * 100.0,
                mrr=total_rr / evaluated_retrieval_count,
                mean_precision_at_k=total_precision_at_k / evaluated_retrieval_count,
                mean_recall_at_k=total_recall_at_k / evaluated_retrieval_count,
            )
        else:
            agg_retrieval = AggregateRetrievalMetrics(
                evaluated_count=0,
                hit_at_1_pct=100.0,
                hit_at_3_pct=100.0,
                hit_at_5_pct=100.0,
                mrr=1.0,
                mean_precision_at_k=1.0,
                mean_recall_at_k=1.0,
            )

        # -------------------------------------------------------------------
        # 2. Verification Metrics Evaluation
        # -------------------------------------------------------------------
        all_statuses = [
            "FULFILLED",
            "EXPLICIT_OMISSION",
            "CONTRADICTED",
            "MISSING",
            "NEEDS_REVIEW",
        ]
        confusion_matrix: Dict[str, Dict[str, int]] = {
            s_exp: {s_act: 0 for s_act in all_statuses}
            for s_exp in all_statuses
        }

        subreq_verification_metrics: List[SubRequirementVerificationMetric] = []
        correct_verifications = 0
        total_subreqs = len(gt_subreq_map)

        for sub_id, gt_sr in gt_subreq_map.items():
            exp_status = gt_sr.ground_truth.verification.expected_status.value
            actual_res = actual_subreq_map.get(sub_id)

            actual_status = actual_res.status.value if actual_res else "MISSING"
            confidence = actual_res.confidence if actual_res else 0.0
            best_block = (
                actual_res.best_evidence.candidate.block_id
                if actual_res and actual_res.best_evidence
                else None
            )
            rationale = actual_res.rationale if actual_res else "No verification result found"

            is_match = (exp_status == actual_status)
            if is_match:
                correct_verifications += 1
            else:
                discrepancies.append(DiscrepancyItem(
                    sub_id=sub_id,
                    category="VERIFICATION_MISMATCH",
                    description=f"Status mismatch for {sub_id}: expected {exp_status}, got {actual_status}.",
                    expected=exp_status,
                    actual=actual_status,
                    details={
                        "confidence": confidence,
                        "best_block": best_block,
                        "rationale": rationale,
                    }
                ))

            if exp_status in confusion_matrix and actual_status in confusion_matrix[exp_status]:
                confusion_matrix[exp_status][actual_status] += 1

            subreq_verification_metrics.append(SubRequirementVerificationMetric(
                sub_id=sub_id,
                parent_point_id=actual_res.parent_point_id if actual_res else sub_id.split(".")[0],
                expected_status=exp_status,
                actual_status=actual_status,
                is_match=is_match,
                confidence=confidence,
                best_evidence_block=best_block,
                actual_rationale=rationale,
            ))

        # Per-status classification metrics
        per_status_metrics: Dict[str, StatusClassMetric] = {}
        for s in all_statuses:
            tp = confusion_matrix[s][s]
            fp = sum(confusion_matrix[other][s] for other in all_statuses if other != s)
            fn = sum(confusion_matrix[s][other] for other in all_statuses if other != s)
            support = sum(confusion_matrix[s].values())

            precision = (tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if tp == 0 and fp == 0 else 0.0)
            recall = (tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if support == 0 else 0.0)
            f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            per_status_metrics[s] = StatusClassMetric(
                support=support,
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
            )

        # Macro averages across statuses present in ground truth
        active_statuses = [s for s in all_statuses if per_status_metrics[s].support > 0]
        if active_statuses:
            macro_p = sum(per_status_metrics[s].precision for s in active_statuses) / len(active_statuses)
            macro_r = sum(per_status_metrics[s].recall for s in active_statuses) / len(active_statuses)
            macro_f1 = sum(per_status_metrics[s].f1 for s in active_statuses) / len(active_statuses)
        else:
            macro_p = 1.0; macro_r = 1.0; macro_f1 = 1.0

        accuracy_pct = (correct_verifications / total_subreqs * 100.0) if total_subreqs > 0 else 100.0

        agg_verification = AggregateVerificationMetrics(
            total_sub_requirements=total_subreqs,
            correct_predictions=correct_verifications,
            accuracy_pct=accuracy_pct,
            macro_precision=macro_p,
            macro_recall=macro_r,
            macro_f1=macro_f1,
            per_status=per_status_metrics,
            confusion_matrix=confusion_matrix,
        )

        # -------------------------------------------------------------------
        # 3. Compliance Aggregation & Document Metrics
        # -------------------------------------------------------------------
        parent_metrics: List[ParentComplianceMetric] = []
        correct_parents = 0
        actual_parent_map = {p.point_id: p for p in report.parent_points}

        for p in manifest.parent_requirements:
            exp_p_status = p.expected_compliance_status.value
            actual_parent = actual_parent_map.get(p.parent_id)
            actual_p_status = actual_parent.status.value if actual_parent else "NON_COMPLIANT"

            is_p_match = (exp_p_status == actual_p_status)
            if is_p_match:
                correct_parents += 1
            else:
                discrepancies.append(DiscrepancyItem(
                    sub_id=p.parent_id,
                    category="PARENT_STATUS_MISMATCH",
                    description=f"Parent point {p.parent_id} status mismatch: expected {exp_p_status}, got {actual_p_status}.",
                    expected=exp_p_status,
                    actual=actual_p_status,
                    details={"parent_id": p.parent_id}
                ))

            parent_metrics.append(ParentComplianceMetric(
                parent_id=p.parent_id,
                expected_status=exp_p_status,
                actual_status=actual_p_status,
                is_match=is_p_match,
            ))

        total_parents = len(manifest.parent_requirements)
        parent_accuracy_pct = (correct_parents / total_parents * 100.0) if total_parents > 0 else 100.0

        exp_verdict = manifest.expected_document_compliance.verdict.value
        actual_verdict = report.verdict.value
        verdict_matches = (exp_verdict == actual_verdict)

        if not verdict_matches:
            discrepancies.append(DiscrepancyItem(
                sub_id="DOCUMENT_VERDICT",
                category="VERDICT_MISMATCH",
                description=f"Overall document verdict mismatch: expected {exp_verdict}, got {actual_verdict}.",
                expected=exp_verdict,
                actual=actual_verdict,
            ))

        exp_m_cov = manifest.expected_document_compliance.mandatory_coverage_pct
        actual_m_cov = report.mandatory_coverage_pct if report.mandatory_coverage_pct is not None else 0.0
        m_cov_diff = abs(actual_m_cov - exp_m_cov)

        exp_o_cov = manifest.expected_document_compliance.optional_coverage_pct
        actual_o_cov = report.optional_coverage_pct

        agg_compliance = AggregateComplianceMetrics(
            total_parents=total_parents,
            correct_parents=correct_parents,
            parent_accuracy_pct=parent_accuracy_pct,
            expected_verdict=exp_verdict,
            actual_verdict=actual_verdict,
            verdict_matches=verdict_matches,
            expected_mandatory_coverage_pct=exp_m_cov,
            actual_mandatory_coverage_pct=actual_m_cov,
            mandatory_coverage_diff=m_cov_diff,
            expected_optional_coverage_pct=exp_o_cov,
            actual_optional_coverage_pct=actual_o_cov,
        )

        return BenchmarkEvaluationReport(
            pair_id=manifest.metadata.pair_id,
            domain=manifest.metadata.domain,
            title=manifest.metadata.title,
            manifest_path=manifest_path,
            reference_path=manifest.metadata.reference_document.path,
            target_path=manifest.metadata.target_document.path,
            top_k=top_k,
            retrieval_metrics=agg_retrieval,
            verification_metrics=agg_verification,
            compliance_metrics=agg_compliance,
            subreq_retrieval_metrics=subreq_retrieval_metrics,
            subreq_verification_metrics=subreq_verification_metrics,
            parent_metrics=parent_metrics,
            discrepancies=discrepancies,
        )

    def evaluate_manifest(
        self,
        manifest_or_path: Union[str, Path, BenchmarkManifest],
        base_dir: Optional[Union[str, Path]] = None,
        top_k: int = 5,
    ) -> BenchmarkEvaluationReport:
        """
        Loads and validates a benchmark manifest, executes the live CompliancePipeline,
        and returns a comprehensive BenchmarkEvaluationReport.
        """
        if isinstance(manifest_or_path, (str, Path)):
            manifest_path = Path(manifest_or_path).resolve()
            manifest = load_and_validate_manifest(manifest_path, base_dir=base_dir)
            manifest_str = str(manifest_path)
        else:
            manifest = manifest_or_path
            manifest_str = "<in-memory manifest>"

        root_dir = Path(base_dir).resolve() if base_dir is not None else Path(__file__).resolve().parent.parent.parent

        ref_rel = Path(manifest.metadata.reference_document.path)
        tgt_rel = Path(manifest.metadata.target_document.path)

        ref_path = (root_dir / ref_rel).resolve() if not ref_rel.is_absolute() else ref_rel
        tgt_path = (root_dir / tgt_rel).resolve() if not tgt_rel.is_absolute() else tgt_rel

        # Execute live pipeline
        report = self.pipeline.run(
            reference_path=ref_path,
            target_path=tgt_path,
            top_k=top_k,
        )

        return self.compute_metrics(
            manifest=manifest,
            report=report,
            top_k=top_k,
            manifest_path=manifest_str,
        )

    def evaluate_all(
        self,
        benchmarks_dir: Optional[Union[str, Path]] = None,
        top_k: int = 5,
    ) -> List[BenchmarkEvaluationReport]:
        """
        Discovers all benchmark manifests in benchmarks_dir and evaluates each.
        """
        root_dir = Path(__file__).resolve().parent.parent.parent
        bench_dir = Path(benchmarks_dir).resolve() if benchmarks_dir else (root_dir / "data" / "benchmarks")

        if not bench_dir.is_dir():
            raise FileNotFoundError(f"Benchmarks directory not found at: {bench_dir}")

        manifest_files = sorted(bench_dir.glob("*.json"))
        if not manifest_files:
            raise ValueError(f"No benchmark manifest JSON files found in {bench_dir}")

        reports = []
        for mf in manifest_files:
            rep = self.evaluate_manifest(mf, base_dir=root_dir, top_k=top_k)
            reports.append(rep)

        return reports


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="V2 Declarative Benchmark Accuracy Evaluator & Diagnostics CLI."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--manifest",
        type=str,
        help="Path to a single benchmark manifest JSON file to evaluate."
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all declarative benchmark manifests in the benchmarks directory."
    )

    parser.add_argument(
        "--benchmarks-dir",
        type=str,
        default="data/benchmarks",
        help="Directory containing benchmark manifests (default: data/benchmarks)."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Retrieval candidate depth to test (default: 5)."
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Optional path to write evaluation results as JSON."
    )
    parser.add_argument(
        "--output-md",
        type=str,
        help="Optional path to write evaluation report as Markdown."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress terminal summary output."
    )

    args = parser.parse_args()

    runner = BenchmarkRunner()

    if args.manifest:
        reports = [runner.evaluate_manifest(args.manifest, top_k=args.top_k)]
    else:
        reports = runner.evaluate_all(benchmarks_dir=args.benchmarks_dir, top_k=args.top_k)

    # Print terminal summaries
    if not args.quiet:
        for r in reports:
            print(r.format_terminal_summary())
            print()

    # Export JSON if requested
    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        export_data = [r.to_dict() for r in reports] if len(reports) > 1 else reports[0].to_dict()
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
        print(f"Exported benchmark evaluation JSON to: {out_p}")

    # Export Markdown if requested
    if args.output_md:
        out_p = Path(args.output_md)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        md_text = "\n\n---\n\n".join(r.to_markdown() for r in reports)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"Exported benchmark evaluation Markdown to: {out_p}")

    # Return non-zero exit code if any discrepancies were found
    total_discrepancies = sum(len(r.discrepancies) for r in reports)
    if total_discrepancies > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

