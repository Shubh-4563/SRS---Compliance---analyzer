"""
V2 Coverage Aggregation and Compliance Scoring Engine (Phase 2.4).
Synthesizes atomic SubRequirementVerificationResult objects into structured
parent-point compliance assessments and document-level compliance reports.
Deterministic, cross-platform, and fully decoupled from model inference.
"""
from typing import List, Optional, Dict, Any

from src.models.requirement import (
    SubRequirement,
    DecomposedTemplatePoint,
    DecomposedReferenceTemplate,
)
from src.models.verification import (
    VerificationStatus,
    SubRequirementVerificationResult,
)
from src.models.compliance import (
    ParentComplianceStatus,
    DocumentComplianceVerdict,
    ParentPointCompliance,
    DocumentComplianceReport,
)


class ComplianceAggregator:
    """
    Deterministic aggregator that rolls up atomic sub-requirement verification results
    into hierarchical parent-point evaluations and an overall document compliance report.
    """

    def aggregate(
        self,
        template: DecomposedReferenceTemplate,
        verification_results: List[SubRequirementVerificationResult],
        target_document_path: str = "",
        target_document_name: str = "",
        total_pages: Optional[int] = None,
        target_page: Optional[int] = None,
    ) -> DocumentComplianceReport:
        """
        Aggregates verification results against a decomposed reference template.

        Args:
            template: DecomposedReferenceTemplate containing all parent points and sub-requirements.
            verification_results: List of SubRequirementVerificationResult from evidence verifier.
            target_document_path: Path to target PDF document being audited.
            target_document_name: Base name or title of target document.
            total_pages: Total page count of target document if available.

        Returns:
            DocumentComplianceReport populated with hierarchical rollups and audit verdicts.
        """
        # 1. Index verification results by sub_id (handling duplicates deterministically)
        results_by_sub_id: Dict[str, SubRequirementVerificationResult] = {}
        for res in verification_results:
            results_by_sub_id[res.sub_id] = res

        parent_compliances: List[ParentPointCompliance] = []
        all_resolved_results: List[SubRequirementVerificationResult] = []

        total_mandatory_global = 0
        total_optional_global = 0
        mandatory_fulfilled_global = 0
        optional_fulfilled_global = 0

        fulfilled_count_global = 0
        explicit_omission_count_global = 0
        contradicted_count_global = 0
        missing_count_global = 0
        needs_review_count_global = 0
        items_requiring_review: List[SubRequirementVerificationResult] = []

        # 2. Iterate through each template parent point
        for point in template.points:
            point_sub_results: List[SubRequirementVerificationResult] = []

            for sr in point.sub_requirements:
                if sr.sub_id in results_by_sub_id:
                    res = results_by_sub_id[sr.sub_id]
                else:
                    # Missing verification result in input: synthesize deterministic MISSING outcome
                    res = SubRequirementVerificationResult(
                        sub_id=sr.sub_id,
                        parent_point_id=point.point_id,
                        claim_text=sr.claim_text,
                        expected_evidence_type=sr.expected_evidence_type,
                        status=VerificationStatus.MISSING,
                        confidence=1.0,
                        is_mandatory=sr.is_mandatory,
                        rationale="Sub-requirement was not evaluated by verification pipeline.",
                        source_page=sr.source_page,
                        section_name=sr.section_name,
                        topic_label=sr.topic_label,
                        original_text=sr.original_text
                    )

                point_sub_results.append(res)
                all_resolved_results.append(res)

                # Track global counts
                if res.is_mandatory:
                    total_mandatory_global += 1
                    if res.status == VerificationStatus.FULFILLED:
                        mandatory_fulfilled_global += 1
                else:
                    total_optional_global += 1
                    if res.status == VerificationStatus.FULFILLED:
                        optional_fulfilled_global += 1

                if res.status == VerificationStatus.FULFILLED:
                    fulfilled_count_global += 1
                elif res.status == VerificationStatus.EXPLICIT_OMISSION:
                    explicit_omission_count_global += 1
                elif res.status == VerificationStatus.CONTRADICTED:
                    contradicted_count_global += 1
                elif res.status == VerificationStatus.MISSING:
                    missing_count_global += 1
                elif res.status == VerificationStatus.NEEDS_REVIEW:
                    needs_review_count_global += 1
                    items_requiring_review.append(res)

            # Aggregate parent point compliance
            parent_comp = self._aggregate_parent_point(point, point_sub_results)
            parent_compliances.append(parent_comp)

        # 3. Calculate coverage scores
        mandatory_cov_pct = (
            (mandatory_fulfilled_global / total_mandatory_global) * 100.0
            if total_mandatory_global > 0
            else None
        )
        optional_cov_pct = (
            (optional_fulfilled_global / total_optional_global) * 100.0
            if total_optional_global > 0
            else None
        )

        # 4. Determine overall document compliance verdict
        verdict = self._determine_document_verdict(
            total_mandatory=total_mandatory_global,
            mandatory_fulfilled=mandatory_fulfilled_global,
            needs_review_count=needs_review_count_global,
            contradicted_count=contradicted_count_global,
            parent_compliances=parent_compliances,
        )

        # 5. Build executive summary
        summary = self._generate_summary(
            target_name=target_document_name or target_document_path or "Target Document",
            template_title=template.title,
            verdict=verdict,
            mandatory_cov_pct=mandatory_cov_pct,
            optional_cov_pct=optional_cov_pct,
            mandatory_fulfilled=mandatory_fulfilled_global,
            total_mandatory=total_mandatory_global,
            fulfilled_count=fulfilled_count_global,
            omission_count=explicit_omission_count_global,
            contradiction_count=contradicted_count_global,
            missing_count=missing_count_global,
            needs_review_count=needs_review_count_global,
            items_requiring_review=items_requiring_review,
        )

        return DocumentComplianceReport(
            target_document_path=target_document_path,
            target_document_name=target_document_name or (target_document_path.split("/")[-1] if target_document_path else "Target Document"),
            template_title=template.title,
            template_path=template.source_path,
            total_pages=total_pages,
            target_page=target_page,
            total_parent_points=len(template.points),
            total_sub_requirements=len(all_resolved_results),
            parent_points=parent_compliances,
            mandatory_coverage_pct=mandatory_cov_pct,
            optional_coverage_pct=optional_cov_pct,
            total_mandatory=total_mandatory_global,
            total_optional=total_optional_global,
            mandatory_fulfilled=mandatory_fulfilled_global,
            optional_fulfilled=optional_fulfilled_global,
            fulfilled_count=fulfilled_count_global,
            explicit_omission_count=explicit_omission_count_global,
            contradicted_count=contradicted_count_global,
            missing_count=missing_count_global,
            needs_review_count=needs_review_count_global,
            unresolved_count=needs_review_count_global,
            verdict=verdict,
            summary=summary,
            items_requiring_review=items_requiring_review,
        )

    def _aggregate_parent_point(
        self,
        point: DecomposedTemplatePoint,
        sub_results: List[SubRequirementVerificationResult]
    ) -> ParentPointCompliance:
        """
        Applies deterministic parent-level aggregation rules A-E.
        """
        mandatory_results = [r for r in sub_results if r.is_mandatory]
        optional_results = [r for r in sub_results if not r.is_mandatory]

        m_total = len(mandatory_results)
        m_fulfilled = sum(1 for r in mandatory_results if r.status == VerificationStatus.FULFILLED)
        m_missing = sum(1 for r in mandatory_results if r.status == VerificationStatus.MISSING)
        m_omission = sum(1 for r in mandatory_results if r.status == VerificationStatus.EXPLICIT_OMISSION)
        m_contra = sum(1 for r in mandatory_results if r.status == VerificationStatus.CONTRADICTED)
        m_review = sum(1 for r in mandatory_results if r.status == VerificationStatus.NEEDS_REVIEW)

        o_total = len(optional_results)
        o_fulfilled = sum(1 for r in optional_results if r.status == VerificationStatus.FULFILLED)
        o_missing = sum(1 for r in optional_results if r.status == VerificationStatus.MISSING)
        o_omission = sum(1 for r in optional_results if r.status == VerificationStatus.EXPLICIT_OMISSION)
        o_contra = sum(1 for r in optional_results if r.status == VerificationStatus.CONTRADICTED)
        o_review = sum(1 for r in optional_results if r.status == VerificationStatus.NEEDS_REVIEW)

        total_fulfilled = m_fulfilled + o_fulfilled
        total_missing = m_missing + o_missing
        total_omission = m_omission + o_omission
        total_contra = m_contra + o_contra
        total_review = m_review + o_review

        findings: List[str] = []

        # Determine parent status based on strict rules A-E
        if m_total == 0:
            # Parent has no mandatory sub-requirements
            findings.append("No mandatory sub-requirements defined for this parent point.")
            if o_total == 0:
                status = ParentComplianceStatus.NON_COMPLIANT
                findings.append("No sub-requirements associated with this parent point.")
            elif o_review > 0:
                status = ParentComplianceStatus.NEEDS_REVIEW
                findings.append(f"{o_review} optional item(s) require human review.")
            elif o_fulfilled > 0 and o_fulfilled == o_total:
                # All optional fulfilled, but no mandatory: do not classify as fully compliant
                status = ParentComplianceStatus.PARTIALLY_COMPLIANT
                findings.append("All optional sub-requirements are fulfilled, but no mandatory criteria exist.")
            else:
                status = ParentComplianceStatus.PARTIALLY_COMPLIANT
                findings.append(f"{o_fulfilled}/{o_total} optional recommendations fulfilled.")
        elif m_review > 0:
            # Rule D: Any mandatory sub-requirement NEEDS_REVIEW -> NEEDS_REVIEW
            status = ParentComplianceStatus.NEEDS_REVIEW
            findings.append(f"Requires human review: {m_review} mandatory sub-requirement(s) have unresolved evidence signals.")
        elif m_fulfilled == m_total:
            # Rule A: All mandatory sub-requirements FULFILLED -> FULLY_COMPLIANT
            status = ParentComplianceStatus.FULLY_COMPLIANT
            findings.append(f"All {m_total} mandatory sub-requirements are fulfilled.")
        elif m_fulfilled > 0:
            # Rule B: Some mandatory sub-requirements FULFILLED and others incomplete -> PARTIALLY_COMPLIANT
            status = ParentComplianceStatus.PARTIALLY_COMPLIANT
            findings.append(f"Partially compliant: {m_fulfilled} of {m_total} mandatory sub-requirements fulfilled.")
        elif m_contra == m_total:
            # Rule E: All mandatory sub-requirements are CONTRADICTED -> NON_COMPLIANT
            status = ParentComplianceStatus.NON_COMPLIANT
            findings.append(f"Non-compliant: All {m_total} mandatory sub-requirements are contradicted by target evidence.")
        elif (m_missing + m_omission) == m_total:
            # Rule C: No mandatory FULFILLED, all MISSING or EXPLICIT_OMISSION -> NON_COMPLIANT
            status = ParentComplianceStatus.NON_COMPLIANT
            if m_omission > 0 and m_missing == 0:
                findings.append(f"Non-compliant: All {m_total} mandatory sub-requirements are explicitly disclaimed/omitted.")
            elif m_missing > 0 and m_omission == 0:
                findings.append(f"Non-compliant: All {m_total} mandatory sub-requirements are missing with no evidence.")
            else:
                findings.append(f"Non-compliant: All {m_total} mandatory sub-requirements are unaddressed ({m_omission} explicitly disclaimed, {m_missing} missing).")
        else:
            # Other combinations without fulfillment and without review (e.g. mix of contradicted and missing)
            status = ParentComplianceStatus.NON_COMPLIANT
            findings.append(f"Non-compliant: 0 of {m_total} mandatory sub-requirements fulfilled ({m_contra} contradicted, {m_omission} omitted, {m_missing} missing).")

        # Document detailed sub-requirement outcomes in findings
        for r in sub_results:
            kind_str = "mandatory" if r.is_mandatory else "optional"
            best_blk = r.best_evidence.candidate.block_id if r.best_evidence else "None"
            if r.status == VerificationStatus.FULFILLED:
                findings.append(f"[{r.sub_id}] ({kind_str}) FULFILLED via block {best_blk} (confidence: {r.confidence:.2f}).")
            elif r.status == VerificationStatus.EXPLICIT_OMISSION:
                findings.append(f"[{r.sub_id}] ({kind_str}) EXPLICIT_OMISSION disclaimed in block {best_blk}.")
            elif r.status == VerificationStatus.CONTRADICTED:
                findings.append(f"[{r.sub_id}] ({kind_str}) CONTRADICTED by block {best_blk}.")
            elif r.status == VerificationStatus.NEEDS_REVIEW:
                findings.append(f"[{r.sub_id}] ({kind_str}) NEEDS_REVIEW: {r.rationale}")
            elif r.status == VerificationStatus.MISSING:
                findings.append(f"[{r.sub_id}] ({kind_str}) MISSING: No relevant candidate evidence found.")

        return ParentPointCompliance(
            point_id=point.point_id,
            section_name=point.section_name,
            topic_label=point.topic_label,
            description=point.description,
            raw_text=point.raw_text,
            status=status,
            mandatory_fulfilled=m_fulfilled,
            mandatory_total=m_total,
            optional_fulfilled=o_fulfilled,
            optional_total=o_total,
            sub_results=sub_results,
            fulfilled_count=total_fulfilled,
            missing_count=total_missing,
            explicit_omission_count=total_omission,
            contradiction_count=total_contra,
            needs_review_count=total_review,
            findings=findings,
            source_page=point.source_page,
        )

    def _determine_document_verdict(
        self,
        total_mandatory: int,
        mandatory_fulfilled: int,
        needs_review_count: int,
        contradicted_count: int,
        parent_compliances: List[ParentPointCompliance],
    ) -> DocumentComplianceVerdict:
        """
        Determines overall document-level compliance verdict:
        1. COMPLIANT:
           - All mandatory requirements are FULFILLED.
           - No mandatory contradictions.
           - No unresolved mandatory NEEDS_REVIEW items.
        2. CONDITIONALLY_COMPLIANT:
           - At least one mandatory sub-requirement has NEEDS_REVIEW.
           - Human review is required before a definitive verdict.
        3. NON_COMPLIANT:
           - Mandatory coverage is below 100%.
           - No mandatory NEEDS_REVIEW items remain unresolved.
           - Incomplete requirements include MISSING, EXPLICIT_OMISSION or CONTRADICTED.
        """
        if total_mandatory == 0:
            if needs_review_count > 0:
                return DocumentComplianceVerdict.CONDITIONALLY_COMPLIANT
            return DocumentComplianceVerdict.NON_COMPLIANT

        # Check for any unresolved mandatory NEEDS_REVIEW items
        has_mandatory_review = (
            any(
                r.is_mandatory and r.status == VerificationStatus.NEEDS_REVIEW
                for p in parent_compliances
                for r in p.sub_results
            )
            or any(p.status == ParentComplianceStatus.NEEDS_REVIEW for p in parent_compliances)
        )

        # Rule 3: Any mandatory NEEDS_REVIEW requires human review before definitive assessment
        if has_mandatory_review:
            return DocumentComplianceVerdict.CONDITIONALLY_COMPLIANT

        # Count mandatory contradictions
        has_mandatory_contradiction = any(
            r.is_mandatory and r.status == VerificationStatus.CONTRADICTED
            for p in parent_compliances
            for r in p.sub_results
        )

        # Rule 1: All mandatory requirements fulfilled with zero contradictions and zero reviews
        if mandatory_fulfilled == total_mandatory and not has_mandatory_contradiction:
            return DocumentComplianceVerdict.COMPLIANT

        # Rule 2: Mandatory coverage below 100% with no unresolved reviews -> NON_COMPLIANT
        return DocumentComplianceVerdict.NON_COMPLIANT

    def _generate_summary(
        self,
        target_name: str,
        template_title: str,
        verdict: DocumentComplianceVerdict,
        mandatory_cov_pct: Optional[float],
        optional_cov_pct: Optional[float],
        mandatory_fulfilled: int,
        total_mandatory: int,
        fulfilled_count: int,
        omission_count: int,
        contradiction_count: int,
        missing_count: int,
        needs_review_count: int,
        items_requiring_review: List[SubRequirementVerificationResult],
    ) -> str:
        """Generates a structured human-readable executive compliance summary."""
        lines = [
            f"Compliance Audit Summary for: {target_name}",
            f"Reference Standard Template: {template_title}",
            f"Overall Verdict: {verdict.value}",
            f"Mandatory Coverage: {f'{mandatory_cov_pct:.1f}% ({mandatory_fulfilled}/{total_mandatory})' if mandatory_cov_pct is not None else 'N/A'}",
            f"Optional Coverage:  {f'{optional_cov_pct:.1f}%' if optional_cov_pct is not None else 'N/A'}",
            f"Atomic Verification Breakdown: {fulfilled_count} fulfilled, {omission_count} explicit omissions, {contradiction_count} contradicted, {missing_count} missing, {needs_review_count} needs review.",
        ]

        if items_requiring_review:
            lines.append("Items Requiring Human Review:")
            for item in items_requiring_review:
                lines.append(f"  - [{item.sub_id}] ({item.parent_point_id}): {item.rationale}")
        else:
            lines.append("No items requiring immediate human review.")

        return "\n".join(lines)

