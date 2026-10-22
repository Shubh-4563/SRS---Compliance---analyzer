"""
Proof of Concept: AI-powered SRS Compliance and Quality Analyzer
Designed for Data Science / AI-ML Internship Prototype.

Demonstrates:
1. Document hierarchy extraction & requirement segmentation
2. Semantic section alignment against arbitrary reference standards (TF-IDF + Cosine Similarity)
3. Quality & ambiguity analysis (Measurability, Normative language, Atomicity, Identifiers)
4. Evidence extraction & automated compliance report generation
"""

import re
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RequirementItem:
    req_id: Optional[str]
    raw_text: str
    section_name: str
    line_number: int


@dataclass
class ComplianceFinding:
    rule_id: str
    rule_name: str
    status: str  # "COMPLIANT", "PARTIALLY_COMPLIANT", "NON_COMPLIANT"
    reason: str
    evidence: str
    confidence: float


@dataclass
class RequirementAuditResult:
    req_id: Optional[str]
    raw_text: str
    section: str
    overall_status: str
    findings: List[ComplianceFinding]


@dataclass
class DocumentSectionAudit:
    reference_section_id: str
    reference_section_name: str
    matched_document_section: Optional[str]
    similarity_score: float
    status: str  # "COMPLIANT", "NON_COMPLIANT"
    note: str


# Common vague/ambiguous terms in requirements engineering (IEEE 830 / ISO 29148 / DO-178C guidelines)
VAGUE_TERMS = {
    "speed_latency": ["quickly", "promptly", "fast", "rapidly", "real-time", "as soon as possible", "asap"],
    "subjective_quality": ["user-friendly", "intuitive", "modern", "easy to use", "easy", "robust", "simple", "efficient"],
    "approximate_metrics": ["approximately", "about", "almost", "around", "nearly", "sufficient", "adequate"],
    "unbounded_quantifiers": ["maximum possible", "optimal", "as much as possible", "minimal"]
}

# Weak or non-normative modal verbs
WEAK_MODALS = ["should", "may", "could", "might", "can", "will"]
BINDING_MODALS = ["shall", "must"]

# Common unit patterns indicating quantifiable constraints
MEASURABLE_UNIT_PATTERN = re.compile(
    r'(\b\d+(\.\d+)?\s*(ms|s|sec|seconds|hz|khz|mhz|kbps|mbps|gbps|bytes|kb|mb|gb|deg|degrees|\%|v|mv|ma|a|hours|g|knots|ft|feet|nm|rad)\b)'
    r'|(\+\/-\s*\d+(\.\d+)?)'
    r'|(\b(at least|at most|greater than|less than|exceeding|within)\s+\d+)',
    re.IGNORECASE
)


class SRSComplianceAnalyzer:
    def __init__(self, reference_path: str):
        with open(reference_path, "r", encoding="utf-8") as f:
            self.reference_data = json.load(f)
        
        self.mandatory_sections = self.reference_data.get("mandatory_sections", [])
        self.rules = {r["rule_id"]: r for r in self.reference_data.get("requirement_rules", [])}

    def parse_document(self, doc_text: str) -> Tuple[List[str], List[RequirementItem]]:
        """
        Parses an SRS document into detected sections and individual requirement items.
        """
        lines = doc_text.splitlines()
        current_section = "General / Uncategorized"
        detected_sections = []
        requirements = []

        req_id_pattern = re.compile(r'\b(REQ-[A-Z0-9_-]+)\s*[:\-]\s*(.*)', re.IGNORECASE)

        for line_num, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue

            # Header detection (Markdown headers or numbered sections)
            if line.startswith("#") or re.match(r'^\d+(\.\d+)*\s+[A-Z]', line):
                section_title = re.sub(r'^[#\s\d\.]+', '', line).strip()
                if section_title:
                    current_section = section_title
                    if section_title not in detected_sections:
                        detected_sections.append(section_title)
                continue

            # Requirement detection heuristic:
            # Check for explicit ID or normative verbs ("shall", "must", "should")
            has_id = req_id_pattern.search(line)
            has_normative = any(re.search(rf'\b{modal}\b', line, re.IGNORECASE) for modal in (BINDING_MODALS + WEAK_MODALS))

            if has_id:
                req_id = has_id.group(1).upper()
                req_text = has_id.group(2).strip()
                requirements.append(RequirementItem(
                    req_id=req_id,
                    raw_text=req_text if req_text else line,
                    section_name=current_section,
                    line_number=line_num
                ))
            elif has_normative and len(line.split()) >= 4:
                # Requirement without explicit ID
                requirements.append(RequirementItem(
                    req_id=None,
                    raw_text=line,
                    section_name=current_section,
                    line_number=line_num
                ))

        return detected_sections, requirements

    def audit_document_structure(self, detected_sections: List[str]) -> List[DocumentSectionAudit]:
        """
        Applies Semantic Text Similarity (TF-IDF vector space model)
        to match candidate document headings to the reference standard's mandatory sections.
        """
        if not detected_sections:
            return [
                DocumentSectionAudit(
                    reference_section_id=sec["id"],
                    reference_section_name=sec["name"],
                    matched_document_section=None,
                    similarity_score=0.0,
                    status="NON_COMPLIANT",
                    note="No sections identified in document."
                ) for sec in self.mandatory_sections
            ]

        results = []
        for ref_sec in self.mandatory_sections:
            target_text = f"{ref_sec['name']} {ref_sec['description']}"
            corpus = [target_text] + detected_sections

            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(corpus)
            sim_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            best_idx = int(np.argmax(sim_scores))
            best_score = float(sim_scores[best_idx])
            best_match = detected_sections[best_idx]

            # Matching threshold
            if best_score >= 0.25:
                results.append(DocumentSectionAudit(
                    reference_section_id=ref_sec["id"],
                    reference_section_name=ref_sec["name"],
                    matched_document_section=best_match,
                    similarity_score=round(best_score, 3),
                    status="COMPLIANT",
                    note=f"Matched to document section '{best_match}' with similarity {best_score:.2f}."
                ))
            else:
                results.append(DocumentSectionAudit(
                    reference_section_id=ref_sec["id"],
                    reference_section_name=ref_sec["name"],
                    matched_document_section=None,
                    similarity_score=round(best_score, 3),
                    status="NON_COMPLIANT",
                    note=f"Missing mandatory section. Highest similarity match was '{best_match}' ({best_score:.2f}) which is below threshold (0.25)."
                ))

        return results

    def analyze_requirement(self, req: RequirementItem) -> RequirementAuditResult:
        """
        Evaluates a single requirement statement against individual quality and compliance rules.
        """
        findings: List[ComplianceFinding] = []
        text = req.raw_text.strip()
        lower_text = text.lower()

        # 1. Identifier Check (RULE-IDENTIFIER)
        if "RULE-IDENTIFIER" in self.rules:
            rule = self.rules["RULE-IDENTIFIER"]
            if req.req_id:
                findings.append(ComplianceFinding(
                    rule_id=rule["rule_id"],
                    rule_name=rule["rule_name"],
                    status="COMPLIANT",
                    reason="Requirement contains a structured unique identifier.",
                    evidence=f"ID: {req.req_id}",
                    confidence=1.0
                ))
            else:
                findings.append(ComplianceFinding(
                    rule_id=rule["rule_id"],
                    rule_name=rule["rule_name"],
                    status="NON_COMPLIANT",
                    reason="Requirement is missing a designated unique alphanumeric identifier (e.g. REQ-XXX-000).",
                    evidence=text[:45] + "...",
                    confidence=0.95
                ))

        # 2. Normative Verb Check (RULE-NORMATIVE)
        if "RULE-NORMATIVE" in self.rules:
            rule = self.rules["RULE-NORMATIVE"]
            found_weak = [modal for modal in WEAK_MODALS if re.search(rf'\b{modal}\b', lower_text)]
            found_binding = [modal for modal in BINDING_MODALS if re.search(rf'\b{modal}\b', lower_text)]

            if found_weak:
                findings.append(ComplianceFinding(
                    rule_id=rule["rule_id"],
                    rule_name=rule["rule_name"],
                    status="NON_COMPLIANT",
                    reason=f"Uses non-binding or ambiguous modal verb: {', '.join(found_weak)}. Standard requires 'shall' or 'must'.",
                    evidence=f"Clause contains: '{found_weak[0]}'",
                    confidence=0.95
                ))
            elif found_binding:
                findings.append(ComplianceFinding(
                    rule_id=rule["rule_id"],
                    rule_name=rule["rule_name"],
                    status="COMPLIANT",
                    reason=f"Properly uses mandatory normative modal auxiliary verb ('{found_binding[0]}').",
                    evidence=f"Modal verb: '{found_binding[0]}'",
                    confidence=1.0
                ))
            else:
                findings.append(ComplianceFinding(
                    rule_id=rule["rule_id"],
                    rule_name=rule["rule_name"],
                    status="PARTIALLY_COMPLIANT",
                    reason="No standard normative modal auxiliary verb ('shall' or 'must') detected in statement.",
                    evidence=text[:40] + "...",
                    confidence=0.85
                ))

        # 3. Measurability & Vagueness Check (RULE-VERIFIABLE)
        if "RULE-VERIFIABLE" in self.rules:
            rule = self.rules["RULE-VERIFIABLE"]
            detected_vague_words = []
            for category, term_list in VAGUE_TERMS.items():
                for term in term_list:
                    match = re.search(rf'\b{re.escape(term)}\b', lower_text)
                    if match:
                        detected_vague_words.append((term, category))

            has_quantifiable_metric = bool(MEASURABLE_UNIT_PATTERN.search(text))

            if detected_vague_words:
                vague_terms_str = ", ".join([f"'{term}'" for term, cat in detected_vague_words])
                findings.append(ComplianceFinding(
                    rule_id=rule["rule_id"],
                    rule_name=rule["rule_name"],
                    status="NON_COMPLIANT",
                    reason=f"Contains subjective or unmeasurable qualitative descriptor(s): {vague_terms_str}.",
                    evidence=f"Found unmeasurable term(s): {vague_terms_str}",
                    confidence=0.90
                ))
            elif has_quantifiable_metric:
                metric_match = MEASURABLE_UNIT_PATTERN.search(text).group(0)
                findings.append(ComplianceFinding(
                    rule_id=rule["rule_id"],
                    rule_name=rule["rule_name"],
                    status="COMPLIANT",
                    reason="Specification contains verifiable quantitative boundaries / numerical constraints.",
                    evidence=f"Quantifiable constraint detected: '{metric_match}'",
                    confidence=0.92
                ))
            else:
                findings.append(ComplianceFinding(
                    rule_id=rule["rule_id"],
                    rule_name=rule["rule_name"],
                    status="PARTIALLY_COMPLIANT",
                    reason="No quantitative metrics or numeric boundaries found. Requirement may rely on qualitative verification.",
                    evidence=text[:50] + "...",
                    confidence=0.75
                ))

        # 4. Atomicity Check (RULE-ATOMICITY)
        if "RULE-ATOMICITY" in self.rules:
            rule = self.rules["RULE-ATOMICITY"]
            # Look for multiple 'shall' statements or compound conjunction connectors
            shall_count = len(re.findall(r'\bshall\b', lower_text))
            has_compound_conj = bool(re.search(r'\b(and\s+the\s+unit\s+shall|and\s+also|additionally|as\s+well\s+as)\b', lower_text))

            if shall_count > 1 or has_compound_conj:
                findings.append(ComplianceFinding(
                    rule_id=rule["rule_id"],
                    rule_name=rule["rule_name"],
                    status="NON_COMPLIANT",
                    reason=f"Violates atomicity: bundles multiple independent obligations (detected {shall_count} 'shall' clauses or compound coordinating conjunctions).",
                    evidence=f"Multiple action conjunctions found in statement.",
                    confidence=0.88
                ))
            else:
                findings.append(ComplianceFinding(
                    rule_id=rule["rule_id"],
                    rule_name=rule["rule_name"],
                    status="COMPLIANT",
                    reason="Requirement expresses a single discrete obligation.",
                    evidence="Atomic single-obligation sentence structure.",
                    confidence=0.90
                ))

        # Determine overall requirement status
        statuses = [f.status for f in findings]
        if "NON_COMPLIANT" in statuses:
            overall_status = "NON_COMPLIANT"
        elif "PARTIALLY_COMPLIANT" in statuses:
            overall_status = "PARTIALLY_COMPLIANT"
        else:
            overall_status = "COMPLIANT"

        return RequirementAuditResult(
            req_id=req.req_id,
            raw_text=req.raw_text,
            section=req.section_name,
            overall_status=overall_status,
            findings=findings
        )

    def run_full_audit(self, doc_text: str) -> Dict[str, Any]:
        """
        Executes full audit pipeline on the input document against the reference standard.
        """
        sections, requirements = self.parse_document(doc_text)
        structural_audit = self.audit_document_structure(sections)
        req_audits = [self.analyze_requirement(req) for req in requirements]

        # Calculate high-level metrics
        total_reqs = len(req_audits)
        compliant_reqs = sum(1 for r in req_audits if r.overall_status == "COMPLIANT")
        partial_reqs = sum(1 for r in req_audits if r.overall_status == "PARTIALLY_COMPLIANT")
        non_compliant_reqs = sum(1 for r in req_audits if r.overall_status == "NON_COMPLIANT")

        total_sections = len(structural_audit)
        compliant_sections = sum(1 for s in structural_audit if s.status == "COMPLIANT")

        overall_compliance_score = round(
            ((compliant_reqs + 0.5 * partial_reqs) / total_reqs * 0.7 + (compliant_sections / total_sections) * 0.3) * 100, 1
        ) if total_reqs and total_sections else 0.0

        return {
            "summary": {
                "overall_compliance_score": f"{overall_compliance_score}%",
                "total_requirements_analyzed": total_reqs,
                "compliant_requirements": compliant_reqs,
                "partially_compliant_requirements": partial_reqs,
                "non_compliant_requirements": non_compliant_reqs,
                "mandatory_sections_present": f"{compliant_sections}/{total_sections}"
            },
            "structural_audit": [asdict(s) for s in structural_audit],
            "requirements_audit": [asdict(r) for r in req_audits]
        }


def format_markdown_report(audit_data: Dict[str, Any]) -> str:
    summary = audit_data["summary"]
    md = []
    md.append("# SRS Compliance & Quality Verification Report\n")
    md.append("## 1. Executive Summary")
    md.append(f"- **Overall Compliance Score**: `{summary['overall_compliance_score']}`")
    md.append(f"- **Total Requirements Analyzed**: {summary['total_requirements_analyzed']}")
    md.append(f"- **Compliant**: {summary['compliant_requirements']} | **Partially Compliant**: {summary['partially_compliant_requirements']} | **Non-Compliant**: {summary['non_compliant_requirements']}")
    md.append(f"- **Mandatory Sections Coverage**: {summary['mandatory_sections_present']}\n")

    md.append("## 2. Structural & Section Coverage Analysis")
    md.append("| Reference Section | Matched Document Section | Similarity | Status | Observations |")
    md.append("| :--- | :--- | :---: | :---: | :--- |")
    for sec in audit_data["structural_audit"]:
        matched = sec["matched_document_section"] or "*None (Missing)*"
        status_badge = "✅ COMPLIANT" if sec["status"] == "COMPLIANT" else "❌ NON-COMPLIANT"
        md.append(f"| **{sec['reference_section_name']}** | {matched} | {sec['similarity_score']} | {status_badge} | {sec['note']} |")
    md.append("\n")

    md.append("## 3. Detailed Requirement Quality & Compliance Findings")
    for idx, req in enumerate(audit_data["requirements_audit"], start=1):
        status_emoji = {"COMPLIANT": "✅", "PARTIALLY_COMPLIANT": "⚠️", "NON_COMPLIANT": "❌"}[req["overall_status"]]
        req_id_display = req["req_id"] if req["req_id"] else "*[Missing Identifier]*"
        md.append(f"### {idx}. {status_emoji} [{req_id_display}] - {req['overall_status']}")
        md.append(f"**Section**: *{req['section']}*  ")
        md.append(f"> \"{req['raw_text']}\"\n")
        md.append("**Findings & Evidence:**")
        for finding in req["findings"]:
            sub_badge = {"COMPLIANT": "✓", "PARTIALLY_COMPLIANT": "!", "NON_COMPLIANT": "✗"}[finding["status"]]
            md.append(f"- `[{sub_badge}]` **{finding['rule_name']}** ({finding['status']}): {finding['reason']}")
            md.append(f"  - *Evidence*: `{finding['evidence']}` (Confidence: {finding['confidence'] * 100:.0f}%)")
        md.append("")

    return "\n".join(md)
