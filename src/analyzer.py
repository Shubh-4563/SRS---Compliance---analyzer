"""
SRS Compliance Analyzer - Version 0.3
======================================
Pipeline Architecture:
    Reference Standard + SRS Document
                  ↓
    Document Parsing (Sections + Requirements)
                  ↓
    Semantic Section Alignment (sentence-transformers / all-MiniLM-L6-v2)
                  ↓
    Semantic Rule Matching (sentence-transformers / all-MiniLM-L6-v2: Requirement ↔ Rules)
                  ↓
    Rule-Based Compliance Checks (R1 - R4)
                  ↓
          Compliance Report

Core Rules (Unchanged from Version 0.1):
- R1: Unique Requirement ID
- R2: Clear Behaviour (Binding 'shall' normative action)
- R3: Measurable Criteria (Quantifiable / verifiable criteria)
- R4: Avoid Vague Language (Detect terms like 'quickly', 'fast', 'easy', 'user-friendly', etc.)

Version 0.2 Capability:
- Semantic Section Matching using dense sentence embeddings (all-MiniLM-L6-v2).

Version 0.3 Capability:
- Semantic Rule Matching between individual SRS requirements and reference standard rules (all-MiniLM-L6-v2).
"""

import os
import re
import json
import warnings
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any, Set, Tuple, Union


@dataclass
class Requirement:
    """Represents an extracted requirement item from an SRS document."""
    req_id: Optional[str]
    statement: str
    raw_line: str
    line_number: int
    section_name: Optional[str] = None


@dataclass
class ReferenceSection:
    """A required or recommended section defined in a reference standard."""
    section_id: str
    name: str
    description: str = ""


@dataclass
class ReferenceRule:
    """A compliance rule defined in the reference standard."""
    rule_id: str
    name: str
    description: str = ""

    @property
    def id(self) -> str:
        return self.rule_id


@dataclass
class DocumentSection:
    """A section heading identified within an SRS document."""
    title: str
    clean_title: str
    line_number: int
    raw_header: str


@dataclass
class SectionMatchResult:
    """Outcome of semantic matching between a standard section and document sections."""
    reference_section_id: str
    reference_section_name: str
    reference_section_description: str
    matched_document_section: Optional[str]
    similarity_score: float
    is_matched: bool
    notes: str
    candidate_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class RuleMatchResult:
    """Outcome of semantic matching between an individual requirement and reference rules."""
    best_rule_id: str
    best_rule_name: str
    similarity_score: float
    candidate_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def matched_rule_id(self) -> str:
        return self.best_rule_id

    @property
    def matched_rule_name(self) -> str:
        return self.best_rule_name

    @property
    def rule_scores(self) -> Dict[str, float]:
        return self.candidate_scores


@dataclass
class RuleCheckResult:
    """Outcome of a single compliance rule check (R1, R2, R3, R4)."""
    rule_id: str
    rule_name: str
    is_compliant: bool
    reason: str
    evidence: Optional[str] = None


@dataclass
class RequirementAudit:
    """Aggregated audit results for a single requirement."""
    req_id: Optional[str]
    statement: str
    line_number: int
    is_compliant: bool
    primary_reason: Optional[str]
    checks: List[RuleCheckResult] = field(default_factory=list)
    section_name: Optional[str] = None
    matched_rule_id: Optional[str] = None
    matched_rule_name: Optional[str] = None
    rule_similarity_score: Optional[float] = None
    rule_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class ComplianceReport:
    """Complete summary and collection of requirement audits and section alignment."""
    total_requirements: int
    compliant_count: int
    non_compliant_count: int
    compliance_percentage: float
    audits: List[RequirementAudit]
    section_matches: List[SectionMatchResult] = field(default_factory=list)
    sections_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemplatePoint:
    """A single required section or content point from a reference template."""
    point_id: str
    section_name: str
    text: str
    topic_label: str = ""
    description: str = ""
    line_number: int = 0
    raw_line: str = ""


@dataclass
class ReferenceTemplate:
    """A collection of required sections and points extracted from a reference template."""
    title: str
    points: List[TemplatePoint] = field(default_factory=list)
    sections: List[str] = field(default_factory=list)


@dataclass
class PointMatchResult:
    """Outcome of comparing a single required template point against the target document."""
    point_id: str
    section_name: str
    reference_text: str
    status: str                         # "COVERED", "PARTIAL", or "MISSING"
    similarity_score: float             # Angular cosine similarity in [-1.0, 1.0]
    matched_text: Optional[str] = None  # Closest matching text evidence found in target
    notes: str = ""
    winning_perspective: str = ""       # "canonical", "full", or "topic"
    evidence_tier: str = "NONE"         # "STRONG", "PARTIAL", "WEAK", "NONE", or "EXPLICIT_ABSENCE"
    is_explicit_absence: bool = False   # True if target explicitly confirms absence/omission
    matched_page: Optional[int] = None  # Page number of matched evidence snippet


@dataclass
class TemplateComplianceReport:
    """Compliance verification report comparing a reference template against a target document."""
    template_title: str
    target_name: str
    target_page: Optional[int]
    total_pages: int
    covered_threshold: float
    partial_threshold: float
    min_evidence_threshold: float
    total_points: int
    covered_count: int
    partial_count: int
    missing_count: int
    coverage_percentage: float
    results: List[PointMatchResult]
    sections_summary: Dict[str, Any] = field(default_factory=dict)


class ReferenceStandard:
    """Loads and encapsulates compliance rules, mandatory sections, and parameters."""

    DEFAULT_VAGUE_WORDS = ["quickly", "fast", "easy", "adequate", "user-friendly"]
    DEFAULT_MANDATORY_SECTIONS = [
        {
            "id": "SEC-1",
            "name": "Scope and System Overview",
            "description": "High-level description of system purpose, operational scope, and subsystem boundaries."
        },
        {
            "id": "SEC-2",
            "name": "Functional Requirements",
            "description": "Functional capabilities, actions, transformations, and system behavioral requirements."
        },
        {
            "id": "SEC-3",
            "name": "Interface Requirements",
            "description": "Hardware, software, bus, communications protocol, and user interface specifications."
        },
        {
            "id": "SEC-4",
            "name": "Safety and Failure Handling",
            "description": "Fault detection, exception handling, containment modes, and fail-safe recovery procedures."
        }
    ]
    DEFAULT_RULES = [
        {
            "id": "R1",
            "name": "Unique Requirement ID",
            "description": "Every requirement must have a unique identifier (e.g., REQ-001)."
        },
        {
            "id": "R2",
            "name": "Clear Behaviour",
            "description": "The requirement must describe what the system shall do using clear binding normative language.",
            "binding_modal": "shall"
        },
        {
            "id": "R3",
            "name": "Measurable Criteria",
            "description": "The requirement should contain measurable/verifiable criteria where applicable."
        },
        {
            "id": "R4",
            "name": "Avoid Vague Language",
            "description": "Detect vague words that introduce ambiguity into specifications.",
            "vague_words": ["quickly", "fast", "easy", "adequate", "user-friendly"]
        }
    ]

    def __init__(self, filepath: Optional[str] = None):
        self.standard_name = "SRS Quality and Compliance Standard"
        self.version = "0.3"
        self.vague_words = list(self.DEFAULT_VAGUE_WORDS)
        self.binding_modal = "shall"
        self.sections: List[ReferenceSection] = [
            ReferenceSection(
                section_id=s["id"],
                name=s["name"],
                description=s.get("description", "")
            )
            for s in self.DEFAULT_MANDATORY_SECTIONS
        ]
        self.rules: List[ReferenceRule] = [
            ReferenceRule(
                rule_id=r["id"],
                name=r["name"],
                description=r.get("description", "")
            )
            for r in self.DEFAULT_RULES
        ]

        if filepath and os.path.exists(filepath):
            if filepath.lower().endswith(".pdf"):
                self._load_from_pdf(filepath)
            else:
                self._load_from_json(filepath)

    def _load_from_pdf(self, filepath: str):
        from src.pdf_extractor import PDFExtractor
        extractor = PDFExtractor()
        std = extractor.extract_reference_standard(filepath)
        self.standard_name = std.standard_name
        self.version = std.version
        self.vague_words = std.vague_words
        self.binding_modal = std.binding_modal
        self.sections = std.sections
        self.rules = std.rules

    def _load_from_json(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.standard_name = data.get("standard_name", self.standard_name)
        self.version = data.get("version", self.version)

        raw_sections = data.get("mandatory_sections") or data.get("sections")
        if raw_sections:
            self.sections = [
                ReferenceSection(
                    section_id=s.get("id", f"SEC-{idx}"),
                    name=s.get("name", ""),
                    description=s.get("description", "")
                )
                for idx, s in enumerate(raw_sections, start=1)
                if s.get("name")
            ]

        raw_rules = data.get("rules") or data.get("requirement_rules")
        if raw_rules:
            self.rules = [
                ReferenceRule(
                    rule_id=r.get("id") or r.get("rule_id", f"R{idx}"),
                    name=r.get("name") or r.get("rule_name", ""),
                    description=r.get("description", "")
                )
                for idx, r in enumerate(raw_rules, start=1)
                if r.get("name") or r.get("rule_name")
            ]

        for rule in data.get("rules", []):
            if rule.get("id") == "R4" and "vague_words" in rule:
                self.vague_words = rule["vague_words"]
            if rule.get("id") == "R2" and "binding_modal" in rule:
                self.binding_modal = rule["binding_modal"]


def _import_sentence_transformers():
    """
    Validates and imports sentence_transformers with clear troubleshooting guidance.
    Ensures that when running in any virtualenv/conda/system environment, missing
    dependencies produce an actionable, user-friendly diagnostic message.
    """
    try:
        from sentence_transformers import SentenceTransformer, util
        return SentenceTransformer, util
    except ImportError as exc:
        import sys
        raise ImportError(
            f"\n{'=' * 72}\n"
            f"[Dependency Error] The 'sentence-transformers' package is not installed in:\n"
            f"  Python Interpreter: {sys.executable}\n\n"
            f"To resolve this, please install dependencies via:\n"
            f"  {sys.executable} -m pip install -r requirements.txt\n"
            f"  or: {sys.executable} -m pip install sentence-transformers torch\n"
            f"{'=' * 72}\n"
        ) from exc


class SemanticSectionMatcher:
    """
    Semantic embedding engine utilizing sentence-transformers (all-MiniLM-L6-v2)
    for dense vector cosine similarity matching:
    1. Section-to-section semantic alignment (Version 0.2)
    2. Requirement-to-rule semantic matching (Version 0.3)
    """
    DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
    DEFAULT_THRESHOLD = 0.55

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        threshold: float = DEFAULT_THRESHOLD,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy loads SentenceTransformer to ensure fast startup when not used."""
        if self._model is None:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            SentenceTransformer, _ = _import_sentence_transformers()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        local_files_only=True
                    )
                except Exception:
                    self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Computes cosine similarity between two sentences/phrases."""
        _, util = _import_sentence_transformers()
        embeddings = self.model.encode(
            [text_a, text_b],
            convert_to_tensor=True,
            show_progress_bar=False
        )
        sim = util.cos_sim(embeddings[0], embeddings[1])
        return round(float(sim[0][0]), 4)

    def match_sections(
        self,
        ref_sections: List[ReferenceSection],
        doc_sections: List[DocumentSection],
        threshold: Optional[float] = None
    ) -> List[SectionMatchResult]:
        """
        Calculates dense embedding cosine similarities between reference standard
        sections and extracted document sections using all-MiniLM-L6-v2.
        """
        match_threshold = threshold if threshold is not None else self.threshold

        if not ref_sections:
            return []

        if not doc_sections:
            return [
                SectionMatchResult(
                    reference_section_id=ref.section_id,
                    reference_section_name=ref.name,
                    reference_section_description=ref.description,
                    matched_document_section=None,
                    similarity_score=0.0,
                    is_matched=False,
                    notes="Document contains no section headers. Section alignment skipped.",
                    candidate_scores={}
                )
                for ref in ref_sections
            ]

        _, util = _import_sentence_transformers()
        import torch

        # Candidate document titles (clean titles for semantic purity)
        doc_clean_titles = [d.clean_title for d in doc_sections]
        doc_embs = self.model.encode(
            doc_clean_titles,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        results: List[SectionMatchResult] = []

        for ref in ref_sections:
            ref_title_emb = self.model.encode(
                ref.name,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            sims_title = util.cos_sim(ref_title_emb, doc_embs)[0]

            if ref.description:
                ref_rich = f"{ref.name}: {ref.description}"
                ref_rich_emb = self.model.encode(
                    ref_rich,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                sims_rich = util.cos_sim(ref_rich_emb, doc_embs)[0]
                sims = torch.maximum(sims_title, sims_rich)
            else:
                sims = sims_title

            candidate_scores: Dict[str, float] = {}
            for idx, doc_sec in enumerate(doc_sections):
                candidate_scores[doc_sec.title] = round(float(sims[idx]), 4)

            best_idx = int(sims.argmax())
            best_score = round(float(sims[best_idx]), 4)
            best_doc = doc_sections[best_idx]

            if best_score >= match_threshold:
                is_matched = True
                notes = f"Matched to document section '{best_doc.title}' (similarity: {best_score:.4f} >= {match_threshold})."
                matched_name = best_doc.title
            else:
                is_matched = False
                notes = f"Missing mandatory section. Highest similarity candidate was '{best_doc.title}' ({best_score:.4f} < {match_threshold})."
                matched_name = None

            results.append(
                SectionMatchResult(
                    reference_section_id=ref.section_id,
                    reference_section_name=ref.name,
                    reference_section_description=ref.description,
                    matched_document_section=matched_name,
                    similarity_score=best_score,
                    is_matched=is_matched,
                    notes=notes,
                    candidate_scores=candidate_scores
                )
            )

        return results

    def match_requirement_to_rules(
        self,
        requirement_text: str,
        rules: List[ReferenceRule]
    ) -> RuleMatchResult:
        """
        Version 0.3: Computes cosine similarity between an individual requirement
        statement and reference compliance rules using all-MiniLM-L6-v2 embeddings.
        Returns the best matching rule and similarity score.
        """
        if not rules:
            return RuleMatchResult(
                best_rule_id="NONE",
                best_rule_name="No rules defined",
                similarity_score=0.0,
                candidate_scores={}
            )

        _, util = _import_sentence_transformers()

        rule_texts = [f"{r.name}: {r.description}" for r in rules]
        rule_embs = self.model.encode(
            rule_texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        req_emb = self.model.encode(
            requirement_text,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        sims = util.cos_sim(req_emb, rule_embs)[0]
        candidate_scores: Dict[str, float] = {}
        for idx, rule in enumerate(rules):
            candidate_scores[rule.rule_id] = round(float(sims[idx]), 4)

        best_idx = int(sims.argmax())
        best_rule = rules[best_idx]
        best_score = round(float(sims[best_idx]), 4)

        return RuleMatchResult(
            best_rule_id=best_rule.rule_id,
            best_rule_name=best_rule.name,
            similarity_score=best_score,
            candidate_scores=candidate_scores
        )

    def match_requirements_to_rules(
        self,
        requirements: Union[List[Requirement], List[str]],
        rules: List[ReferenceRule]
    ) -> List[RuleMatchResult]:
        """Matches a batch of requirements (either Requirement objects or strings) against reference rules."""
        results = []
        for req in requirements:
            text = req.statement if hasattr(req, "statement") else str(req)
            results.append(self.match_requirement_to_rules(text, rules))
        return results


# Aliases for semantic matcher
SemanticMatcher = SemanticSectionMatcher
SemanticRuleMatcher = SemanticSectionMatcher


class RequirementExtractor:
    """
    Extracts individual requirement items and structural sections from raw SRS text.
    Handles structured tags (e.g. 'REQ-001: ...', 'REQ-FDM-001 - ...')
    as well as plain requirement statements without IDs.
    """

    ID_PATTERN = re.compile(
        r'^\s*(?:[-*]\s*)?((?:REQ|SRS|SYS)[-_][A-Za-z0-9_-]+)\s*[:\-]\s*(.*)$',
        re.IGNORECASE
    )

    NORMATIVE_HEURISTIC = re.compile(r'\b(shall|must|should)\b', re.IGNORECASE)

    def extract_sections_from_text(self, text: str) -> List[DocumentSection]:
        """Extracts candidate structural section headings from SRS text/markdown."""
        sections: List[DocumentSection] = []
        lines = text.splitlines()

        for line_num, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue

            # Markdown header detection: # Header, ## Subsystem, ### 1. Scope
            if line.startswith("#"):
                clean = re.sub(r'^#+\s*', '', line).strip()
                clean_title = re.sub(
                    r'^(?:\d+(?:\.\d+)*[\.\:\-\)]?|section\s+\d+[\.\:\-\)]?)\s*',
                    '',
                    clean,
                    flags=re.IGNORECASE
                ).strip()
                title = clean if clean else line
                sections.append(
                    DocumentSection(
                        title=title,
                        clean_title=clean_title if clean_title else title,
                        line_number=line_num,
                        raw_header=line
                    )
                )
            # Plain numbered header detection: e.g. "1. Scope and System Overview"
            elif (
                re.match(r'^(?:\d+(?:\.\d+)*|Section\s+\d+)[\.\:\-\s]+[A-Z]', line)
                and not self.ID_PATTERN.match(line)
                and not self.NORMATIVE_HEURISTIC.search(line)
            ):
                clean_title = re.sub(
                    r'^(?:\d+(?:\.\d+)*[\.\:\-\)]?|Section\s+\d+[\.\:\-\)]?)\s*',
                    '',
                    line,
                    flags=re.IGNORECASE
                ).strip()
                sections.append(
                    DocumentSection(
                        title=line,
                        clean_title=clean_title if clean_title else line,
                        line_number=line_num,
                        raw_header=line
                    )
                )

        return sections

    def extract_sections_from_file(self, filepath: str) -> List[DocumentSection]:
        """Extracts candidate structural sections from a text, markdown, or PDF file."""
        if filepath.lower().endswith(".pdf"):
            from src.pdf_extractor import PDFExtractor
            extractor = PDFExtractor()
            sections, _ = extractor.extract_srs_document(filepath)
            return sections
        with open(filepath, "r", encoding="utf-8") as f:
            return self.extract_sections_from_text(f.read())

    def extract_from_text(self, text: str) -> List[Requirement]:
        """
        Extracts individual requirements while tracking their parent section.
        Maintains 100% backward compatibility with Version 0.1 callers.
        """
        requirements: List[Requirement] = []
        lines = text.splitlines()
        current_section: Optional[str] = None

        for line_num, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue

            # Track section headers
            if line.startswith("#"):
                clean = re.sub(r'^#+\s*', '', line).strip()
                current_section = re.sub(
                    r'^(?:\d+(?:\.\d+)*[\.\:\-\)]?|section\s+\d+[\.\:\-\)]?)\s*',
                    '',
                    clean,
                    flags=re.IGNORECASE
                ).strip()
                continue
            elif (
                re.match(r'^(?:\d+(?:\.\d+)*|Section\s+\d+)[\.\:\-\s]+[A-Z]', line)
                and not self.ID_PATTERN.match(line)
                and not self.NORMATIVE_HEURISTIC.search(line)
            ):
                current_section = re.sub(
                    r'^(?:\d+(?:\.\d+)*[\.\:\-\)]?|Section\s+\d+[\.\:\-\)]?)\s*',
                    '',
                    line,
                    flags=re.IGNORECASE
                ).strip()
                continue

            match = self.ID_PATTERN.match(line)
            if match:
                req_id = match.group(1).upper()
                req_text = match.group(2).strip()
                requirements.append(
                    Requirement(
                        req_id=req_id,
                        statement=req_text if req_text else line,
                        raw_line=line,
                        line_number=line_num,
                        section_name=current_section
                    )
                )
            elif self.NORMATIVE_HEURISTIC.search(line):
                # Statement with normative modal but missing an ID
                requirements.append(
                    Requirement(
                        req_id=None,
                        statement=line,
                        raw_line=line,
                        line_number=line_num,
                        section_name=current_section
                    )
                )

        return requirements

    def extract_from_file(self, filepath: str) -> List[Requirement]:
        """Extracts individual requirements from a text, markdown, or PDF file."""
        if filepath.lower().endswith(".pdf"):
            from src.pdf_extractor import PDFExtractor
            extractor = PDFExtractor()
            _, reqs = extractor.extract_srs_document(filepath)
            return reqs
        with open(filepath, "r", encoding="utf-8") as f:
            return self.extract_from_text(f.read())


class ComplianceAnalyzer:
    """
    Executes compliance checks on extracted requirements:
    1. Version 0.1 Rules (R1 - R4) - deterministic checks
    2. Version 0.2 Semantic section alignment (sentence-transformers)
    3. Version 0.3 Semantic rule matching (sentence-transformers)
    """

    MEASURABLE_PATTERNS = re.compile(
        r'(\b\d+(?:\.\d+)?\s*(?:seconds?|secs?|ms|hz|khz|mhz|kbps|mbps|gbps|bytes|kb|mb|gb|%|percent|degrees?|deg|v|mv|ma|a|hours?|knots|ft|feet|nm)\b)'
        r'|(\b(?:within|at least|at most|less than|greater than|exceeding|between|\+\/-)\s*\d+(?:\.\d+)?)',
        re.IGNORECASE
    )

    WEAK_MODALS = ["should", "may", "could", "might", "can", "will"]

    def __init__(
        self,
        standard: Optional[ReferenceStandard] = None,
        section_matcher: Optional[SemanticSectionMatcher] = None,
        enable_semantic_matching: bool = True,
        similarity_threshold: float = 0.55
    ):
        self.standard = standard or ReferenceStandard()
        self.seen_ids: Set[str] = set()
        self.enable_semantic_matching = enable_semantic_matching
        self.similarity_threshold = similarity_threshold

        if section_matcher is not None:
            self.section_matcher = section_matcher
        elif enable_semantic_matching:
            self.section_matcher = SemanticSectionMatcher(threshold=similarity_threshold)
        else:
            self.section_matcher = None

    def reset_state(self):
        """Clears tracked IDs for a fresh document audit."""
        self.seen_ids.clear()

    def check_r1_id(self, req: Requirement) -> RuleCheckResult:
        """R1: Every requirement must have a unique identifier."""
        if not req.req_id:
            return RuleCheckResult(
                rule_id="R1",
                rule_name="Unique Requirement ID",
                is_compliant=False,
                reason="Requirement is missing a designated unique identifier (e.g., REQ-001).",
                evidence=None
            )

        if req.req_id in self.seen_ids:
            return RuleCheckResult(
                rule_id="R1",
                rule_name="Unique Requirement ID",
                is_compliant=False,
                reason=f"Duplicate requirement identifier '{req.req_id}' detected.",
                evidence=f"ID: {req.req_id}"
            )

        self.seen_ids.add(req.req_id)
        return RuleCheckResult(
            rule_id="R1",
            rule_name="Unique Requirement ID",
            is_compliant=True,
            reason=f"Requirement contains unique identifier '{req.req_id}'.",
            evidence=f"ID: {req.req_id}"
        )

    def check_r2_behavior(self, req: Requirement) -> RuleCheckResult:
        """R2: The requirement should describe what the system shall do."""
        statement = req.statement
        lower_stmt = statement.lower()
        binding = self.standard.binding_modal.lower()

        # Check for standard binding modal ('shall')
        if re.search(rf'\b{re.escape(binding)}\b', lower_stmt):
            return RuleCheckResult(
                rule_id="R2",
                rule_name="Clear Behaviour",
                is_compliant=True,
                reason=f"Properly specifies mandatory behavior using '{binding}'.",
                evidence=f"Modal verb: '{binding}'"
            )

        # Check for non-binding or ambiguous modals
        weak_found = [m for m in self.WEAK_MODALS if re.search(rf'\b{m}\b', lower_stmt)]
        if weak_found:
            return RuleCheckResult(
                rule_id="R2",
                rule_name="Clear Behaviour",
                is_compliant=False,
                reason=f"Uses non-binding modal verb '{weak_found[0]}'. Standard requires '{binding}'.",
                evidence=f"Modal: '{weak_found[0]}'"
            )

        return RuleCheckResult(
            rule_id="R2",
            rule_name="Clear Behaviour",
            is_compliant=False,
            reason=f"Missing explicit mandatory behavior specification ('{binding}').",
            evidence=statement[:40] + "..." if len(statement) > 40 else statement
        )

    def check_r3_measurable(self, req: Requirement) -> RuleCheckResult:
        """
        R3: The requirement should contain measurable/verifiable criteria where applicable.
        For Version 0.1, 0.2 & 0.3:
          - Detects quantifiable metrics (e.g., 'within 0.5 seconds').
          - Discrete functional actions without unmeasurable ambiguities are compliant.
        """
        match = self.MEASURABLE_PATTERNS.search(req.statement)
        if match:
            metric = match.group(0).strip()
            return RuleCheckResult(
                rule_id="R3",
                rule_name="Measurable Criteria",
                is_compliant=True,
                reason=f"Specification contains verifiable quantitative metric ('{metric}').",
                evidence=metric
            )

        return RuleCheckResult(
            rule_id="R3",
            rule_name="Measurable Criteria",
            is_compliant=True,
            reason="Specifies verifiable functional behavior (compliant for initial prototype).",
            evidence=None
        )

    def check_r4_vague(self, req: Requirement) -> RuleCheckResult:
        """R4: Detect vague words such as: quickly, fast, easy, adequate, user-friendly."""
        lower_stmt = req.statement.lower()
        detected: List[str] = []

        for word in self.standard.vague_words:
            pattern = rf'(?<!\w){re.escape(word.lower())}(?!\w)'
            if re.search(pattern, lower_stmt):
                detected.append(word)

        if detected:
            if len(detected) == 1:
                vague_desc = f'"{detected[0]}" is vague.'
            else:
                formatted = ", ".join(f'"{w}"' for w in detected)
                vague_desc = f"{formatted} are vague."

            return RuleCheckResult(
                rule_id="R4",
                rule_name="Avoid Vague Language",
                is_compliant=False,
                reason=vague_desc,
                evidence=f"Detected: {', '.join(detected)}"
            )

        return RuleCheckResult(
            rule_id="R4",
            rule_name="Avoid Vague Language",
            is_compliant=True,
            reason="No vague or subjective language detected.",
            evidence=None
        )

    def audit_requirement(self, req: Requirement) -> RequirementAudit:
        """Audits an individual requirement across R1-R4 rules and matches against reference rules."""
        checks = [
            self.check_r1_id(req),
            self.check_r2_behavior(req),
            self.check_r3_measurable(req),
            self.check_r4_vague(req),
        ]

        failed_checks = [c for c in checks if not c.is_compliant]
        is_compliant = len(failed_checks) == 0

        # Prioritize failure reason (R4 vague term, then R1 ID, then R2 behavior)
        primary_reason = None
        if not is_compliant:
            r4_fail = next((c for c in failed_checks if c.rule_id == "R4"), None)
            if r4_fail:
                primary_reason = r4_fail.reason
            else:
                primary_reason = failed_checks[0].reason

        # Version 0.3: Semantic matching against reference rules (all-MiniLM-L6-v2)
        matched_rule_id = None
        matched_rule_name = None
        rule_similarity_score = None
        rule_scores: Dict[str, float] = {}

        if self.enable_semantic_matching and self.section_matcher and self.standard.rules:
            try:
                rule_match = self.section_matcher.match_requirement_to_rules(
                    req.statement,
                    self.standard.rules
                )
                matched_rule_id = rule_match.best_rule_id
                matched_rule_name = rule_match.best_rule_name
                rule_similarity_score = rule_match.similarity_score
                rule_scores = rule_match.candidate_scores
            except Exception:
                pass

        return RequirementAudit(
            req_id=req.req_id,
            statement=req.statement,
            line_number=req.line_number,
            is_compliant=is_compliant,
            primary_reason=primary_reason,
            checks=checks,
            section_name=req.section_name,
            matched_rule_id=matched_rule_id,
            matched_rule_name=matched_rule_name,
            rule_similarity_score=rule_similarity_score,
            rule_scores=rule_scores
        )

    def audit_document(
        self,
        requirements: List[Requirement],
        sections: Optional[List[DocumentSection]] = None
    ) -> ComplianceReport:
        """
        Runs compliance audit on all extracted requirements, aligns document
        sections against standard sections, and semantically matches requirements to rules.
        """
        self.reset_state()
        audits = [self.audit_requirement(r) for r in requirements]

        total = len(audits)
        compliant = sum(1 for a in audits if a.is_compliant)
        non_compliant = total - compliant
        percentage = round((compliant / total * 100), 1) if total > 0 else 0.0

        section_matches: List[SectionMatchResult] = []
        sections_summary: Dict[str, Any] = {}

        if self.enable_semantic_matching and self.section_matcher:
            try:
                doc_sections = sections if sections is not None else []
                section_matches = self.section_matcher.match_sections(
                    self.standard.sections,
                    doc_sections,
                    threshold=self.similarity_threshold
                )
                matched_cnt = sum(1 for m in section_matches if m.is_matched)
                total_sec = len(section_matches)
                sections_summary = {
                    "total_sections": total_sec,
                    "matched_sections": matched_cnt,
                    "missing_sections": total_sec - matched_cnt,
                    "coverage_percentage": round(matched_cnt / total_sec * 100, 1) if total_sec > 0 else 0.0,
                    "semantic_model": self.section_matcher.model_name,
                    "similarity_threshold": self.similarity_threshold,
                    "document_has_headers": bool(doc_sections)
                }
            except Exception as e:
                sections_summary = {
                    "error": str(e),
                    "semantic_model": getattr(self.section_matcher, "model_name", "all-MiniLM-L6-v2")
                }

        return ComplianceReport(
            total_requirements=total,
            compliant_count=compliant,
            non_compliant_count=non_compliant,
            compliance_percentage=percentage,
            audits=audits,
            section_matches=section_matches,
            sections_summary=sections_summary
        )


def canonicalize_instruction(text: str) -> str:
    """
    Deterministic canonicalization of instructional boilerplate into a declarative retrieval query.
    Preserves all core domain terms and original requirement meaning.
    """
    s = text.strip()
    pat = (
        r'^(?:Explain|Describe|State|Identify|Provide|List|Record|Detail|Include|Specify|Outline|Define|Discuss)'
        r'(?:\s+(?:what\s+is|what|how|whether|the|any|all|at\s+least\s+one))?\s+'
    )
    match = re.match(pat, s, re.IGNORECASE)
    if match:
        s = s[match.end():].strip()
    if s:
        s = s[0].upper() + s[1:]
    return s


class TemplateContentChecker:
    """
    Compares required points from a ReferenceTemplate against candidate text
    chunks extracted from a target document or single page using dense embeddings
    (sentence-transformers all-MiniLM-L6-v2) and configurable cosine similarity thresholds.

    Classification Rules (strictly threshold-based cosine similarity, not probabilities):
    - COVERED : similarity_score >= covered_threshold (default: 0.70)
    - PARTIAL : partial_threshold <= similarity_score < covered_threshold (default: 0.50)
    - MISSING : similarity_score < partial_threshold (default: < 0.50)

    Evidence Safeguards:
    - Minimum evidence threshold (default: 0.38): Chunks scoring below this are not shown as evidence.
    - Topic-only safeguard: A topic label match cannot establish compliance without substantive description support.
    - Explicit absence detection: Statements confirming that a requirement is missing are flagged and never count as compliance.
    - Multi-sentence window preference: Prefers shortest evidence candidate when scores are within 0.03.
    """
    DEFAULT_COVERED_THRESHOLD: float = 0.70
    DEFAULT_PARTIAL_THRESHOLD: float = 0.50
    DEFAULT_MIN_EVIDENCE_THRESHOLD: float = 0.38

    def __init__(
        self,
        matcher: Optional[SemanticSectionMatcher] = None,
        covered_threshold: float = DEFAULT_COVERED_THRESHOLD,
        partial_threshold: float = DEFAULT_PARTIAL_THRESHOLD,
        min_evidence_threshold: float = DEFAULT_MIN_EVIDENCE_THRESHOLD
    ):
        if partial_threshold > covered_threshold:
            raise ValueError(
                f"partial_threshold ({partial_threshold}) cannot be greater than covered_threshold ({covered_threshold})."
            )
        if min_evidence_threshold > partial_threshold:
            raise ValueError(
                f"min_evidence_threshold ({min_evidence_threshold}) cannot exceed partial_threshold ({partial_threshold})."
            )
        self.matcher = matcher or SemanticSectionMatcher()
        self.covered_threshold = covered_threshold
        self.partial_threshold = partial_threshold
        self.min_evidence_threshold = min_evidence_threshold

    def check_compliance(
        self,
        template: ReferenceTemplate,
        target_chunks: List[Any],
        target_name: str = "Target Document",
        target_page: Optional[int] = None,
        total_pages: int = 1
    ) -> TemplateComplianceReport:
        """
        Compares each template point against candidate target chunks using
        multi-perspective embeddings, multi-sentence sliding windows, and evidence safeguards.
        """
        if not template.points:
            return TemplateComplianceReport(
                template_title=template.title,
                target_name=target_name,
                target_page=target_page,
                total_pages=total_pages,
                covered_threshold=self.covered_threshold,
                partial_threshold=self.partial_threshold,
                min_evidence_threshold=self.min_evidence_threshold,
                total_points=0,
                covered_count=0,
                partial_count=0,
                missing_count=0,
                coverage_percentage=0.0,
                results=[],
                sections_summary={"message": "No points found in reference template."}
            )

        # Import absence pattern and CandidateChunk from pdf_extractor
        from src.pdf_extractor import CandidateChunk, EXPLICIT_ABSENCE_PATTERN

        # Parse target chunk metadata
        chunk_texts: List[str] = []
        chunk_reprs: List[str] = []
        chunk_absences: List[bool] = []
        chunk_windows: List[int] = []
        chunk_pages: List[Optional[int]] = []

        for c in target_chunks:
            c_str = str(c).strip()
            if len(c_str) < 5:
                continue
            chunk_texts.append(c_str)
            if isinstance(c, CandidateChunk):
                heading_ctx = f"[{c.section_heading}] " if c.section_heading else ""
                chunk_reprs.append(f"{heading_ctx}{c_str}")
                chunk_absences.append(bool(c.is_explicit_absence))
                chunk_windows.append(c.window_size)
                chunk_pages.append(c.page_number)
            else:
                chunk_reprs.append(c_str)
                chunk_absences.append(bool(EXPLICIT_ABSENCE_PATTERN.search(c_str)))
                chunk_windows.append(1)
                chunk_pages.append(None)

        # Handle empty target document
        if not chunk_texts:
            results = [
                PointMatchResult(
                    point_id=p.point_id,
                    section_name=p.section_name,
                    reference_text=p.text,
                    status="MISSING",
                    similarity_score=0.0,
                    matched_text=None,
                    notes="Target document or page contains no extractable text content.",
                    evidence_tier="NONE"
                )
                for p in template.points
            ]
            return TemplateComplianceReport(
                template_title=template.title,
                target_name=target_name,
                target_page=target_page,
                total_pages=total_pages,
                covered_threshold=self.covered_threshold,
                partial_threshold=self.partial_threshold,
                min_evidence_threshold=self.min_evidence_threshold,
                total_points=len(results),
                covered_count=0,
                partial_count=0,
                missing_count=len(results),
                coverage_percentage=0.0,
                results=results,
                sections_summary={"message": "Target document contains no text."}
            )

        _, util = _import_sentence_transformers()

        # Multi-perspective point representations:
        # 1. Full prompt: combines topic label and detailed description
        full_prompts = [
            f"{p.topic_label}: {p.description}" if p.topic_label and p.description else p.text
            for p in template.points
        ]
        # 2. Canonical query projection: topic + canonical description
        canon_prompts = [
            f"{p.topic_label}: {canonicalize_instruction(p.description or p.text)}" if p.topic_label
            else canonicalize_instruction(p.text)
            for p in template.points
        ]

        # 3. Topic/Concept prompt: pairs section context with core topic name
        topic_prompts = [
            f"{p.section_name}: {p.topic_label}" if p.topic_label else f"{p.section_name}: {p.text}"
            for p in template.points
        ]

        # Batch encode all embeddings
        target_embs = self.matcher.model.encode(chunk_reprs, convert_to_tensor=True, show_progress_bar=False)
        full_embs = self.matcher.model.encode(full_prompts, convert_to_tensor=True, show_progress_bar=False)
        canon_embs = self.matcher.model.encode(canon_prompts, convert_to_tensor=True, show_progress_bar=False)
        topic_embs = self.matcher.model.encode(topic_prompts, convert_to_tensor=True, show_progress_bar=False)

        # Compute pairwise cosine similarities across representations
        sim_full = util.cos_sim(full_embs, target_embs)
        sim_canon = util.cos_sim(canon_embs, target_embs)
        sim_topic = util.cos_sim(topic_embs, target_embs)

        results: List[PointMatchResult] = []
        covered_count = 0
        partial_count = 0
        missing_count = 0

        for i, point in enumerate(template.points):
            candidate_evals = []
            for j in range(len(chunk_texts)):
                s_f = float(sim_full[i, j].item())
                s_c = float(sim_canon[i, j].item())
                s_t = float(sim_topic[i, j].item())

                # Substantive support from requirement description / canonical representation
                s_substantive = max(s_f, s_c)

                # Safeguard 1: Topic-only match cannot establish compliance on its own
                # If topic score is high but description support is below min_evidence_threshold,
                # the match is deemed coincidental keyword overlap.
                if s_t > s_substantive and s_substantive < self.min_evidence_threshold:
                    eff_score = s_substantive
                    persp = "canonical" if s_c >= s_f else "full"
                else:
                    eff_score = max(s_f, s_c, s_t)
                    if eff_score == s_t:
                        persp = "topic"
                    elif eff_score == s_c:
                        persp = "canonical"
                    else:
                        persp = "full"

                candidate_evals.append({
                    "idx": j,
                    "text": chunk_texts[j],
                    "score": round(eff_score, 4),
                    "substantive_score": round(s_substantive, 4),
                    "window_size": chunk_windows[j],
                    "is_absence": chunk_absences[j],
                    "perspective": persp,
                    "page": chunk_pages[j]
                })

            # Check for explicit absence statement semantically relevant to this requirement
            relevant_absence_candidates = [
                c for c in candidate_evals
                if c["is_absence"] and c["substantive_score"] >= self.min_evidence_threshold
            ]

            # Positive (non-absence) candidates
            positive_candidates = [c for c in candidate_evals if not c["is_absence"]]
            max_positive_score = max((c["score"] for c in positive_candidates), default=0.0)

            # Safeguard 2: Explicit Absence Handling
            if relevant_absence_candidates and max_positive_score < self.partial_threshold:
                # Target explicitly indicates this requirement is absent/omitted
                best_absence = max(relevant_absence_candidates, key=lambda x: x["substantive_score"])
                status = "MISSING"
                missing_count += 1
                evidence = best_absence["text"]
                evidence_tier = "EXPLICIT_ABSENCE"
                notes = (
                    f"Target document explicitly confirms absence or omission of this requirement "
                    f"(cosine similarity: {best_absence['score']:.4f}, via {best_absence['perspective']}): '{evidence}'."
                )
                results.append(
                    PointMatchResult(
                        point_id=point.point_id,
                        section_name=point.section_name,
                        reference_text=point.text,
                        status=status,
                        similarity_score=best_absence["score"],
                        matched_text=evidence,
                        notes=notes,
                        winning_perspective=best_absence["perspective"],
                        evidence_tier=evidence_tier,
                        is_explicit_absence=True,
                        matched_page=best_absence["page"]
                    )
                )
                continue

            # Multi-Sentence Window Preference:
            # When multiple candidates have comparable scores (within 0.03), prefer the shortest snippet.
            max_score = max(c["score"] for c in candidate_evals)
            close_candidates = [c for c in candidate_evals if c["score"] >= max_score - 0.03]
            close_candidates.sort(key=lambda x: (x["window_size"], -x["score"]))
            best = close_candidates[0]

            score = best["score"]
            evidence_text = best["text"]
            persp = best["perspective"]
            page = best["page"]
            is_absence_noted = False

            # Check if there is an omission note accompanying partial positive evidence
            if relevant_absence_candidates and score >= self.partial_threshold:
                is_absence_noted = True
                absence_note_text = relevant_absence_candidates[0]["text"]
            else:
                absence_note_text = ""

            # Classification and Evidence Tiers
            if score >= self.covered_threshold:
                status = "COVERED"
                covered_count += 1
                evidence_tier = "STRONG"
                evidence = evidence_text
                notes = f"Requirement satisfied (cosine similarity {score:.4f} >= {self.covered_threshold:.2f}, via {persp})."
            elif score >= self.partial_threshold:
                status = "PARTIAL"
                partial_count += 1
                evidence_tier = "PARTIAL"
                evidence = evidence_text
                if is_absence_noted:
                    notes = (
                        f"Partial semantic match (cosine similarity {score:.4f} in "
                        f"[{self.partial_threshold:.2f}, {self.covered_threshold:.2f}), via {persp}), "
                        f"with explicit target omission noted: '{absence_note_text}'."
                    )
                else:
                    notes = (
                        f"Partial semantic match (cosine similarity {score:.4f} in "
                        f"[{self.partial_threshold:.2f}, {self.covered_threshold:.2f}), via {persp}). Review required."
                    )
            elif score >= self.min_evidence_threshold:
                status = "MISSING"
                missing_count += 1
                evidence_tier = "WEAK"
                evidence = evidence_text
                notes = (
                    f"Requirement missing in target (weak candidate with cosine similarity "
                    f"{score:.4f} in [{self.min_evidence_threshold:.2f}, {self.partial_threshold:.2f}), via {persp}). "
                    f"Insufficient evidence to establish coverage."
                )
            else:
                status = "MISSING"
                missing_count += 1
                evidence_tier = "NONE"
                evidence = None
                notes = (
                    f"No relevant evidence found in target document "
                    f"(highest candidate cosine similarity was {score:.4f} < {self.min_evidence_threshold:.2f})."
                )

            results.append(
                PointMatchResult(
                    point_id=point.point_id,
                    section_name=point.section_name,
                    reference_text=point.text,
                    status=status,
                    similarity_score=score,
                    matched_text=evidence,
                    notes=notes,
                    winning_perspective=persp,
                    evidence_tier=evidence_tier,
                    is_explicit_absence=is_absence_noted,
                    matched_page=page
                )
            )

        total_pts = len(results)
        coverage_pct = round((covered_count / total_pts * 100), 1) if total_pts > 0 else 0.0

        # Section-level breakdown
        sections_summary: Dict[str, Dict[str, int]] = {}
        for r in results:
            sec = r.section_name
            if sec not in sections_summary:
                sections_summary[sec] = {"total": 0, "covered": 0, "partial": 0, "missing": 0}
            sections_summary[sec]["total"] += 1
            if r.status == "COVERED":
                sections_summary[sec]["covered"] += 1
            elif r.status == "PARTIAL":
                sections_summary[sec]["partial"] += 1
            else:
                sections_summary[sec]["missing"] += 1

        return TemplateComplianceReport(
            template_title=template.title,
            target_name=target_name,
            target_page=target_page,
            total_pages=total_pages,
            covered_threshold=self.covered_threshold,
            partial_threshold=self.partial_threshold,
            min_evidence_threshold=self.min_evidence_threshold,
            total_points=total_pts,
            covered_count=covered_count,
            partial_count=partial_count,
            missing_count=missing_count,
            coverage_percentage=coverage_pct,
            results=results,
            sections_summary=sections_summary
        )


class ReportGenerator:
    """Formats and exports compliance audit reports (Terminal, Markdown, JSON)."""

    @staticmethod
    def format_terminal(report: ComplianceReport) -> str:
        lines = []
        lines.append("=" * 72)
        lines.append("       SRS COMPLIANCE & QUALITY AUDIT REPORT (VERSION 0.3)")
        lines.append("=" * 72)
        lines.append("")

        # Section 1: Semantic Section Alignment
        if report.section_matches:
            lines.append("SEMANTIC SECTION ALIGNMENT (sentence-transformers / all-MiniLM-L6-v2):")
            lines.append("-" * 72)
            has_headers = report.sections_summary.get("document_has_headers", True)
            if not has_headers:
                lines.append("  [Notice] Document contains no section headers. Section alignment skipped.")
                lines.append("-" * 72)
            else:
                threshold = report.sections_summary.get("similarity_threshold", 0.55)
                for match in report.section_matches:
                    status_tag = "[✓ MATCHED]" if match.is_matched else "[✗ MISSING]"
                    lines.append(f"  {status_tag} {match.reference_section_id}: {match.reference_section_name}")
                    if match.is_matched:
                        lines.append(f"      Matched Section  : \"{match.matched_document_section}\"")
                        lines.append(f"      Similarity Score : {match.similarity_score:.4f} (Threshold >= {threshold})")
                    else:
                        top_candidate = (
                            max(match.candidate_scores.items(), key=lambda x: x[1])[0]
                            if match.candidate_scores
                            else "None"
                        )
                        lines.append(f"      Highest Candidate: \"{top_candidate}\"")
                        lines.append(f"      Similarity Score : {match.similarity_score:.4f} (Below Threshold: {threshold})")
                        lines.append(f"      Status           : Mandatory section missing from SRS document")
                    lines.append("")

                summary = report.sections_summary
                lines.append("  SECTION COVERAGE SUMMARY:")
                lines.append(f"    Mandatory Sections   : {summary.get('total_sections', len(report.section_matches))}")
                lines.append(f"    Matched Sections     : {summary.get('matched_sections', 0)} ({summary.get('coverage_percentage', 0.0)}%)")
                lines.append(f"    Missing Sections     : {summary.get('missing_sections', 0)}")
                lines.append("-" * 72)
                lines.append("")

        # Section 2: Requirement Quality Findings (R1 - R4) + Semantic Rule Alignment
        lines.append("REQUIREMENT QUALITY FINDINGS (RULES R1 - R4 & SEMANTIC RULE MATCHING):")
        lines.append("-" * 72)
        for audit in report.audits:
            req_tag = f"[{audit.req_id}]" if audit.req_id else "[MISSING-ID]"
            status_text = "COMPLIANT" if audit.is_compliant else "NON-COMPLIANT"
            sec_info = f" (Section: {audit.section_name})" if audit.section_name else ""

            lines.append(f"{req_tag}{sec_info} → {status_text}")
            lines.append(f"  Statement: \"{audit.statement}\"")

            if audit.matched_rule_id and audit.rule_similarity_score is not None:
                lines.append(f"  Semantic Rule Alignment : {audit.matched_rule_id} ({audit.matched_rule_name}) | Score: {audit.rule_similarity_score:.4f}")

            if not audit.is_compliant and audit.primary_reason:
                lines.append(f"  Reason: {audit.primary_reason}")

            lines.append("  Rule Breakdown:")
            for check in audit.checks:
                mark = "✓" if check.is_compliant else "✗"
                lines.append(f"    [{mark}] {check.rule_id} ({check.rule_name}): {check.reason}")
            lines.append("-" * 72)

        lines.append("")
        lines.append("AUDIT SUMMARY:")
        lines.append(f"  Total Requirements Analyzed : {report.total_requirements}")
        lines.append(f"  Compliant Requirements       : {report.compliant_count}")
        lines.append(f"  Non-Compliant Requirements   : {report.non_compliant_count}")
        lines.append(f"  Overall Compliance Rate      : {report.compliance_percentage}%")
        if report.sections_summary and report.sections_summary.get("document_has_headers", True):
            lines.append(
                f"  Section Alignment Coverage   : {report.sections_summary.get('coverage_percentage', 0.0)}% "
                f"({report.sections_summary.get('matched_sections', 0)}/{report.sections_summary.get('total_sections', 0)} sections)"
            )
        lines.append("=" * 72)

        return "\n".join(lines)

    @staticmethod
    def format_markdown(report: ComplianceReport) -> str:
        md = []
        md.append("# SRS Compliance Verification Report (v0.3)\n")
        md.append("## 1. Executive Summary")
        md.append(f"- **Total Requirements**: {report.total_requirements}")
        md.append(f"- **Compliant Requirements**: {report.compliant_count} ({report.compliance_percentage}%)")
        md.append(f"- **Non-Compliant Requirements**: {report.non_compliant_count}")
        if report.sections_summary and report.sections_summary.get("document_has_headers", True):
            matched = report.sections_summary.get('matched_sections', 0)
            total = report.sections_summary.get('total_sections', 0)
            cov = report.sections_summary.get('coverage_percentage', 0.0)
            md.append(f"- **Mandatory Sections Coverage**: {matched}/{total} ({cov}%)")
        md.append(f"- **Semantic Matching Engine**: `sentence-transformers` (`{report.sections_summary.get('semantic_model', 'all-MiniLM-L6-v2')}`)")
        md.append("")

        if report.section_matches and report.sections_summary.get("document_has_headers", True):
            md.append("## 2. Semantic Section Alignment (all-MiniLM-L6-v2)")
            md.append("| Ref Section ID | Standard Section Name | Matched Document Section | Similarity Score | Status | Notes |")
            md.append("| :--- | :--- | :--- | :---: | :---: | :--- |")
            for m in report.section_matches:
                status_badge = "✅ MATCHED" if m.is_matched else "❌ MISSING"
                matched_sec = f"`{m.matched_document_section}`" if m.matched_document_section else "*None (Missing)*"
                md.append(f"| {m.reference_section_id} | **{m.reference_section_name}** | {matched_sec} | **{m.similarity_score:.4f}** | {status_badge} | {m.notes} |")
            md.append("")

        md.append("## 3. Requirement Quality Findings (Rules R1 - R4 & Semantic Rule Alignment)")
        for idx, audit in enumerate(report.audits, start=1):
            badge = "✅ COMPLIANT" if audit.is_compliant else "❌ NON-COMPLIANT"
            req_label = audit.req_id or "MISSING-ID"
            sec_label = f" *[Section: {audit.section_name}]*" if audit.section_name else ""
            md.append(f"### {idx}. [{req_label}]{sec_label} → {badge}")
            md.append(f"> \"{audit.statement}\"\n")

            if audit.matched_rule_id and audit.rule_similarity_score is not None:
                md.append(f"**Semantic Rule Alignment**: `{audit.matched_rule_id} ({audit.matched_rule_name})` — Similarity Score: **`{audit.rule_similarity_score:.4f}`**\n")

            if not audit.is_compliant and audit.primary_reason:
                md.append(f"**Reason**: `{audit.primary_reason}`\n")

            md.append("| Rule | Name | Status | Details |")
            md.append("| :--- | :--- | :---: | :--- |")
            for c in audit.checks:
                s = "PASS" if c.is_compliant else "FAIL"
                md.append(f"| {c.rule_id} | {c.rule_name} | {s} | {c.reason} |")
            md.append("")

        return "\n".join(md)

    @staticmethod
    def to_json(report: ComplianceReport) -> Dict[str, Any]:
        return {
            "version": "0.3",
            "summary": {
                "total_requirements": report.total_requirements,
                "compliant_count": report.compliant_count,
                "non_compliant_count": report.non_compliant_count,
                "compliance_percentage": report.compliance_percentage,
                "sections_summary": report.sections_summary
            },
            "section_matches": [
                {
                    "reference_section_id": m.reference_section_id,
                    "reference_section_name": m.reference_section_name,
                    "reference_section_description": m.reference_section_description,
                    "matched_document_section": m.matched_document_section,
                    "similarity_score": m.similarity_score,
                    "is_matched": m.is_matched,
                    "notes": m.notes,
                    "candidate_scores": m.candidate_scores
                }
                for m in report.section_matches
            ],
            "audits": [
                {
                    "req_id": a.req_id,
                    "section_name": a.section_name,
                    "statement": a.statement,
                    "line_number": a.line_number,
                    "is_compliant": a.is_compliant,
                    "primary_reason": a.primary_reason,
                    "matched_rule_id": a.matched_rule_id,
                    "matched_rule_name": a.matched_rule_name,
                    "rule_similarity_score": a.rule_similarity_score,
                    "rule_scores": a.rule_scores,
                    "checks": [asdict(c) for c in a.checks]
                }
                for a in report.audits
            ]
        }

    @staticmethod
    def format_template_terminal(report: TemplateComplianceReport) -> str:
        """Formats TemplateComplianceReport into a clean, Windows-safe terminal string."""
        lines = []
        lines.append("=" * 76)
        lines.append("       REFERENCE-TEMPLATE CONTENT COMPLIANCE REPORT (PHASE 1.1)")
        lines.append("=" * 76)
        lines.append(f"Reference Template : {report.template_title}")
        lines.append(f"Target Document    : {report.target_name}")
        if report.target_page is not None:
            scope_str = f"Single Page (Page {report.target_page} of {report.total_pages})"
        else:
            scope_str = f"Entire Document ({report.total_pages} page(s))"
        lines.append(f"Target Scope       : {scope_str}")
        lines.append(f"Model Engine       : sentence-transformers (all-MiniLM-L6-v2)")
        lines.append(f"Similarity Metric  : Cosine Similarity [-1.0, 1.0]")
        lines.append(
            f"Decision Boundary  : COVERED >= {report.covered_threshold:.2f} | "
            f"PARTIAL >= {report.partial_threshold:.2f} | MIN_EVIDENCE >= {report.min_evidence_threshold:.2f}"
        )
        lines.append("-" * 76)
        lines.append("EXECUTIVE SUMMARY:")
        lines.append(f"  Total Template Points Required : {report.total_points}")
        cov_pct = round(report.covered_count / report.total_points * 100, 1) if report.total_points else 0.0
        part_pct = round(report.partial_count / report.total_points * 100, 1) if report.total_points else 0.0
        miss_pct = round(report.missing_count / report.total_points * 100, 1) if report.total_points else 0.0
        lines.append(f"  [COVERED]  Requirements Met    : {report.covered_count} ({cov_pct}%)")
        lines.append(f"  [PARTIAL]  Ambiguous / Gap     : {report.partial_count} ({part_pct}%)")
        lines.append(f"  [MISSING]  Not Found           : {report.missing_count} ({miss_pct}%)")
        lines.append(f"  Overall Coverage Rate          : {report.coverage_percentage}%")
        lines.append("-" * 76)

        if report.sections_summary:
            lines.append("SECTION BREAKDOWN:")
            for sec_name, counts in report.sections_summary.items():
                if isinstance(counts, dict) and "total" in counts:
                    lines.append(
                        f"  * {sec_name:<36} : {counts['covered']}/{counts['total']} covered "
                        f"({counts['partial']} partial, {counts['missing']} missing)"
                    )
            lines.append("-" * 76)

        lines.append("POINT-BY-POINT COMPLIANCE FINDINGS:")
        lines.append("-" * 76)
        for idx, res in enumerate(report.results, start=1):
            if res.is_explicit_absence and res.status == "MISSING":
                badge = "[MISSING (Explicit Absence)]"
            else:
                badge = f"[{res.status}]"
            page_info = f" (Page {res.matched_page})" if res.matched_page else ""
            lines.append(f"{idx}. {badge} {res.point_id} (Section: {res.section_name})")
            lines.append(f"   Required : \"{res.reference_text}\"")
            if res.is_explicit_absence and res.status == "MISSING":
                lines.append(f"   Target Absence Statement{page_info} : \"{res.matched_text}\"")
            elif res.matched_text:
                lines.append(f"   Evidence [{res.evidence_tier}]{page_info} : \"{res.matched_text}\"")
            else:
                lines.append("   Evidence : None (no target statement met the minimum evidence threshold)")
            lines.append(f"   Metric   : Cosine Similarity = {res.similarity_score:.4f} (via {res.winning_perspective or 'none'})")
            lines.append(f"   Details  : {res.notes}")
            lines.append("-" * 76)

        lines.append("=" * 76)
        return "\n".join(lines)

    @staticmethod
    def format_template_markdown(report: TemplateComplianceReport) -> str:
        """Formats TemplateComplianceReport into a comprehensive Markdown document."""
        md = []
        md.append("# Reference-Template Content Compliance Report\n")
        md.append("## 1. Executive Summary\n")
        md.append(f"- **Reference Template**: `{report.template_title}`")
        md.append(f"- **Target Document**: `{report.target_name}`")
        if report.target_page is not None:
            md.append(f"- **Target Scope**: Single Page (Page {report.target_page} of {report.total_pages})")
        else:
            md.append(f"- **Target Scope**: Entire Document ({report.total_pages} page(s))")
        md.append(f"- **Overall Coverage Rate**: **{report.coverage_percentage}%**")
        md.append(f"- **Covered Points**: {report.covered_count} / {report.total_points}")
        md.append(f"- **Partial Points**: {report.partial_count} / {report.total_points}")
        md.append(f"- **Missing Points**: {report.missing_count} / {report.total_points}")
        md.append(
            f"- **Thresholds**: COVERED >= {report.covered_threshold:.2f}, "
            f"PARTIAL >= {report.partial_threshold:.2f}, MIN_EVIDENCE >= {report.min_evidence_threshold:.2f}\n"
        )

        if report.sections_summary:
            md.append("## 2. Section Coverage Breakdown\n")
            md.append("| Section Name | Total Required | Covered | Partial | Missing | Coverage % |")
            md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
            for sec_name, counts in report.sections_summary.items():
                if isinstance(counts, dict) and "total" in counts:
                    pct = round(counts['covered'] / counts['total'] * 100, 1) if counts['total'] else 0.0
                    md.append(
                        f"| **{sec_name}** | {counts['total']} | {counts['covered']} | "
                        f"{counts['partial']} | {counts['missing']} | {pct}% |"
                    )
            md.append("")

        md.append("## 3. Detailed Point Findings\n")
        for idx, res in enumerate(report.results, start=1):
            if res.is_explicit_absence and res.status == "MISSING":
                status_icon = "MISSING (EXPLICIT ABSENCE)"
            elif res.status == "COVERED":
                status_icon = "PASS"
            elif res.status == "PARTIAL":
                status_icon = "PARTIAL"
            else:
                status_icon = "MISSING"

            page_info = f" (Page {res.matched_page})" if res.matched_page else ""
            md.append(f"### {idx}. [{status_icon}] {res.point_id} — Section: {res.section_name}\n")
            md.append(f"**Required Statement:**\n> {res.reference_text}\n")
            if res.is_explicit_absence and res.status == "MISSING":
                md.append(f"**Target Absence Statement{page_info}:**\n> {res.matched_text}\n")
            elif res.matched_text:
                md.append(f"**Target Evidence [{res.evidence_tier}]{page_info}:**\n> {res.matched_text}\n")
            else:
                md.append("**Target Evidence:** *None found in target content.*\n")
            md.append(f"- **Cosine Similarity**: `{res.similarity_score:.4f}` (via `{res.winning_perspective or 'none'}`)")
            md.append(f"- **Analysis**: {res.notes}\n")

        return "\n".join(md)

    @staticmethod
    def to_template_json(report: TemplateComplianceReport) -> Dict[str, Any]:
        """Serializes TemplateComplianceReport to a structured dictionary."""
        return {
            "version": "1.1-phase1.1",
            "metadata": {
                "template_title": report.template_title,
                "target_name": report.target_name,
                "target_page": report.target_page,
                "total_pages": report.total_pages,
                "covered_threshold": report.covered_threshold,
                "partial_threshold": report.partial_threshold,
                "min_evidence_threshold": report.min_evidence_threshold,
                "similarity_metric": "cosine_similarity"
            },
            "summary": {
                "total_points": report.total_points,
                "covered_count": report.covered_count,
                "partial_count": report.partial_count,
                "missing_count": report.missing_count,
                "coverage_percentage": report.coverage_percentage,
                "sections_summary": report.sections_summary
            },
            "results": [
                {
                    "point_id": r.point_id,
                    "section_name": r.section_name,
                    "reference_text": r.reference_text,
                    "status": r.status,
                    "similarity_score": r.similarity_score,
                    "matched_text": r.matched_text,
                    "notes": r.notes,
                    "winning_perspective": r.winning_perspective,
                    "evidence_tier": r.evidence_tier,
                    "is_explicit_absence": r.is_explicit_absence,
                    "matched_page": r.matched_page
                }
                for r in report.results
            ]
        }
