"""
Generates the AI SRS Compliance Analyzer Intern Architecture & Onboarding Guide PDF.
This script uses ReportLab to produce a clean, comprehensive, multi-page reference
manual explaining the system architecture, component interactions, code snippets,
testing workflows, and step-by-step instructions for incoming developers/interns.
"""
import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Preformatted,
    PageBreak,
    HRFlowable,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Canvas that computes total pages dynamically and adds headers/footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#666666"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "AI-SRS Compliance Analyzer — Architecture & Intern Developer Guide")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Footer (all pages)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_str)
        self.drawString(54, 36, "Confidential — Internal Engineering & Onboarding Reference")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)

        self.restoreState()


def build_pdf(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#1A365D")   # Deep Navy
    secondary_color = colors.HexColor("#2B6CB0") # Slate Blue
    body_color = colors.HexColor("#2D3748")      # Charcoal Body Text
    code_bg = colors.HexColor("#F7FAFC")         # Off-white / light slate
    code_border = colors.HexColor("#E2E8F0")

    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=30,
        textColor=primary_color,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=secondary_color,
        spaceAfter=20,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=secondary_color,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=body_color,
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=4,
    )

    code_style = ParagraphStyle(
        "CodeBlock",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#1A202C"),
        backColor=code_bg,
        borderColor=code_border,
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=8,
    )

    callout_style = ParagraphStyle(
        "Callout",
        parent=body_style,
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2C5282"),
        backColor=colors.HexColor("#EBF8FF"),
        borderColor=colors.HexColor("#BEE3F8"),
        borderWidth=0.5,
        borderPadding=8,
        spaceBefore=6,
        spaceAfter=8,
    )

    story = []

    # -------------------------------------------------------------------------
    # COVER / HEADER SECTION
    # -------------------------------------------------------------------------
    story.append(Paragraph("AI-SRS Compliance Analyzer", title_style))
    story.append(Paragraph("System Architecture, Component Interactions & Developer Onboarding Guide", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=14))

    metadata_text = (
        "<b>Target Audience:</b> Incoming Software Engineers, AI/ML Interns, and Project Stakeholders<br/>"
        "<b>Primary Technologies:</b> Python 3.13, PyMuPDF, Sentence-Transformers (MiniLM), DeBERTa-v3 NLI, Rank-BM25<br/>"
        "<b>Repository Status:</b> Complete End-to-End V2 Pipeline with Declarative Benchmarks (151 passing tests)"
    )
    story.append(Paragraph(metadata_text, callout_style))
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # SECTION 1: WHAT DOES THIS PROJECT DO?
    # -------------------------------------------------------------------------
    story.append(Paragraph("1. What Does This Project Do? (Plain English)", h1_style))
    story.append(Paragraph(
        "When engineering teams build complex software or scientific systems, they write a "
        "<b>System Requirements Specification (SRS)</b> or a technical report. Companies and regulatory bodies "
        "have strict checklists and reference standards (e.g., IEEE, ISO, safety guidelines, lab rubrics) "
        "that these documents must comply with.",
        body_style
    ))
    story.append(Paragraph(
        "Traditionally, auditing an SRS is a painful manual process: an engineer or auditor reads a 20-page document "
        "line-by-line with a checklist, searches for matching sentences, verifies numbers, checks whether required features "
        "were omitted, and manually writes an audit report. It is slow, subjective, and error-prone.",
        body_style
    ))
    story.append(Paragraph(
        "<b>The AI-SRS Compliance Analyzer automates this entire audit process:</b>",
        body_style
    ))
    story.append(Paragraph("• <b>Input 1 (Reference Template PDF):</b> The standard, checklist, or rubric containing required sections and rules.", bullet_style))
    story.append(Paragraph("• <b>Input 2 (Target Document PDF):</b> The candidate SRS, lab report, or technical specification being audited.", bullet_style))
    story.append(Paragraph("• <b>Automated Output:</b> An executive compliance report with an overall verdict (COMPLIANT, CONDITIONALLY_COMPLIANT, NON_COMPLIANT), mandatory coverage percentage, and line-by-line evidence proof with exact page numbers and quotes.", bullet_style))
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # SECTION 2: END-TO-END PIPELINE ARCHITECTURE
    # -------------------------------------------------------------------------
    story.append(Paragraph("2. How Does It Work? The 5-Stage Pipeline", h1_style))
    story.append(Paragraph(
        "The system executes a deterministic 5-stage pipeline orchestrated by <code>src/pipeline.py</code>:",
        body_style
    ))

    arch_diagram = (
        "+---------------------------------------------------------------------------------+\n"
        "|                             END-TO-END PIPELINE FLOW                             |\n"
        "+---------------------------------------------------------------------------------+\n"
        "|  Reference Template PDF                     Target Document PDF                 |\n"
        "|            |                                         |                          |\n"
        "|            v                                         v                          |\n"
        "|  [1. Template Ingestion]                    [1. Target PDF Parsing]             |\n"
        "|  (PDFExtractor & TemplateParser)            (TargetPDFParser)                   |\n"
        "|            |                                         |                          |\n"
        "|            v                                         v                          |\n"
        "|  [2. Atomic Decomposition]                  [Structured DocumentBlocks]         |\n"
        "|  (Splits into SubRequirements)              (p1_b1, p1_b2, p1_b3 with bbox)     |\n"
        "|            \\                                         /                          |\n"
        "|             \\                                       /                           |\n"
        "|              v                                     v                            |\n"
        "|     +-----------------------------------------------------+                     |\n"
        "|     | [3. Hybrid Candidate Retrieval (HybridRetriever)]    |                     |\n"
        "|     | - BM25 Lexical Keyword Search                       |                     |\n"
        "|     | - Dense Semantic Embeddings (all-MiniLM-L6-v2)      |                     |\n"
        "|     | - Reciprocal Rank Fusion (RRF: k=60)                |                     |\n"
        "|     +-----------------------------------------------------+                     |\n"
        "|                                |                                                |\n"
        "|                                v                                                |\n"
        "|     +-----------------------------------------------------+                     |\n"
        "|     | [4. NLI Evidence Verification (EvidenceVerifier)]   |                     |\n"
        "|     | - Cross-Encoder Transformer (nli-deberta-v3-small)  |                     |\n"
        "|     | - Premise = Candidate Block, Hypothesis = Claim    |                     |\n"
        "|     | - Refinements: Boundary Clauses, Meta-Discourse     |                     |\n"
        "|     | - Classifies: FULFILLED, CONTRADICTED, OMISSION, ...|                     |\n"
        "|     +-----------------------------------------------------+                     |\n"
        "|                                |                                                |\n"
        "|                                v                                                |\n"
        "|     +-----------------------------------------------------+                     |\n"
        "|     | [5. Compliance Aggregation (ComplianceAggregator)]  |                     |\n"
        "|     | - Rolls up sub-results to parent sections           |                     |\n"
        "|     | - Computes mandatory & optional coverage %          |                     |\n"
        "|     | - Computes document verdict (COMPLIANT / NON)       |                     |\n"
        "|     +-----------------------------------------------------+                     |\n"
        "|                                |                                                |\n"
        "|                                v                                                |\n"
        "|     [Terminal ANSI Summary] [compliance_report.md] [compliance_report.json]     |\n"
        "+---------------------------------------------------------------------------------+"
    )
    story.append(Preformatted(arch_diagram, code_style))
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # SECTION 3: DEEP DIVE INTO COMPONENTS & CODE SNIPPETS
    # -------------------------------------------------------------------------
    story.append(Paragraph("3. Deep Dive into the Core Components (With Code Snippets)", h1_style))

    story.append(Paragraph("Stage 1: Document Parsing & Layout Chunking (src/parsers/pdf_parser.py)", h2_style))
    story.append(Paragraph(
        "Instead of reading raw, messy text strings, <code>TargetPDFParser</code> parses the PDF layout. "
        "It groups contiguous spans of text into cohesive <code>DocumentBlock</code> objects and assigns each "
        "a deterministic identifier like <code>p1_b3</code> (Page 1, Block 3). It classifies each block as "
        "a HEADING, PARAGRAPH, LIST_ITEM, TABLE_ROW, or DISCLAIMER.",
        body_style
    ))
    code_block_snippet = (
        "@dataclass\n"
        "class DocumentBlock:\n"
        "    block_id: str                          # e.g., 'p1_b3'\n"
        "    text: str                              # Clean text content\n"
        "    block_type: BlockType                  # PARAGRAPH, HEADING, DISCLAIMER\n"
        "    page_number: int                       # 1-based source page\n"
        "    section_heading: Optional[str] = None  # Parent section title\n"
        "    is_explicit_omission: bool = False     # True if text states 'not included'"
    )
    story.append(Preformatted(code_block_snippet, code_style))

    story.append(Paragraph("Stage 2: Requirement Decomposition (src/parsers/requirement_decomposer.py)", h2_style))
    story.append(Paragraph(
        "A single human requirement in a checklist often contains multiple testable claims (e.g., 'State the purpose "
        "and describe what it is intended to measure'). The <code>RequirementDecomposer</code> breaks parent points into "
        "atomic <code>SubRequirement</code> claims, each with its own expectation type (FACTUAL_STATEMENT, "
        "NUMERIC_TARGET, SAFETY_CONSTRAINT, or INTERFACE_SPECIFICATION).",
        body_style
    ))
    code_subreq_snippet = (
        "@dataclass\n"
        "class SubRequirement:\n"
        "    sub_id: str                            # e.g., 'B1.1', 'B1.2'\n"
        "    parent_point_id: str                   # e.g., 'B1'\n"
        "    claim_text: str                        # Atomic verifiable hypothesis\n"
        "    expected_evidence_type: EvidenceType   # FACTUAL_STATEMENT, NUMERIC_TARGET...\n"
        "    is_mandatory: bool = True              # Mandatory vs Optional flag\n"
        "    decomposition_rationale: str = \"\""
    )
    story.append(Preformatted(code_subreq_snippet, code_style))

    story.append(Paragraph("Stage 3: Hybrid Retrieval with Reciprocal Rank Fusion (src/retrieval/)", h2_style))
    story.append(Paragraph(
        "Why Hybrid? Keyword search alone fails when documents use synonyms ('cooling rate' vs 'temperature decrease'). "
        "Dense embedding search alone fails when matching exact acronyms, numbers, or section codes ('ISO 26262', '70 °C'). "
        "The <code>HybridRetriever</code> runs both and mathematically fuses their ranks using <b>Reciprocal Rank Fusion (RRF)</b>:",
        body_style
    ))
    code_retrieval_snippet = (
        "# Reciprocal Rank Fusion formula with smoothing factor k=60:\n"
        "rrf_score = 1.0 / (60 + rank_bm25) + 1.0 / (60 + rank_dense)\n\n"
        "# Retrieves Top-K fused candidates for each atomic sub-requirement claim\n"
        "candidates = retriever.retrieve_candidates_for_subrequirement(sr, top_k=5)"
    )
    story.append(Preformatted(code_retrieval_snippet, code_style))

    story.append(Paragraph("Stage 4: AI Evidence Verification via NLI (src/verification/evidence_verifier.py)", h2_style))
    story.append(Paragraph(
        "The system does not rely on naive keyword matching. It feeds the retrieved candidate passage and the requirement claim "
        "into a cross-encoder Natural Language Inference transformer (<code>nli-deberta-v3-small</code>). "
        "The model computes probabilities for <b>Entailment</b> (proof found), <b>Contradiction</b> (conflicting claim), "
        "and <b>Neutral</b> (unrelated text).",
        body_style
    ))
    story.append(Paragraph("• <b>FULFILLED:</b> Entailment probability exceeds threshold (e.g. >0.70).", bullet_style))
    story.append(Paragraph("• <b>CONTRADICTED:</b> Evidence directly opposes the requirement.", bullet_style))
    story.append(Paragraph("• <b>EXPLICIT_OMISSION:</b> Document contains an explicit disclaimer statement (e.g. 'Safety precautions are not covered').", bullet_style))
    story.append(Paragraph("• <b>MISSING:</b> No relevant or entailing evidence found in candidate blocks.", bullet_style))
    story.append(Paragraph("• <b>NEEDS_REVIEW:</b> Conflicting signals or ambiguous evidence flagged for human inspection.", bullet_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Stage 5: Hierarchical Compliance Aggregation (src/verification/compliance_aggregator.py)", h2_style))
    story.append(Paragraph(
        "The aggregator rolls up granular atomic decisions into parent sections (FULLY_COMPLIANT, PARTIALLY_COMPLIANT, "
        "NON_COMPLIANT, NEEDS_REVIEW) and calculates exact mathematical coverage metrics:",
        body_style
    ))
    code_agg_snippet = (
        "# Mandatory Coverage formula (rounded to 1 decimal place):\n"
        "mandatory_coverage_pct = round((mandatory_fulfilled / mandatory_total) * 100.0, 1)\n\n"
        "# Document Verdict Rules:\n"
        "if any_mandatory_has_needs_review:\n"
        "    verdict = DocumentComplianceVerdict.CONDITIONALLY_COMPLIANT\n"
        "elif mandatory_coverage_pct == 100.0 and no_mandatory_contradictions:\n"
        "    verdict = DocumentComplianceVerdict.COMPLIANT\n"
        "else:\n"
        "    verdict = DocumentComplianceVerdict.NON_COMPLIANT"
    )
    story.append(Preformatted(code_agg_snippet, code_style))

    # -------------------------------------------------------------------------
    # SECTION 4: THE DECLARATIVE BENCHMARK SUITE
    # -------------------------------------------------------------------------
    story.append(Paragraph("4. The Declarative Benchmark Suite (Phase 2.6)", h1_style))
    story.append(Paragraph(
        "To guarantee that future AI model updates or code refactorings never degrade compliance accuracy, "
        "the project includes an automated benchmark evaluation harness. Each test pair has a declarative JSON manifest "
        "(e.g., <code>data/benchmarks/pair_b_srs.json</code>) defining ground-truth evidence blocks and expected statuses.",
        body_style
    ))
    story.append(Paragraph(
        "The <b>Three-Layer Validator</b> (<code>src/benchmarks/manifest_validator.py</code>) validates all manifests:",
        body_style
    ))
    story.append(Paragraph("• <b>Layer A (JSON Schema):</b> Enforces JSON Schema Draft 2020-12 structural constraints and data types.", bullet_style))
    story.append(Paragraph("• <b>Layer B (Cross-Field Logic):</b> Enforces business invariants (mandatory_total >= 1, ID hierarchy, status conservation).", bullet_style))
    story.append(Paragraph("• <b>Layer C (Document Integrity):</b> Parses the physical PDFs, checks 64-char SHA-256 digests, and validates that every quoted evidence excerpt actually exists word-for-word in the target PDF.", bullet_style))
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # SECTION 5: STEP-BY-STEP GUIDE FOR THE NEW INTERN
    # -------------------------------------------------------------------------
    story.append(Paragraph("5. Step-by-Step Developer Guide: How to Run & Test", h1_style))
    story.append(Paragraph(
        "Here are the exact commands the incoming intern needs to get started, run tests, and execute audits:",
        body_style
    ))

    story.append(Paragraph("Step 1: Environment Setup", h2_style))
    step1_snippet = (
        "# 1. Clone repository\n"
        "git clone https://github.com/Shubh-4563/SRS---Compliance---analyzer.git\n"
        "cd SRS---Compliance---analyzer\n\n"
        "# 2. Create virtual environment (Python 3.10+ recommended, tested on 3.13)\n"
        "python3 -m venv .venv\n"
        "source .venv/bin/activate\n\n"
        "# 3. Install dependencies\n"
        "pip install -r requirements.txt"
    )
    story.append(Preformatted(step1_snippet, code_style))

    story.append(Paragraph("Step 2: Run the Full Test Suite", h2_style))
    step2_snippet = (
        "# Runs all 151 unit and regression tests across all subsystems:\n"
        "python3 -m unittest discover -s tests -p \"test_*.py\" -v\n\n"
        "# Expected Output: Ran 151 tests in ~25-30s ... OK"
    )
    story.append(Preformatted(step2_snippet, code_style))

    story.append(Paragraph("Step 3: Run the Live Pipeline on Pre-Built Test Pairs", h2_style))
    step3_snippet = (
        "# Evaluate Pair B (Equipment Monitoring Dashboard vs Technical Template):\n"
        "python3 main.py \\\n"
        "  --template data/test_packs/B_reference_technical_template.pdf \\\n"
        "  --target data/test_packs/B_target_technical_document.pdf \\\n"
        "  --format all\n\n"
        "# Output generated:\n"
        "# - Terminal ANSI summary\n"
        "# - compliance_report.md (formatted Markdown audit document)\n"
        "# - compliance_report.json (machine-readable data with exact quotes)"
    )
    story.append(Preformatted(step3_snippet, code_style))

    story.append(Paragraph("Step 4: Run the Automated Benchmark Accuracy Runner", h2_style))
    step4_snippet = (
        "# Evaluates accuracy across all declarative benchmark pairs:\n"
        "python3 scripts/evaluate_manifest_benchmark.py --all\n\n"
        "# Evaluates Hit@1..5, MRR, Accuracy (100%), and zero discrepancies"
    )
    story.append(Preformatted(step4_snippet, code_style))

    story.append(Paragraph("Step 5: Testing a New Document from Ma'am or Customer", h2_style))
    step5_snippet = (
        "# Place the reference checklist PDF and candidate document PDF in data/:\n"
        "python3 main.py \\\n"
        "  --template data/my_checklist.pdf \\\n"
        "  --target data/my_candidate_srs.pdf \\\n"
        "  --output-dir reports/my_audit/ \\\n"
        "  --format all"
    )
    story.append(Preformatted(step5_snippet, code_style))

    # -------------------------------------------------------------------------
    # SECTION 6: FUTURE DEVELOPMENT ROADMAP
    # -------------------------------------------------------------------------
    story.append(Paragraph("6. Future Development Roadmap for the Next Intern", h1_style))
    story.append(Paragraph(
        "Here are high-priority directions the next intern can take forward:",
        body_style
    ))
    story.append(Paragraph("1. <b>OCR Integration (Tesseract / EasyOCR):</b> Add support for scanned paper PDFs or photos of requirements documents (currently only digital PDFs with selectable text are supported).", bullet_style))
    story.append(Paragraph("2. <b>Large Document Scaling (Pair C & D):</b> Benchmark against multi-page industrial SRS documents (50+ pages) to stress-test hybrid retrieval candidate ranking.", bullet_style))
    story.append(Paragraph("3. <b>Interactive Web UI (Streamlit / FastHTML):</b> Build a drag-and-drop web dashboard where users can upload two PDFs and view interactive compliance findings.", bullet_style))
    story.append(Paragraph("4. <b>Model Optimization:</b> Test quantized or ONNX models (e.g. MiniLM-L6 ONNX) to decrease CPU inference latency for faster local audits.", bullet_style))
    story.append(Spacer(1, 14))

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E0"), spaceAfter=10))
    story.append(Paragraph(
        "<i>Document compiled for internship onboarding and technical handover. "
        "Source code and git history are synchronized at origin/main.</i>",
        callout_style
    ))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF guide at: {output_path}")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    target_pdf = out_dir / "AI_SRS_Compliance_Intern_Guide.pdf"
    build_pdf(target_pdf)

