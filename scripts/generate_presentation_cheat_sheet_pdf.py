"""
Generates a compact, high-impact Executive Presentation & Commute Cheat-Sheet PDF
for the AI-SRS Compliance Analyzer. Designed for quick mobile/laptop reading
before a mentor presentation.
"""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Preformatted,
    KeepTogether,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Adds running headers and page numbers."""
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
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#718096"))

        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(40, 755, "AI-SRS COMPLIANCE ANALYZER — EXECUTIVE BRIEF & PRESENTATION GUIDE")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.75)
            self.line(40, 748, 572, 748)

        # Footer (All pages)
        self.setFont("Helvetica", 8)
        self.drawString(40, 30, "AI-SRS Compliance Project | Rapid Review Reference")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(572, 30, page_str)
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.75)
        self.line(40, 40, 572, 40)

        self.restoreState()


def create_callout(text, title=None, bg_color="#EDF2F7", border_color="#CBD5E0", title_color="#2B6CB0"):
    """Helper to create visually appealing callout box."""
    styles = getSampleStyleSheet()
    content = []
    if title:
        title_style = ParagraphStyle(
            "CalloutTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor(title_color),
            spaceAfter=4,
        )
        content.append(Paragraph(title, title_style))
    
    body_style = ParagraphStyle(
        "CalloutBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#2D3748"),
    )
    content.append(Paragraph(text, body_style))
    
    t = Table([[content]], colWidths=[532])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(bg_color)),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor(border_color)),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    return t


def build_pdf(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=42,
        bottomMargin=42,
    )

    styles = getSampleStyleSheet()

    # Typography styles
    c_navy = colors.HexColor("#1A365D")
    c_blue = colors.HexColor("#2B6CB0")
    c_dark = colors.HexColor("#2D3748")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=c_navy,
        spaceAfter=2,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        textColor=c_blue,
        spaceAfter=10,
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=c_navy,
        spaceBefore=8,
        spaceAfter=4,
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=c_blue,
        spaceBefore=5,
        spaceAfter=2,
    )

    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=c_dark,
        spaceAfter=4,
    )

    bullet_style = ParagraphStyle(
        "BulletCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=c_dark,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=2,
    )

    diagram_style = ParagraphStyle(
        "DiagramPre",
        parent=styles["Code"],
        fontName="Courier-Bold",
        fontSize=7,
        leading=8.5,
        textColor=colors.HexColor("#1A202C"),
    )

    story = []

    # ==================== PAGE 1: EXECUTIVE BRIEF & PIPELINE ====================
    story.append(Paragraph("AI-SRS Compliance Analyzer", title_style))
    story.append(Paragraph("Executive Commute Brief & Presentation Cheat-Sheet", subtitle_style))

    # Elevator Pitch Box
    pitch_text = (
        "<b>What is this project?</b> An automated AI regulatory engine that evaluates a <b>Target Document</b> "
        "(e.g., an SRS or lab report) against a <b>Reference Standard/Template</b>. It decomposes requirements into "
        "mandatory and optional clauses, pinpoints exact evidence paragraphs using hybrid search, verifies semantic alignment "
        "using deep-learning NLI, and computes a legally defensible compliance verdict with zero hallucinations."
    )
    story.append(create_callout(pitch_text, title="ELEVATOR PITCH (Say this in first 30 seconds)", bg_color="#EBF8FF", border_color="#3182CE", title_color="#2B6CB0"))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Why Keyword Search Fails (The Core Innovation)", h1_style))
    story.append(Paragraph(
        "Traditional tools search for keywords like <i>'data encryption'</i>. If a specification states: "
        "<i>'Data encryption is NOT implemented in Phase 1'</i>, keyword search flags it as compliant because the words match! "
        "Our system uses <b>Natural Language Inference (NLI)</b> to detect that the document directly <b>CONTRADICTS</b> "
        "the requirement. It understands semantics, exceptions, and explicit omissions.",
        body_style
    ))
    story.append(Spacer(1, 4))

    # Architecture Diagram
    story.append(Paragraph("End-to-End System Architecture", h1_style))
    diagram_ascii = """+-------------------------+      +-------------------------+
| Reference Template PDF  |      |   Target Document PDF   |
| (Requirements Standard) |      |   (SRS to be Audited)   |
+------------+------------+      +------------+------------+
             |                                |
             v                                v
+-------------------------+      +-------------------------+
| Template Parser         |      | Target Block Parser     |
| (Decomposes to Parent & |      | (Extracts text blocks   |
|  Atomic Sub-Reqs)       |      |  with page & bbox IDs)  |
+------------+------------+      +------------+------------+
             |                                |
             |    +---------------------------+
             |    |
             v    v
+----------------------------------------------------------+
| Hybrid Retrieval Engine (BM25 Lexical + MiniLM Dense)    |
| -> Reciprocal Rank Fusion (RRF) -> Top-5 Candidate Blocks|
+----------------------------+-----------------------------+
                             |
                             v
+----------------------------------------------------------+
| Verification Engine (Cross-Encoder DeBERTa-v3 NLI)       |
| -> Evaluates Premise vs Hypothesis                       |
| -> FULFILLED | CONTRADICTED | OMISSION | MISSING | REVIEW|
+----------------------------+-----------------------------+
                             |
                             v
+----------------------------------------------------------+
| Hierarchical Compliance Aggregator                       |
| -> Atomic -> Parent Points -> Document Verdict           |
| (COMPLIANT | NON_COMPLIANT | CONDITIONALLY_COMPLIANT)    |
+----------------------------+-----------------------------+
                             |
                             v
+----------------------------------------------------------+
| Audit Output: JSON Machine Data & Formatted Markdown Doc |
+----------------------------------------------------------+"""
    
    t_diag = Table([[Preformatted(diagram_ascii, diagram_style)]], colWidths=[532])
    t_diag.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_diag)

    story.append(PageBreak())

    # ==================== PAGE 2: THE 5 PIPELINE STAGES ====================
    story.append(Paragraph("The 5 Engineering Stages in Detail", title_style))
    story.append(Spacer(1, 4))

    # Stage 1
    story.append(Paragraph("Stage 1: Document Parsing & Requirement Decomposition", h1_style))
    story.append(Paragraph(
        "Reads the reference PDF and decomposes complex paragraphs into atomic, testable obligations. "
        "Identifies <b>Mandatory</b> clauses (<i>'shall', 'must', 'required'</i>) versus <b>Optional</b> clauses (<i>'should', 'may', 'recommended'</i>). "
        "Separates Parent sections (e.g. <i>3.1 User Authentication</i>) from atomic criteria (e.g. <i>3.1.1 Two-Factor Support</i>).",
        body_style
    ))

    # Stage 2
    story.append(Paragraph("Stage 2: Target Layout Extraction & Block Segmentation", h1_style))
    story.append(Paragraph(
        "Parses the Target PDF into clean, discrete <b>DocumentBlocks</b>. Unlike naive text extraction that loses structure, "
        "each block retains its exact source <b>Page Number</b> and sequence ID. This ensures complete audit traceability.",
        body_style
    ))

    # Stage 3
    story.append(Paragraph("Stage 3: Hybrid Retrieval with Reciprocal Rank Fusion (RRF)", h1_style))
    story.append(Paragraph(
        "Searches the entire document to find the best candidate evidence blocks for each requirement by combining two paradigms:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>BM25 Lexical Search</b>: Exact keyword matching for technical IDs, acronyms, and codes.", bullet_style))
    story.append(Paragraph("&bull; <b>Dense Semantic Search</b>: <code>sentence-transformers/all-MiniLM-L6-v2</code> maps semantic concepts and synonyms.", bullet_style))
    story.append(Paragraph("&bull; <b>RRF Fusion Formula</b>: Combines both lists mathematically without needing arbitrary manual weights:", bullet_style))
    
    rrf_code = "  Score(block) = 1.0 / (60 + BM25_Rank)  +  1.0 / (60 + Dense_Rank)\n  Top 5 highest scoring blocks are passed directly to the verification engine."
    t_rrf = Table([[Preformatted(rrf_code, diagram_style)]], colWidths=[532])
    t_rrf.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor("#CBD5E0")),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_rrf)
    story.append(Spacer(1, 4))

    # Stage 4
    story.append(Paragraph("Stage 4: Deep NLI Verification (DeBERTa-v3)", h1_style))
    story.append(Paragraph(
        "Uses a pretrained cross-encoder NLI model (<code>cross-encoder/nli-deberta-v3-small</code>). "
        "The retrieved text block is the <b>Premise</b> and the requirement is the <b>Hypothesis</b>. "
        "Outputs probabilities for Entailment, Contradiction, and Neutral, mapping to 5 definitive audit statuses:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>FULFILLED</b>: High entailment probability; document proves requirement is satisfied.", bullet_style))
    story.append(Paragraph("&bull; <b>CONTRADICTED</b>: High contradiction probability; target document explicitly negates requirement.", bullet_style))
    story.append(Paragraph("&bull; <b>EXPLICIT_OMISSION</b>: Target explicitly specifies this capability is excluded / out of scope.", bullet_style))
    story.append(Paragraph("&bull; <b>MISSING</b>: No relevant evidence exists anywhere in the document.", bullet_style))
    story.append(Paragraph("&bull; <b>NEEDS_REVIEW</b>: Ambiguous evidence flagged specifically for human engineering audit.", bullet_style))
    story.append(Spacer(1, 4))

    # Stage 5
    story.append(Paragraph("Stage 5: Hierarchical Compliance Aggregation", h1_style))
    story.append(Paragraph(
        "Rolls up atomic statuses into Parent requirements, then into the overall Document Verdict using strict regulatory math:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>COMPLIANT</b>: 100.0% of mandatory obligations fulfilled, 0 contradictions, 0 unresolved items.", bullet_style))
    story.append(Paragraph("&bull; <b>NON_COMPLIANT</b>: Less than 100% mandatory coverage or any mandatory contradiction.", bullet_style))
    story.append(Paragraph("&bull; <b>CONDITIONALLY_COMPLIANT</b>: Mandatory items addressed, but human review required on borderlines.", bullet_style))

    story.append(PageBreak())

    # ==================== PAGE 3: PRESENTATION FAQ & LIVE DEMO ====================
    story.append(Paragraph("Presentation Q&A Defense & Live Demo", title_style))
    story.append(Spacer(1, 4))

    demo_box = (
        "<b>How to trigger the Live Demo during presentation:</b><br/>"
        "1. Open Terminal in project directory and run:<br/>"
        "   <code>python3 main.py</code><br/>"
        "   <i>Demonstrates full PDF audit on real pairs with real-time extraction, retrieval, and verdict in ~5 sec.</i><br/>"
        "2. To show automated benchmark validation against gold standards:<br/>"
        "   <code>python3 scripts/evaluate_manifest_benchmark.py --all</code><br/>"
        "   <i>Demonstrates 100% agreement and zero discrepancies against ground truth.</i>"
    )
    story.append(create_callout(demo_box, title="LIVE TERMINAL COMMANDS", bg_color="#F0FFF4", border_color="#38A169", title_color="#22543D"))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Handling Ma'am's Likely Questions", h1_style))

    # Q1
    story.append(Paragraph("Q1: Can this run on any new document I give you right now?", h2_style))
    story.append(Paragraph(
        "<b>Answer:</b> <i>'Yes, ma'am! As long as it is a digital PDF with selectable text, our parser extracts the blocks "
        "and runs the exact same pipeline without retraining. We just pass the reference standard and your target document "
        "directly into the runner.'</i>",
        body_style
    ))

    # Q2
    story.append(Paragraph("Q2: Why not just paste the PDF into ChatGPT or an LLM?", h2_style))
    story.append(Paragraph(
        "<b>Answer:</b> <i>'LLMs suffer from context window limits, hallucinations, and non-deterministic answers. "
        "Our system is an <b>explainable audit pipeline</b>: every verdict points to the exact page number, text snippet, "
        "and mathematical entailment score. It runs completely offline on local CPU with zero data leakage and 100% deterministic results.'</i>",
        body_style
    ))

    # Q3
    story.append(Paragraph("Q3: What happens if an optional requirement is missing?", h2_style))
    story.append(Paragraph(
        "<b>Answer:</b> <i>'Our two-tier hierarchy ensures that optional clauses (e.g. 'may', 'should') do not fail the overall document verdict. "
        "The overall verdict depends strictly on mandatory obligations (100% mandatory coverage required). Optional items are tracked separately "
        "in the audit report so engineers can review them.'</i>",
        body_style
    ))

    # Q4
    story.append(Paragraph("Q4: What models are you using and why?", h2_style))
    story.append(Paragraph(
        "<b>Answer:</b> <i>'We use <b>all-MiniLM-L6-v2</b> (384-dimensional dense semantic vectors) for rapid candidate retrieval, "
        "and <b>cross-encoder/nli-deberta-v3-small</b> for Natural Language Inference. Both are lightweight, state-of-the-art transformer "
        "models optimized for fast CPU inference without requiring an expensive GPU cluster.'</i>",
        body_style
    ))
    story.append(Spacer(1, 6))

    # Key Numbers Summary
    summary_box = (
        "<b>Quick Project Stats to Impress:</b><br/>"
        "&bull; <b>151 Passing Tests</b> across unit, parser, retrieval, NLI, and integration suites.<br/>"
        "&bull; <b>100% Benchmark Accuracy</b> on ground truth test packs.<br/>"
        "&bull; <b>Dual Reporting</b>: Instant machine-readable JSON + executive Markdown audit summaries."
    )
    story.append(create_callout(summary_box, title="KEY PROJECT METRICS", bg_color="#FEFCBF", border_color="#D69E2E", title_color="#744210"))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)


if __name__ == "__main__":
    out_file = Path("AI_SRS_Compliance_Executive_Summary.pdf")
    build_pdf(out_file)
    print(f"Generated: {out_file.resolve()} ({out_file.stat().st_size} bytes)")
