"""
Generates a practical, step-by-step Quick Reference PDF Guide on:
'How to Run the AI-SRS Compliance Analyzer on New Office PDFs'
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
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#718096"))

        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(40, 755, "HOW TO RUN THE AI-SRS ANALYZER ON NEW PDFs — FIELD GUIDE")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.75)
            self.line(40, 748, 572, 748)

        # Footer (All pages)
        self.setFont("Helvetica", 8)
        self.drawString(40, 30, "AI-SRS Compliance Project | New Document Audit Guide")
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

    code_style = ParagraphStyle(
        "CodeStyle",
        parent=styles["Code"],
        fontName="Courier-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1A202C"),
    )

    story = []

    # ==================== PAGE 1: PRE-FLIGHT CHECK & SCENARIO A ====================
    story.append(Paragraph("AI-SRS Compliance Analyzer", title_style))
    story.append(Paragraph("Operator Field Guide: Auditing New Office PDFs", subtitle_style))

    # Pre-Flight Card
    preflight_text = (
        "<b>Before testing any new PDF, verify these 2 quick conditions:</b><br/>"
        "1. <b>Selectable Text</b>: Open the PDF in Preview or Acrobat and try to highlight a sentence. If you can copy-paste text, it is a digital PDF and can be audited immediately.<br/>"
        "2. <b>No Scanned Images</b>: If the PDF is a scanned photo of paper without selectable text, the text layer is missing and requires OCR pre-processing."
    )
    story.append(create_callout(preflight_text, title="30-SECOND PRE-FLIGHT CHECK", bg_color="#EBF8FF", border_color="#3182CE", title_color="#2B6CB0"))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Scenario 1: Ma'am gives you a Target PDF to audit", h1_style))
    story.append(Paragraph(
        "This is the most common office scenario. Your mentor gives you an SRS, lab report, or design document and asks: "
        "<i>'Can your system audit this document right now?'</i>",
        body_style
    ))
    story.append(Paragraph(
        "You can audit her target document against one of our existing reference standards already in the project:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>For Software Requirements Specifications (SRS):</b> Use <code>data/test_packs/pair_b_srs/reference_template.pdf</code>", bullet_style))
    story.append(Paragraph("&bull; <b>For Laboratory / Engineering Reports:</b> Use <code>data/test_packs/pair_a_lab_report/reference_template.pdf</code>", bullet_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Terminal Command for Scenario 1:", h2_style))
    cmd_1 = (
        "# Audit a new SRS document using the existing IEEE SRS reference standard:\n"
        "python3 main.py \\\n"
        "  --reference data/test_packs/pair_b_srs/reference_template.pdf \\\n"
        "  --target /path/to/maams_target_document.pdf"
    )
    t_cmd1 = Table([[Preformatted(cmd_1, code_style)]], colWidths=[532])
    t_cmd1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_cmd1)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Scenario 2: Ma'am gives you BOTH a Reference Standard & Target PDF", h1_style))
    story.append(Paragraph(
        "If she has a custom rubric or guideline document (e.g. <i>Project_Rubric.pdf</i>) and a student document (<i>Submission.pdf</i>):",
        body_style
    ))
    cmd_2 = (
        "python3 main.py \\\n"
        "  --reference /path/to/rubric_standard.pdf \\\n"
        "  --target /path/to/student_submission.pdf"
    )
    t_cmd2 = Table([[Preformatted(cmd_2, code_style)]], colWidths=[532])
    t_cmd2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_cmd2)

    story.append(PageBreak())

    # ==================== PAGE 2: ADVANCED CLI OPTIONS & OUTPUT ====================
    story.append(Paragraph("Targeting Specific Pages & Custom Output Paths", title_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Scenario 3: Auditing a Single Page of a Large Document", h1_style))
    story.append(Paragraph(
        "If the document is 50 pages long and Ma'am asks you to verify section 4 on Page 12 specifically, "
        "use the <code>--page</code> flag to run a focused audit in under 2 seconds:",
        body_style
    ))
    cmd_3 = (
        "python3 main.py \\\n"
        "  --reference data/test_packs/pair_b_srs/reference_template.pdf \\\n"
        "  --target /path/to/large_document.pdf \\\n"
        "  --page 12"
    )
    t_cmd3 = Table([[Preformatted(cmd_3, code_style)]], colWidths=[532])
    t_cmd3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_cmd3)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Exporting Reports to Custom Paths", h1_style))
    story.append(Paragraph(
        "By default, results are printed to the console and saved in <code>reports/audit_report.json</code> "
        "and <code>reports/audit_report.md</code>. You can specify custom output filenames:",
        body_style
    ))
    cmd_custom = (
        "python3 main.py \\\n"
        "  --reference data/test_packs/pair_b_srs/reference_template.pdf \\\n"
        "  --target /path/to/target.pdf \\\n"
        "  --json-out reports/maam_eval_result.json \\\n"
        "  --md-out reports/maam_eval_result.md"
    )
    t_cmd_custom = Table([[Preformatted(cmd_custom, code_style)]], colWidths=[532])
    t_cmd_custom.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_cmd_custom)
    story.append(Spacer(1, 8))

    story.append(Paragraph("How to Interpret the Console Output in Real Time", h1_style))
    story.append(Paragraph(
        "When the command finishes, the console immediately outputs a summary table. Here is what to explain:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Mandatory Coverage (%)</b>: Percentage of mandatory requirements with verified evidence.", bullet_style))
    story.append(Paragraph("&bull; <b>Parent Points Breakdown</b>: Shows which sections are Fully Compliant vs. Partially Compliant.", bullet_style))
    story.append(Paragraph("&bull; <b>Overall Verdict</b>: <code>COMPLIANT</code> (100% mandatory satisfied), <code>NON_COMPLIANT</code> (mandatory item missing or contradicted), or <code>CONDITIONALLY_COMPLIANT</code> (needs human review).", bullet_style))
    story.append(Paragraph("&bull; <b>Human Review Queue</b>: If any requirement had borderline confidence scores, it lists the exact requirement ID and reason so a human engineer can review it.", bullet_style))

    story.append(PageBreak())

    # ==================== PAGE 3: TROUBLESHOOTING & OFFICE TIPS ====================
    story.append(Paragraph("Troubleshooting & Live Demo Tips", title_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Handling File Paths with Spaces", h1_style))
    story.append(Paragraph(
        "In macOS, file paths often contain spaces (e.g., <i>Google Drive/Final SRS Report.pdf</i>). "
        "Always enclose the path in quotes or drag-and-drop the file into the terminal window:",
        body_style
    ))
    cmd_quotes = (
        '# Correct: Using quotes around paths with spaces\n'
        'python3 main.py \\\n'
        '  --reference "data/test_packs/pair_b_srs/reference_template.pdf" \\\n'
        '  --target "/Users/shubhamsharma/Downloads/Maam Final SRS.pdf"'
    )
    t_cmd_quotes = Table([[Preformatted(cmd_quotes, code_style)]], colWidths=[532])
    t_cmd_quotes.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_cmd_quotes)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Common Errors & Instant Fixes", h1_style))

    # Error 1
    story.append(Paragraph("Error: 'Target document has only X characters, minimum required is 50'", h2_style))
    story.append(Paragraph(
        "<b>Cause:</b> The PDF has no digital text layer (it is an image/scanned document) or is an empty page.<br/>"
        "<b>Fix:</b> Open the PDF in Preview, choose <i>Export as PDF</i> with text, or test a different page with text.",
        body_style
    ))

    # Error 2
    story.append(Paragraph("Error: 'FileNotFoundError: No such file or directory'", h2_style))
    story.append(Paragraph(
        "<b>Cause:</b> Typo in the file path or running the command from the wrong folder.<br/>"
        "<b>Fix:</b> Make sure you are in the project folder (<code>cd '/Users/shubhamsharma/Downloads/AI - SRS - Compliance '</code>) "
        "and drag the target PDF from Finder directly into Terminal to paste its exact path.",
        body_style
    ))
    story.append(Spacer(1, 8))

    # Pro-tip Box
    pro_tip = (
        "<b>Pro Tip for Presenting to Ma'am:</b><br/>"
        "Before running a live test on a new PDF in front of her, first run the proven benchmark pair:<br/>"
        "<code>python3 main.py</code><br/>"
        "This warms up the local model cache and shows her a flawless run. Then say:<br/>"
        "<i>'Now let us try it on your new document!'</i> and run the custom command."
    )
    story.append(create_callout(pro_tip, title="PRO PRESENTATION TIP", bg_color="#FEFCBF", border_color="#D69E2E", title_color="#744210"))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)


if __name__ == "__main__":
    out_file = Path("How_To_Test_New_PDFs_Guide.pdf")
    build_pdf(out_file)
    print(f"Generated: {out_file.resolve()} ({out_file.stat().st_size} bytes)")

