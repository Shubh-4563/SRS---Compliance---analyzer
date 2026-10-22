#!/usr/bin/env python3
"""
Generate sample PDF test files for Version 0.4:
1. data/sample_srs.pdf (Digital text SRS matching sample_srs.txt)
2. data/avionics_sample_srs.pdf (Multi-section avionics SRS matching avionics_sample_srs.md)
3. data/reference_standard.pdf (Digital text reference standard matching reference_standard.json)
4. data/scanned_sample_srs.pdf (Simulated scanned/image-only PDF with zero extractable text)
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def create_sample_srs_pdf():
    pdf_path = os.path.join(DATA_DIR, "sample_srs.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 730, "Software Requirements Specification (SRS)")
    c.setFont("Helvetica", 10)
    c.drawString(72, 715, "Baseline Sample - Avionics Software Specification v0.4")
    c.setStrokeColor(colors.gray)
    c.line(72, 705, 540, 705)

    requirements = [
        "REQ-001: The system shall display the aircraft altitude within 0.5 seconds.",
        "REQ-002: The system shall display altitude quickly.",
        "REQ-003: The system shall process incoming sensor data.",
        "REQ-004: The system shall provide a user-friendly interface.",
        "REQ-005: The system shall record the pilot login attempt and store the timestamp."
    ]

    c.setFont("Helvetica", 11)
    y = 670
    for req in requirements:
        c.drawString(72, y, req)
        y -= 30

    c.save()
    print(f"Generated: {pdf_path}")


def create_avionics_srs_pdf():
    pdf_path = os.path.join(DATA_DIR, "avionics_sample_srs.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Page 1
    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, 730, "Avionics Flight Display Management (FDM) SRS")
    c.setFont("Helvetica", 10)
    c.drawString(72, 715, "Standard: DO-178C Level B Compliant Draft")
    c.setStrokeColor(colors.gray)
    c.line(72, 705, 540, 705)

    y = 675
    sections_and_reqs = [
        ("HEADER", "1. Scope and System Overview"),
        ("TEXT", "This document defines the primary flight display and avionics sensor telemetry requirements."),
        ("HEADER", "2. Functional Requirements"),
        ("REQ", "REQ-FDM-001: The flight display unit shall refresh the barometric altitude reading at a frequency of at least 20 Hz with a maximum end-to-end latency of 50 ms."),
        ("REQ", "REQ-FDM-002: The system should update the ground speed display quickly whenever wind gust changes occur."),
        ("REQ", "REQ-FDM-003: The autopilot mode indicator shall display current navigation state, and the unit shall transmit GPS coordinates to the ground control station and emit an audio alarm if link latency exceeds 500 ms."),
        ("REQ", "The primary flight display shall present pitch and roll attitude within +/- 0.5 degrees accuracy."),
        ("REQ", "REQ-FDM-004: If the primary Pitot-static airspeed sensor reports an electrical fault or timeout exceeding 100 ms, the software shall transition into secondary synthetic airspeed estimation mode and flag a Level-2 caution on the annunciator."),
        ("HEADER", "3. Interface Requirements"),
        ("REQ", "REQ-INT-001: The software shall communicate with the Air Data Computer across the ARINC 429 dual-redundant bus at a baud rate of 100 kbps."),
        ("REQ", "REQ-INT-002: The pilot UI shall be modern, intuitive, robust, and easy to use during high-turbulence maneuvers.")
    ]

    for item_type, content in sections_and_reqs:
        if item_type == "HEADER":
            y -= 15
            c.setFont("Helvetica-Bold", 14)
            c.drawString(72, y, content)
            y -= 22
        elif item_type == "TEXT":
            c.setFont("Helvetica-Oblique", 10)
            c.drawString(72, y, content)
            y -= 20
        elif item_type == "REQ":
            c.setFont("Helvetica", 10)
            # Wrap if long
            if len(content) > 85:
                part1 = content[:85]
                part2 = content[85:]
                c.drawString(72, y, part1)
                y -= 14
                c.drawString(90, y, part2)
                y -= 18
            else:
                c.drawString(72, y, content)
                y -= 22

    c.save()
    print(f"Generated: {pdf_path}")


def create_reference_standard_pdf():
    pdf_path = os.path.join(DATA_DIR, "reference_standard.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, 730, "Reference Standard: SRS Quality & Compliance Specification")
    c.setFont("Helvetica", 10)
    c.drawString(72, 715, "Standard Version: 0.4 | Regulatory Baseline: DO-178C / IEEE 830")
    c.setStrokeColor(colors.gray)
    c.line(72, 705, 540, 705)

    y = 675
    c.setFont("Helvetica-Bold", 13)
    c.drawString(72, y, "A. Mandatory Specification Sections")
    y -= 22

    sections = [
        "SEC-1: Scope and System Overview - High-level description of system purpose, operational scope, and subsystem boundaries.",
        "SEC-2: Functional Requirements - Functional capabilities, actions, transformations, and system behavioral requirements.",
        "SEC-3: Interface Requirements - Hardware, software, bus, communications protocol, and user interface specifications.",
        "SEC-4: Safety and Failure Handling - Fault detection, exception handling, containment modes, and fail-safe recovery procedures."
    ]

    c.setFont("Helvetica", 10)
    for s in sections:
        c.drawString(85, y, s)
        y -= 20

    y -= 15
    c.setFont("Helvetica-Bold", 13)
    c.drawString(72, y, "B. Quality Verification Rules (R1 - R4)")
    y -= 22

    rules = [
        "R1: Unique Requirement ID - Every requirement must have a unique identifier (e.g., REQ-001).",
        "R2: Clear Behaviour - The requirement must describe what the system shall do using clear binding normative language.",
        "R3: Measurable Criteria - The requirement should contain measurable/verifiable criteria where applicable.",
        "R4: Avoid Vague Language - Detect vague words that introduce ambiguity into specifications."
    ]

    c.setFont("Helvetica", 10)
    for r in rules:
        c.drawString(85, y, r)
        y -= 20

    y -= 15
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y, "C. Regulatory Parameters & Lexicons")
    y -= 18
    c.setFont("Helvetica", 10)
    c.drawString(85, y, "Binding Modal Verb: shall")
    y -= 16
    c.drawString(85, y, "Prohibited Vague Words: quickly, fast, easy, adequate, user-friendly")

    c.save()
    print(f"Generated: {pdf_path}")


def create_scanned_srs_pdf():
    """Creates a simulated scanned/image-only PDF with no extractable text stream."""
    pdf_path = os.path.join(DATA_DIR, "scanned_sample_srs.pdf")
    img_path = os.path.join(DATA_DIR, "temp_scanned_page.png")

    # Render a raster image of scanned text
    img = Image.new("RGB", (612, 792), color=(248, 248, 248))
    draw = ImageDraw.Draw(img)
    # Draw simulated scanned page border and noisy lines
    draw.rectangle([40, 40, 572, 752], outline=(200, 200, 200), width=2)
    draw.line([60, 80, 500, 80], fill=(120, 120, 120), width=3)
    draw.line([60, 110, 420, 110], fill=(150, 150, 150), width=2)
    draw.line([60, 140, 460, 140], fill=(150, 150, 150), width=2)
    img.save(img_path)

    # Embed raster image onto PDF canvas with NO font or text operations
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawImage(img_path, 0, 0, width=612, height=792)
    c.save()

    if os.path.exists(img_path):
        os.remove(img_path)
    print(f"Generated (Scanned): {pdf_path}")


if __name__ == "__main__":
    create_sample_srs_pdf()
    create_avionics_srs_pdf()
    create_reference_standard_pdf()
    create_scanned_srs_pdf()
    print("All sample PDF files generated successfully.")
