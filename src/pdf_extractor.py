#!/usr/bin/env python3
"""
PDF Extraction Module - Version 0.4
====================================
Modular PDF parser for Software Requirements Specifications (SRS) and Reference Standards.
Extracts text streams from digital PDFs, parses structural sections, identifies rules,
and detects scanned / image-only documents where OCR is required.
"""

import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import pypdf

# Relative imports handled gracefully
from src.analyzer import (
    Requirement,
    DocumentSection,
    ReferenceSection,
    ReferenceRule,
    ReferenceStandard,
    RequirementExtractor,
)


class ScannedPDFError(ValueError):
    """
    Raised when a PDF contains little or no extractable text, indicating it
    is a scanned document or image-only file requiring OCR.
    """
    def __init__(
        self,
        filepath: str,
        char_count: int = 0,
        page_count: int = 0,
        message: Optional[str] = None
    ):
        self.filepath = filepath
        self.char_count = char_count
        self.page_count = page_count
        filename = os.path.basename(filepath)
        default_msg = (
            f"Scanned or image-only PDF detected ('{filename}'). "
            f"Text extraction returned only {char_count} character(s) across {page_count} page(s). "
            f"Optical Character Recognition (OCR) is required to process this document."
        )
        super().__init__(message or default_msg)


@dataclass
class PDFExtractionResult:
    """Detailed result of extracting text from a PDF file."""
    filepath: str
    text: str
    page_count: int
    char_count: int
    word_count: int
    pages_text: List[str] = field(default_factory=list)
    is_scanned: bool = False
    warning_message: Optional[str] = None


class PDFExtractor:
    """
    Modular engine for extracting text and domain entities from PDF documents:
    - Text extraction page-by-page using pypdf
    - Scanned/image-only PDF detection and graceful notification
    - SRS section and requirement extraction
    - Reference standard section and rule extraction
    """

    MIN_TOTAL_CHARS: int = 35
    MIN_WORDS_PER_PAGE: float = 3.0

    def __init__(
        self,
        min_total_chars: int = MIN_TOTAL_CHARS,
        min_words_per_page: float = MIN_WORDS_PER_PAGE
    ):
        self.min_total_chars = min_total_chars
        self.min_words_per_page = min_words_per_page

    def extract_text(
        self,
        filepath: str,
        raise_if_scanned: bool = True
    ) -> PDFExtractionResult:
        """
        Extracts raw text from a PDF document across all pages.
        Detects scanned/image-only documents when extractable text is negligible.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PDF file not found: '{filepath}'")

        try:
            reader = pypdf.PdfReader(filepath)
        except Exception as e:
            raise ValueError(f"Failed to read PDF file '{filepath}': {e}") from e

        pages_text: List[str] = []
        page_count = len(reader.pages)

        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            pages_text.append(page_text)

        raw_full_text = "\n".join(pages_text)
        non_ws_chars = len(re.sub(r'\s+', '', raw_full_text))
        total_words = len(raw_full_text.split())
        words_per_page = total_words / max(page_count, 1)

        is_scanned = (
            page_count > 0 and (
                non_ws_chars < self.min_total_chars or
                words_per_page < self.min_words_per_page
            )
        )

        warning_message = None
        if is_scanned:
            warning_message = (
                f"Optical Character Recognition (OCR) is required: "
                f"Extracted only {non_ws_chars} character(s) across {page_count} page(s)."
            )
            if raise_if_scanned:
                raise ScannedPDFError(
                    filepath=filepath,
                    char_count=non_ws_chars,
                    page_count=page_count,
                    message=warning_message
                )

        unwrapped_text = self.unwrap_text(raw_full_text)

        return PDFExtractionResult(
            filepath=filepath,
            text=unwrapped_text,
            page_count=page_count,
            char_count=non_ws_chars,
            word_count=total_words,
            pages_text=pages_text,
            is_scanned=is_scanned,
            warning_message=warning_message
        )

    def is_scanned_pdf(self, filepath: str) -> bool:
        """Checks if a PDF has little or no extractable text without throwing an exception."""
        try:
            result = self.extract_text(filepath, raise_if_scanned=False)
            return result.is_scanned
        except Exception:
            return True

    def unwrap_text(self, text: str) -> str:
        """
        Cleans and normalises PDF text by joining broken lines caused by
        fixed-width column layout / word-wrapping in PDF text rendering.

        Merging rules:
        - A continuation line (does NOT start a new section heading, bullet,
          or metadata label) is appended to the previous line when the previous
          line does NOT end with terminal punctuation (. : ? ! )).
        - Numbered section headings (e.g. "1. Scope", "SEC-1") and metadata
          labels ("Version:", "Binding Modal:") are hard paragraph boundaries
          and are NEVER appended to the preceding line.
        - Requirement lines (REQ-xxx: ...) are treated as body content:
          their continuations ARE joined back into them.
        """
        lines = [line.strip() for line in text.splitlines()]
        unwrapped: List[str] = []

        # Hard structural boundaries — start a fresh paragraph, cannot be targets of merges
        section_header_pattern = re.compile(
            r'^(?:'
            r'#+\s+'                             # Markdown headings
            r'|\d+(?:\.\d+)*[\.\ :]\s+[A-Z]'  # "1. Title", "2.3. Overview"
            r'|Section\s+\d+'                    # "Section 3"
            r'|SEC[-_]\d+'                       # "SEC-1"
            r'|[-*\u2022]\s+'                    # Bullet points
            r'|\bRule\s+(?:R\d+|\d+)'           # "Rule R1"
            r'|R[1-4][:\.\s]'                    # "R1: ..."
            r'|RULE-[0-9]+'                       # "RULE-001"
            r'|[A-Z]\.\s+'                       # "A. Section"
            r')',
            re.IGNORECASE
        )

        # Metadata/label lines — also hard paragraph boundaries
        label_line_pattern = re.compile(
            r'^(?:Standard|Title|Version|Binding|Prohibited|Vague)\b',
            re.IGNORECASE
        )

        for line in lines:
            if not line:
                if unwrapped and unwrapped[-1] != "":
                    unwrapped.append("")
                continue

            is_section_header = bool(section_header_pattern.match(line))
            is_label = bool(label_line_pattern.match(line))

            # Requirement-tagged lines always start a fresh paragraph (never appended to previous)
            is_req_line = bool(re.match(r'^(?:REQ|SRS|SYS)[-_][A-Za-z0-9_-]+[:\.]', line, re.IGNORECASE))
            # New-paragraph lines start fresh and are never appended to the previous line
            is_new_paragraph = is_section_header or is_label or is_req_line

            if unwrapped and unwrapped[-1] and not is_new_paragraph:
                prev = unwrapped[-1]
                prev_blocks_merge = (
                    bool(section_header_pattern.match(prev)) or
                    bool(label_line_pattern.match(prev))
                )
                # Merge into previous line if previous is not a hard boundary
                # and previous line did not terminate with sentence punctuation.
                # NOTE: REQ-xxx lines are NOT treated as hard boundaries here,
                # so their wrapped continuation lines are correctly rejoined.
                if not prev_blocks_merge and not (
                    prev.endswith('.') or prev.endswith(':') or
                    prev.endswith('?') or prev.endswith('!') or
                    prev.endswith(')')):
                    unwrapped[-1] = f"{prev} {line}"
                    continue

            unwrapped.append(line)

        return "\n".join(unwrapped)

    def extract_srs_document(
        self,
        filepath: str
    ) -> Tuple[List[DocumentSection], List[Requirement]]:
        """
        Extracts structural sections and individual requirement items from an SRS PDF.
        Returns:
            Tuple of (sections, requirements)
        """
        result = self.extract_text(filepath, raise_if_scanned=True)
        extractor = RequirementExtractor()
        sections = extractor.extract_sections_from_text(result.text)
        requirements = extractor.extract_from_text(result.text)
        return sections, requirements

    def extract_reference_standard(self, filepath: str) -> ReferenceStandard:
        """
        Parses reference standard document sections and rules (R1 - R4) from a PDF.
        """
        result = self.extract_text(filepath, raise_if_scanned=True)
        text = result.text

        # 1. Initialize base standard with defaults
        standard = ReferenceStandard()

        # 2. Extract standard title
        title_match = re.search(
            r'(?:Reference Standard|Standard Name|Standard Title)[:\s]+([^\n\|]+)',
            text,
            re.IGNORECASE
        )
        if title_match:
            standard.standard_name = title_match.group(1).strip()

        # 3. Extract version
        version_match = re.search(
            r'(?:Standard Version|Version)[:\s]+([0-9\.]+)',
            text,
            re.IGNORECASE
        )
        if version_match:
            standard.version = version_match.group(1).strip()

        # 4. Extract sections
        sec_pattern = re.compile(
            r'^(SEC-\d+|Section\s+\d+)[:\.\s]+([^\n\-\:–]+)(?:[\-\:–]\s*([^\n]+))?',
            re.MULTILINE | re.IGNORECASE
        )
        extracted_sections: List[ReferenceSection] = []
        for m in sec_pattern.finditer(text):
            sec_id = m.group(1).strip().upper()
            if sec_id.startswith("SECTION"):
                sec_num = re.sub(r'[^\d]', '', sec_id)
                sec_id = f"SEC-{sec_num}"
            name = m.group(2).strip()
            desc = m.group(3).strip() if m.group(3) else ""
            extracted_sections.append(ReferenceSection(section_id=sec_id, name=name, description=desc))

        if extracted_sections:
            sec_map = {s.section_id: s for s in standard.sections}
            for s in extracted_sections:
                sec_map[s.section_id] = s
            standard.sections = list(sec_map.values())

        # 5. Extract rules (R1 - R4 or custom rules)
        rule_pattern = re.compile(
            r'^(R[1-4]|RULE-[0-9]+|Rule\s+[0-9]+)[:\.\s]+([^\n\-\:–]+)(?:[\-\:–]\s*([^\n]+))?',
            re.MULTILINE | re.IGNORECASE
        )
        extracted_rules: List[ReferenceRule] = []
        for m in rule_pattern.finditer(text):
            r_id = m.group(1).strip().upper()
            if r_id.startswith("RULE"):
                r_num = re.sub(r'[^\d]', '', r_id)
                r_id = f"R{r_num}"
            name = m.group(2).strip()
            desc = m.group(3).strip() if m.group(3) else ""
            extracted_rules.append(ReferenceRule(rule_id=r_id, name=name, description=desc))

        if extracted_rules:
            rule_map = {r.rule_id: r for r in standard.rules}
            for r in extracted_rules:
                rule_map[r.rule_id] = r
            standard.rules = list(rule_map.values())

        # 6. Extract parameters (binding modal, vague words)
        modal_match = re.search(
            r'(?:Binding Modal Verb|Binding Modal|Mandatory Modal)[:\s]+([A-Za-z]+)',
            text,
            re.IGNORECASE
        )
        if modal_match:
            standard.binding_modal = modal_match.group(1).strip().lower()

        vague_match = re.search(
            r'^(?:Prohibited\s+Vague\s+Words|Prohibited\s+Words|Vague\s+Words)[:\s]+([^\n]+)',
            text,
            re.MULTILINE | re.IGNORECASE
        )
        if vague_match:
            words_raw = vague_match.group(1).strip()
            words = [w.strip().strip("'\"").lower() for w in re.split(r'[,;]+', words_raw) if w.strip()]
            if words:
                standard.vague_words = words

        return standard


# Module-level convenience functions
def extract_text_from_pdf(filepath: str) -> str:
    """Convenience function to extract text from a PDF file."""
    return PDFExtractor().extract_text(filepath).text


def extract_srs_from_pdf(
    filepath: str
) -> Tuple[List[DocumentSection], List[Requirement]]:
    """Convenience function to extract SRS sections and requirements from a PDF file."""
    return PDFExtractor().extract_srs_document(filepath)


def extract_standard_from_pdf(filepath: str) -> ReferenceStandard:
    """Convenience function to parse a ReferenceStandard from a PDF file."""
    return PDFExtractor().extract_reference_standard(filepath)
