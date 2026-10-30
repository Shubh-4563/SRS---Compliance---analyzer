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
    TemplatePoint,
    ReferenceTemplate,
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


class CandidateChunk(str):
    """
    Candidate text comparison chunk representing a single sentence or
    a paragraph-constrained sliding window of adjacent sentences.
    Inherits from str for complete backward compatibility.
    """
    def __new__(
        cls,
        text: str,
        section_heading: str = "",
        page_number: Optional[int] = None,
        window_size: int = 1,
        is_explicit_absence: bool = False
    ):
        instance = super().__new__(cls, text)
        instance.text = text
        instance.section_heading = section_heading
        instance.page_number = page_number
        instance.window_size = window_size
        instance.is_explicit_absence = is_explicit_absence
        return instance


EXPLICIT_ABSENCE_PATTERN = re.compile(
    r'\b(?:no|none|neither)\s+[^.!?]{1,120}?\s+(?:is|are|was|were)?\s*(?:discussed|provided|specified|included|documented|defined|described|mentioned|given|present|available|found)\b|'
    r'\b(?:does not|do not|did not|is not|are not|cannot|will not|has not|have not)\s+(?:specify|explain|include|provide|contain|document|define|describe|mention|detail|list|cover|address|handle|state)\b|'
    r'\b(?:omitted|deliberately omitted|intentionally omitted|not included|absent|not covered|not documented|not provided|not addressed)\b',
    re.IGNORECASE
)


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
        page_number: Optional[int] = None,
        raise_if_scanned: bool = True
    ) -> PDFExtractionResult:
        """
        Extracts raw text from a PDF document across all pages, or from a specific
        single page (1-based index) if page_number is provided.
        Detects scanned/image-only documents when extractable text is negligible.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PDF file not found: '{filepath}'")

        try:
            reader = pypdf.PdfReader(filepath)
        except Exception as e:
            raise ValueError(f"Failed to read PDF file '{filepath}': {e}") from e

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise ValueError(f"PDF file '{filepath}' has 0 pages.")

        if page_number is not None:
            if page_number < 1 or page_number > total_pages:
                raise ValueError(
                    f"Page number {page_number} is out of range for '{os.path.basename(filepath)}'. "
                    f"Document contains {total_pages} page(s) (valid range: 1 to {total_pages})."
                )
            target_indices = [page_number - 1]
        else:
            target_indices = list(range(total_pages))

        pages_text: List[str] = []
        for idx in target_indices:
            try:
                page_text = reader.pages[idx].extract_text() or ""
            except Exception:
                page_text = ""
            pages_text.append(page_text)

        raw_full_text = "\n".join(pages_text)
        non_ws_chars = len(re.sub(r'\s+', '', raw_full_text))
        total_words = len(raw_full_text.split())
        considered_page_count = len(target_indices)
        words_per_page = total_words / max(considered_page_count, 1)

        is_scanned = (
            considered_page_count > 0 and (
                non_ws_chars < (self.min_total_chars if considered_page_count > 1 else 15) or
                words_per_page < self.min_words_per_page
            )
        )

        warning_message = None
        if is_scanned:
            page_scope = f"page {page_number}" if page_number is not None else f"across {total_pages} page(s)"
            warning_message = (
                f"Optical Character Recognition (OCR) is required: "
                f"Extracted only {non_ws_chars} character(s) {page_scope}."
            )
            if raise_if_scanned:
                raise ScannedPDFError(
                    filepath=filepath,
                    char_count=non_ws_chars,
                    page_count=considered_page_count,
                    message=warning_message
                )

        unwrapped_text = self.unwrap_text(raw_full_text)

        return PDFExtractionResult(
            filepath=filepath,
            text=unwrapped_text,
            page_count=total_pages,
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
        filepath: str,
        page_number: Optional[int] = None
    ) -> Tuple[List[DocumentSection], List[Requirement]]:
        """
        Extracts structural sections and individual requirement items from an SRS PDF
        across all pages or from a single page.
        Returns:
            Tuple of (sections, requirements)
        """
        result = self.extract_text(filepath, page_number=page_number, raise_if_scanned=True)
        extractor = RequirementExtractor()
        sections = extractor.extract_sections_from_text(result.text)
        requirements = extractor.extract_from_text(result.text)
        return sections, requirements

    def extract_template_from_text(
        self,
        text: str,
        default_title: str = "Reference Template"
    ) -> ReferenceTemplate:
        """
        Extracts structural sections and discrete required points from raw template text.

        Extraction Heuristic:
        1. Title: Extracted from common title headers or the first non-metadata, non-watermark line.
        2. Sections: Tracked via markdown (#), numbered (1. Scope), lettered (A. Overview),
           or tagged (Section 1) headings.
        3. Points: Extracted from:
           - Explicitly tagged requirement/checkpoint lines (e.g. REQ-xxx, SEC-xxx, R1, B1-B7, A1-A7, 1.1)
           - Bullet points (-, *, •)
           - Sub-numbered points (1.1, (a), (1))
           - Instructional or normative statements containing actions (shall, must, should,
             explain, describe, state, identify, list, include, provide, ensure, verify).
        4. Topic Labels: If a point is formatted as "Topic: Description" or "Topic - Description",
           both the topic label and description are captured for multi-perspective matching.

        Limitations:
        - Freeform narrative text without section headers, bullets, or requirement tags
          will rely on sentence-level heuristics and may miss nuanced requirements.
        - Tables and complex multi-column callouts rendered as disjointed text streams
          may have broken ordering.
        """
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return ReferenceTemplate(title=default_title, points=[], sections=[])

        # Metadata & noise patterns to ignore as requirement points
        noise_pattern = re.compile(
            r'^(?:Synthetic test document|Testing note|Designed test intent|Evaluation note|'
            r'Standard Version|Standard|Regulatory Baseline|Binding Modal|'
            r'Prohibited Vague Words|Title|Author|Date|Version)[:\s–—\-]',
            re.IGNORECASE
        )

        # Section header patterns
        header_pattern = re.compile(
            r'^(?:'
            r'#+\s+'                             # Markdown headings (# Title)
            r'|[A-Z]\.\s+[A-Za-z]'              # A. Section Title
            r'|\d+(?:\.\d+)*[\.\:\-\s]+[A-Z]'  # 1. Section Title, 1.1 Overview
            r'|Section\s+\d+'                    # Section 1
            r')',
            re.IGNORECASE
        )

        # Explicit requirement/checkpoint ID pattern: supports B1-B7, A1-A7, REQ-001, SEC-1, R1-R4, 1.1, etc.
        id_pattern = re.compile(
            r'^([A-Z]{1,4}[-_]?[0-9]+|(?:REQ|SRS|SYS|RULE|SEC|CHECK|POINT)[-_][A-Za-z0-9_-]+|\d+\.\d+)\s*[:\.\-–]\s*(.*)$',
            re.IGNORECASE
        )

        # Bullet and numbered point patterns
        bullet_pattern = re.compile(r'^[-*•]\s+(.*)$')
        numbered_point_pattern = re.compile(r'^(?:\d+\.\d+|\(\d+\)|\([a-z]\))\s+(.*)$', re.IGNORECASE)

        # Normative and instructional keywords for fallback point detection
        normative_and_instructional_keywords = [
            "shall", "must", "should", "will", "requires", "require", "ensure",
            "verify", "display", "explain", "describe", "state", "identify",
            "list", "include", "provide", "record"
        ]

        # Extract title
        title = default_title
        title_match = re.search(
            r'^(?:Reference Standard|Template|Standard Name|Specification|Software Requirements Specification)[:\s]+([^\n\|]+)',
            text,
            re.MULTILINE | re.IGNORECASE
        )
        if title_match:
            title = title_match.group(1).strip()
        else:
            # Pick first substantive line that is not a banner, header, or ID
            for l in lines:
                if (
                    not noise_pattern.search(l)
                    and "created for prototype evaluation" not in l
                    and not header_pattern.match(l)
                    and not id_pattern.match(l)
                ):
                    title = l.strip()
                    break

        points: List[TemplatePoint] = []
        sections: List[str] = []
        current_section = "General"
        point_counter = 1

        for line_num, line in enumerate(lines, start=1):
            if (
                noise_pattern.search(line)
                or "created for prototype evaluation" in line
                or line.startswith("Testing note:")
            ):
                continue

            id_match = id_pattern.match(line)

            # Check if this line is purely a section header (and not a tagged point)
            if not id_match and header_pattern.match(line):
                clean_sec = re.sub(
                    r'^(?:#+\s*|[A-Z]\.\s*|\d+(?:\.\d+)*[\.\:\-\s]*|Section\s+\d+[\.\:\-\s]*)',
                    '',
                    line
                ).strip()
                current_section = clean_sec if clean_sec else line
                if current_section not in sections:
                    sections.append(current_section)
                continue

            if id_match:
                pid = id_match.group(1).upper()
                body = id_match.group(2).strip()
                text_val = body if body else line

                # Separate topic label vs detailed description if present
                split_match = re.match(r'^([A-Za-z0-9\s/&,]{2,40}?)\s*(?::|\s+[-–—]\s+)(.+)$', body)
                if split_match:
                    topic = split_match.group(1).strip()
                    desc = split_match.group(2).strip()
                else:
                    topic = ""
                    desc = body

                points.append(
                    TemplatePoint(
                        point_id=pid,
                        section_name=current_section,
                        text=text_val,
                        topic_label=topic,
                        description=desc,
                        line_number=line_num,
                        raw_line=line
                    )
                )
                point_counter += 1
            elif bullet_pattern.match(line):
                body = bullet_pattern.match(line).group(1).strip()
                split_match = re.match(r'^([A-Za-z0-9\s/&,]{2,40}?)\s*(?::|\s+[-–—]\s+)(.+)$', body)
                topic = split_match.group(1).strip() if split_match else ""
                desc = split_match.group(2).strip() if split_match else body
                points.append(
                    TemplatePoint(
                        point_id=f"PT-{point_counter:03d}",
                        section_name=current_section,
                        text=body,
                        topic_label=topic,
                        description=desc,
                        line_number=line_num,
                        raw_line=line
                    )
                )
                point_counter += 1
            elif numbered_point_pattern.match(line):
                body = numbered_point_pattern.match(line).group(1).strip()
                split_match = re.match(r'^([A-Za-z0-9\s/&,]{2,40}?)\s*(?::|\s+[-–—]\s+)(.+)$', body)
                topic = split_match.group(1).strip() if split_match else ""
                desc = split_match.group(2).strip() if split_match else body
                points.append(
                    TemplatePoint(
                        point_id=f"PT-{point_counter:03d}",
                        section_name=current_section,
                        text=body,
                        topic_label=topic,
                        description=desc,
                        line_number=line_num,
                        raw_line=line
                    )
                )
                point_counter += 1
            elif len(line) >= 25 and any(re.search(rf'\b{kw}\b', line, re.IGNORECASE) for kw in normative_and_instructional_keywords):
                # Don't duplicate the title line as a point
                if line != title:
                    split_match = re.match(r'^([A-Za-z0-9\s/&,]{2,40}?)\s*(?::|\s+[-–—]\s+)(.+)$', line)
                    topic = split_match.group(1).strip() if split_match else ""
                    desc = split_match.group(2).strip() if split_match else line
                    points.append(
                        TemplatePoint(
                            point_id=f"PT-{point_counter:03d}",
                            section_name=current_section,
                            text=line,
                            topic_label=topic,
                            description=desc,
                            line_number=line_num,
                            raw_line=line
                        )
                    )
                    point_counter += 1

        return ReferenceTemplate(
            title=title,
            points=points,
            sections=sections if sections else ["General"]
        )

    def extract_template(self, filepath: str) -> ReferenceTemplate:
        """
        Extracts required sections and points from a reference template PDF.
        """
        result = self.extract_text(filepath, raise_if_scanned=True)
        default_title = os.path.splitext(os.path.basename(filepath))[0].replace("_", " ").title()
        return self.extract_template_from_text(result.text, default_title=default_title)

    def extract_target_chunks(
        self,
        filepath: str,
        page_number: Optional[int] = None
    ) -> List[CandidateChunk]:
        """
        Extracts candidate comparison text chunks (individual sentences, 2-sentence windows,
        and 3-sentence windows constrained strictly to paragraph boundaries) from a target PDF.
        Preserves original evidence text, page numbers, and enclosing section headings.
        Filters out standalone section headings, running headers/footers, and test notes.
        """
        result = self.extract_text(filepath, page_number=page_number, raise_if_scanned=True)

        def is_target_noise(l: str) -> bool:
            if not l:
                return True
            # 1. Banners, footers, watermarks
            if re.search(r'Synthetic test document|created for prototype evaluation|Target document\s+[A-Z]|Reference document\s+[A-Z]', l, re.IGNORECASE):
                return True
            if re.match(r'^Page\s+\d+(?:\s+of\s+\d+)?$', l, re.IGNORECASE):
                return True
            # 2. Test intent / meta notes
            if re.match(r'^(?:Designed test intent|Testing note|Evaluation note|Test intent|Note:)', l, re.IGNORECASE):
                return True
            return False

        def parse_heading(l: str) -> Optional[str]:
            heading_match = re.match(
                r'^(?:#+\s*|\d+(?:\.\d+)*[\.\:\-\s]+|[A-Z]\.[\.\:\-\s]+|Section\s+\d+[\.\:\-\s]+)([A-Za-z0-9\s/&,-]+)$',
                l,
                re.IGNORECASE
            )
            if heading_match:
                heading_title = heading_match.group(1).strip()
                if not l.endswith(('.', '?', '!', ':')) or len(heading_title.split()) <= 6:
                    verbs = {
                        'is', 'are', 'was', 'were', 'has', 'have', 'shall', 'should',
                        'will', 'must', 'can', 'does', 'do', 'helps', 'monitors',
                        'monitored', 'collects', 'collected', 'displays', 'displayed', 'used'
                    }
                    words = set(re.findall(r'\b[a-z]+\b', heading_title.lower()))
                    if not (words & verbs):
                        return heading_title
            return None

        chunks: List[CandidateChunk] = []
        seen = set()

        # Determine which pages to process
        if page_number is not None:
            pages_to_process = [(page_number, result.pages_text[0] if result.pages_text else result.text)]
        else:
            pages_to_process = list(enumerate(result.pages_text, start=1)) if result.pages_text else [(1, result.text)]

        for page_idx, page_str in pages_to_process:
            page_text = self.unwrap_text(page_str)
            curr_heading = ""
            curr_paragraph_sentences: List[str] = []

            def flush_paragraph():
                nonlocal curr_paragraph_sentences
                if not curr_paragraph_sentences:
                    return

                # Segment into consecutive non-absence runs and individual absence statements
                runs: List[Tuple[List[str], bool]] = []
                curr_run: List[str] = []
                for s in curr_paragraph_sentences:
                    if EXPLICIT_ABSENCE_PATTERN.search(s):
                        if curr_run:
                            runs.append((curr_run, False))
                            curr_run = []
                        runs.append(([s], True))
                    else:
                        curr_run.append(s)
                if curr_run:
                    runs.append((curr_run, False))

                for sentence_group, is_absence in runs:
                    k = len(sentence_group)
                    # 1. Single sentences
                    for s in sentence_group:
                        if s not in seen:
                            seen.add(s)
                            chunks.append(
                                CandidateChunk(
                                    text=s,
                                    section_heading=curr_heading,
                                    page_number=page_idx,
                                    window_size=1,
                                    is_explicit_absence=is_absence
                                )
                            )
                    # 2. Two-sentence sliding windows (only for substantive sentences)
                    if not is_absence and k >= 2:
                        for i in range(k - 1):
                            w2 = f"{sentence_group[i]} {sentence_group[i+1]}"
                            if w2 not in seen:
                                seen.add(w2)
                                chunks.append(
                                    CandidateChunk(
                                        text=w2,
                                        section_heading=curr_heading,
                                        page_number=page_idx,
                                        window_size=2,
                                        is_explicit_absence=False
                                    )
                                )
                    # 3. Three-sentence sliding windows (only for substantive sentences)
                    if not is_absence and k >= 3:
                        for i in range(k - 2):
                            w3 = f"{sentence_group[i]} {sentence_group[i+1]} {sentence_group[i+2]}"
                            if w3 not in seen:
                                seen.add(w3)
                                chunks.append(
                                    CandidateChunk(
                                        text=w3,
                                        section_heading=curr_heading,
                                        page_number=page_idx,
                                        window_size=3,
                                        is_explicit_absence=False
                                    )
                                )

                curr_paragraph_sentences = []

            for line in page_text.splitlines():
                line = line.strip()
                if not line:
                    flush_paragraph()
                    continue

                heading_found = parse_heading(line)
                if heading_found:
                    flush_paragraph()
                    curr_heading = heading_found
                    continue

                if is_target_noise(line):
                    continue

                # Multi-sentence splitting on terminal punctuation
                if "." in line or "?" in line or "!" in line:
                    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', line)
                    for s in sentences:
                        s_clean = s.strip()
                        if len(s_clean) >= 15 and not is_target_noise(s_clean):
                            curr_paragraph_sentences.append(s_clean)
                else:
                    if len(line) >= 15:
                        curr_paragraph_sentences.append(line)

            flush_paragraph()

        return chunks

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
def extract_text_from_pdf(filepath: str, page_number: Optional[int] = None) -> str:
    """Convenience function to extract text from a PDF file (all pages or single page)."""
    return PDFExtractor().extract_text(filepath, page_number=page_number).text


def extract_srs_from_pdf(
    filepath: str,
    page_number: Optional[int] = None
) -> Tuple[List[DocumentSection], List[Requirement]]:
    """Convenience function to extract SRS sections and requirements from a PDF file."""
    return PDFExtractor().extract_srs_document(filepath, page_number=page_number)


def extract_standard_from_pdf(filepath: str) -> ReferenceStandard:
    """Convenience function to parse a ReferenceStandard from a PDF file."""
    return PDFExtractor().extract_reference_standard(filepath)


def extract_template_from_pdf(filepath: str) -> ReferenceTemplate:
    """Convenience function to parse a ReferenceTemplate from a PDF file."""
    return PDFExtractor().extract_template(filepath)


def extract_chunks_from_pdf(filepath: str, page_number: Optional[int] = None) -> List[CandidateChunk]:
    """Convenience function to extract paragraph-constrained candidate chunks from a PDF file."""
    return PDFExtractor().extract_target_chunks(filepath, page_number=page_number)
