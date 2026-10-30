"""
V2 Target PDF Structural Parser.
Extracts headings, paragraphs, list items, and table content with strict cross-platform
compatibility (macOS and Windows), preserving page metadata and section hierarchies.
"""
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.models.document import BlockType, DocumentBlock, TargetDocument


# Explicit absence detection pattern (covers standard disclaimers, negative modals, omissions)
EXPLICIT_ABSENCE_PATTERN = re.compile(
    r"\b("
    r"no\s+(?:specific\s+)?(?:hazards?|safety|precautions?|numeric|capacity|response-time|acceptance|criteria|targets?)\b"
    r"|(?:is|are|was|were)\s+(?:not\s+(?:discussed|addressed|explained|covered|specified|included|provided|implemented)|absent|omitted)"
    r"|(?:does|did|do)\s+not\s+(?:specify|explain|list|include|cover|describe|contain|provide|address)"
    r"|(?:there\s+(?:is|are)\s+no\b)"
    r"|(?:outside\s+(?:the\s+)?scope)"
    r"|(?:not\s+intended\s+to\b)"
    r")\b",
    re.IGNORECASE
)

# Heading patterns: numbered (e.g. "1. Overview", "2.1 Setup") or letter-numbered ("A1. Aim", "B4. Interfaces")
HEADING_REGEXES = [
    re.compile(r"^\s*([0-9]+(?:\.[0-9]+)*)\.?\s+([A-Z][A-Za-z0-9\s,\-_/&()]{2,60})\s*$"),
    re.compile(r"^\s*([A-Za-z][0-9]+)\.?\s*[:\-–—]?\s+([A-Z][A-Za-z0-9\s,\-_/&()]{2,60})\s*$"),
    re.compile(r"^\s*(Section|Chapter|Appendix)\s+([0-9A-Za-z]+)\.?\s*[:\-–—]?\s+(.+)$", re.IGNORECASE),
]

# Bullet list patterns
LIST_ITEM_REGEX = re.compile(r"^\s*(?:[•\-\*\u2022\u2023\u25E6\u2043\u2219]|(?:\([0-9a-z]\))|(?:[0-9a-z]\.))\s+(.+)$", re.IGNORECASE)

# Table row detection heuristics (pipes, semicolon-separated key-value pairs, or aligned tabs)
TABLE_ROW_REGEX = re.compile(
    r"^\s*\|?.+\|.+\|?\s*$"                         # Markdown/pipe table
    r"|^\s*[A-Za-z0-9\s\-_]+:\s*[^;]+;\s*[A-Za-z0-9\s\-_]+:\s*[^;]+;?\s*$"  # Semicolon-delimited records (e.g. "start 70 C; 2 min 61 C;")
)

# Unwanted page headers/footers/synthetic banners
NOISE_PATTERNS = [
    re.compile(r"^Synthetic\s+test\s+document\s*—", re.IGNORECASE),
    re.compile(r"^Page\s+[0-9]+(\s+of\s+[0-9]+)?$", re.IGNORECASE),
    re.compile(r"Designed\s+test\s+intent\s*:", re.IGNORECASE),
    re.compile(r"^Target\s+document\s+[A-Za-z0-9]\s*\|", re.IGNORECASE),
]


class TargetPDFParser:
    """
    Parses digital PDF target documents into a structured collection of DocumentBlocks.
    Preserves exact section hierarchy, page numbers, and structural types without
    mixing unrelated paragraphs or fragmenting sentences.
    """

    def __init__(self, min_total_chars: int = 50):
        self.min_total_chars = min_total_chars

    def parse_document(
        self,
        pdf_path: str,
        target_page: Optional[int] = None
    ) -> TargetDocument:
        """
        Parses an entire PDF file or a specific 1-based page.

        Args:
            pdf_path: Absolute or relative path to PDF document.
            target_page: Optional 1-based page number to restrict extraction.

        Returns:
            TargetDocument populated with structured DocumentBlocks.

        Raises:
            FileNotFoundError: If PDF does not exist.
            ValueError: If target_page is out of range or file is not a valid PDF.
        """
        norm_path = os.path.normpath(str(pdf_path))
        if not os.path.isfile(norm_path):
            raise FileNotFoundError(f"PDF document not found at: {norm_path}")

        try:
            reader = PdfReader(norm_path)
        except (PdfReadError, Exception) as exc:
            raise ValueError(f"Failed to read PDF file '{norm_path}': {exc}") from exc

        num_pages = len(reader.pages)
        if num_pages == 0:
            raise ValueError(f"PDF '{norm_path}' contains 0 pages.")

        if target_page is not None:
            if target_page < 1 or target_page > num_pages:
                raise ValueError(
                    f"Invalid target_page={target_page}. Document has {num_pages} pages (allowed: 1..{num_pages})."
                )
            pages_to_extract = [(target_page, reader.pages[target_page - 1])]
        else:
            pages_to_extract = [(i + 1, page) for i, page in enumerate(reader.pages)]

        filename = os.path.basename(norm_path)
        blocks: List[DocumentBlock] = []
        block_counter = 0

        # Section tracking state
        current_section: Optional[str] = None
        current_hierarchy: List[str] = []
        doc_title: Optional[str] = None

        for page_num, page_obj in pages_to_extract:
            try:
                page_raw_text = page_obj.extract_text() or ""
            except Exception as e:
                # Log extraction failure on individual page and continue
                page_raw_text = ""

            unwrapped_page = self._unwrap_page_text(page_raw_text)
            page_blocks, current_section, current_hierarchy, extracted_title = self._segment_page_blocks(
                unwrapped_page,
                page_num=page_num,
                initial_section=current_section,
                initial_hierarchy=current_hierarchy,
                block_counter_start=block_counter
            )

            if not doc_title and extracted_title:
                doc_title = extracted_title

            block_counter += len(page_blocks)
            blocks.extend(page_blocks)

        return TargetDocument(
            filename=filename,
            total_pages=num_pages,
            blocks=blocks,
            document_title=doc_title or filename,
            target_page=target_page
        )

    def _unwrap_page_text(self, text: str) -> str:
        """
        Unwraps hard line-breaks inside paragraphs while preserving paragraph breaks.
        Lines that are wrapped within a paragraph are joined with a space.
        Consecutive empty lines or structural breaks (e.g. headings) are preserved.
        """
        if not text:
            return ""

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Strip multi-line synthetic test intent blocks before line unwrapping
        text = re.sub(r"Designed\s+test\s+intent\s*:[^\n]*(?:\n[^\n]*)*", "", text, flags=re.IGNORECASE)

        # Strip remaining synthetic/page noise lines early
        lines: List[str] = []
        for line in text.split("\n"):
            s_line = line.strip()
            if not s_line:
                lines.append("")
                continue
            if any(p.search(s_line) for p in NOISE_PATTERNS):
                continue
            lines.append(s_line)

        # 1. First, reconstruct broken paragraphs by joining wrapped lines
        reconstructed_lines: List[str] = []
        for line in lines:
            if not line:
                reconstructed_lines.append("")
                continue

            # Section headings stay on their own lines
            is_h, _, _ = self._detect_heading(line)
            if is_h:
                reconstructed_lines.append(line)
                continue

            # If previous line was non-empty and didn't end with sentence terminal punctuation (. ? ! : ;),
            # it was wrapped by the PDF renderer mid-sentence. Join it with space.
            if reconstructed_lines and reconstructed_lines[-1] and not self._is_terminal_line(reconstructed_lines[-1]):
                if reconstructed_lines[-1].endswith("-"):
                    reconstructed_lines[-1] = reconstructed_lines[-1][:-1] + line
                else:
                    reconstructed_lines[-1] = reconstructed_lines[-1] + " " + line
            else:
                reconstructed_lines.append(line)

        # 2. Next, split lines where an explicit absence statement begins mid-paragraph
        final_lines: List[str] = []
        for r_line in reconstructed_lines:
            if not r_line or self._detect_heading(r_line)[0]:
                final_lines.append(r_line)
                continue

            # Check if an explicit absence sentence begins inside this paragraph
            # e.g. "unavailable. The document does not explain recovery steps..."
            abs_match = EXPLICIT_ABSENCE_PATTERN.search(r_line)
            if abs_match and abs_match.start() > 10:
                # Find the sentence boundary preceding the absence match
                prev_dot = r_line.rfind(". ", 0, abs_match.start())
                if prev_dot != -1:
                    before = r_line[:prev_dot + 1].strip()
                    after = r_line[prev_dot + 1:].strip()
                    if before:
                        final_lines.append(before)
                    if after:
                        final_lines.append(after)
                    continue

            final_lines.append(r_line)

        return "\n".join(final_lines)

    def _is_terminal_line(self, line: str) -> bool:
        """Checks if a line terminates with a full stop, colon, or heading structure."""
        s = line.strip()
        if not s:
            return True
        if self._detect_heading(s)[0]:
            return True
        return s[-1] in {".", "!", "?", ":", ";"}

    def _segment_page_blocks(
        self,
        page_text: str,
        page_num: int,
        initial_section: Optional[str],
        initial_hierarchy: List[str],
        block_counter_start: int
    ) -> Tuple[List[DocumentBlock], Optional[str], List[str], Optional[str]]:
        """
        Segments unwrapped text into discrete typed blocks (HEADING, PARAGRAPH, LIST_ITEM, TABLE_ROW).
        """
        lines = page_text.split("\n")
        blocks: List[DocumentBlock] = []
        counter = block_counter_start

        current_sec = initial_section
        current_hier = list(initial_hierarchy)
        doc_title: Optional[str] = None

        # Accumulator for normal prose paragraphs
        prose_buffer: List[str] = []
        prose_start_line = 0

        def flush_prose():
            nonlocal counter, prose_buffer, prose_start_line
            if not prose_buffer:
                return
            combined_text = " ".join(prose_buffer).strip()
            # Collapse multiple spaces
            combined_text = re.sub(r"\s+", " ", combined_text)
            prose_buffer = []

            if len(combined_text) < 4:
                return

            # Check if this paragraph contains an explicit absence/disclaimer statement
            is_absence = bool(EXPLICIT_ABSENCE_PATTERN.search(combined_text))
            b_type = BlockType.DISCLAIMER if is_absence else BlockType.PARAGRAPH

            counter += 1
            blocks.append(
                DocumentBlock(
                    block_id=f"p{page_num}_b{counter}",
                    text=combined_text,
                    block_type=b_type,
                    page_number=page_num,
                    section_heading=current_sec,
                    section_hierarchy=list(current_hier),
                    line_start=prose_start_line,
                    line_end=prose_start_line + len(combined_text.split()),
                    is_explicit_omission=is_absence,
                )
            )

        for line_idx, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line:
                # Blank line indicates paragraph boundary
                flush_prose()
                continue

            # 1. Check for Section Heading
            is_heading, heading_title, heading_num = self._detect_heading(line)
            if is_heading:
                flush_prose()
                current_sec = line
                current_hier = [line]

                counter += 1
                blocks.append(
                    DocumentBlock(
                        block_id=f"p{page_num}_b{counter}",
                        text=line,
                        block_type=BlockType.HEADING,
                        page_number=page_num,
                        section_heading=current_sec,
                        section_hierarchy=list(current_hier),
                        line_start=line_idx,
                        line_end=line_idx,
                        is_explicit_omission=False,
                        metadata={"heading_number": heading_num, "heading_title": heading_title}
                    )
                )
                continue

            # 2. Check for Table Row / Semi-structured measurement row
            if self._is_table_or_data_row(line):
                flush_prose()
                counter += 1
                blocks.append(
                    DocumentBlock(
                        block_id=f"p{page_num}_b{counter}",
                        text=line,
                        block_type=BlockType.TABLE_ROW,
                        page_number=page_num,
                        section_heading=current_sec,
                        section_hierarchy=list(current_hier),
                        line_start=line_idx,
                        line_end=line_idx,
                        is_explicit_omission=False,
                        metadata={"is_tabular_data": True}
                    )
                )
                continue

            # 3. Check for Bullet / Ordered List Item
            list_match = LIST_ITEM_REGEX.match(line)
            if list_match:
                flush_prose()
                counter += 1
                item_content = list_match.group(1).strip()
                is_absence = bool(EXPLICIT_ABSENCE_PATTERN.search(item_content))
                blocks.append(
                    DocumentBlock(
                        block_id=f"p{page_num}_b{counter}",
                        text=line,
                        block_type=BlockType.LIST_ITEM,
                        page_number=page_num,
                        section_heading=current_sec,
                        section_hierarchy=list(current_hier),
                        line_start=line_idx,
                        line_end=line_idx,
                        is_explicit_omission=is_absence,
                    )
                )
                continue

            # 4. Check for Document Title (if not yet assigned and appears before any section heading)
            if not current_sec and not doc_title and len(line) < 80 and not line.endswith("."):
                # Title candidate
                doc_title = line

            # 5. Check if the line itself is an explicit absence/disclaimer statement
            if EXPLICIT_ABSENCE_PATTERN.search(line):
                flush_prose()
                counter += 1
                blocks.append(
                    DocumentBlock(
                        block_id=f"p{page_num}_b{counter}",
                        text=line,
                        block_type=BlockType.DISCLAIMER,
                        page_number=page_num,
                        section_heading=current_sec,
                        section_hierarchy=list(current_hier),
                        line_start=line_idx,
                        line_end=line_idx,
                        is_explicit_omission=True,
                    )
                )
                continue

            # 6. Regular prose: accumulate into current paragraph buffer
            if not prose_buffer:
                prose_start_line = line_idx
            prose_buffer.append(line)

        # Flush any remaining paragraph at end of page
        flush_prose()

        return blocks, current_sec, current_hier, doc_title

    def _detect_heading(self, line: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Determines if a line is a structural heading."""
        if len(line) > 80:
            return False, None, None

        for regex in HEADING_REGEXES:
            m = regex.match(line)
            if m:
                groups = m.groups()
                if len(groups) >= 2:
                    return True, groups[1].strip(), groups[0].strip()
                return True, groups[0].strip(), None

        return False, None, None

    def _is_table_or_data_row(self, line: str) -> bool:
        """Detects whether a line contains table structure, tab-separated values, or semicolon records."""
        # Check pipe markdown table row
        if line.startswith("|") or (line.count("|") >= 2):
            return True

        # Check semicolon delimited measurement row (e.g. "start 70 degrees C; 2 minutes 61 degrees C...")
        if line.count(";") >= 2 and any(char.isdigit() for char in line):
            return True

        # Check tab-separated columns
        if line.count("\t") >= 2:
            return True

        return False
