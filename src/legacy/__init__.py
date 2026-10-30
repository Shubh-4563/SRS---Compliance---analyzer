"""
Package initialization for src.legacy.
Preserves original V1 and V1.1 compliance analyzers and extractors.
"""
from src.analyzer import (
    ComplianceAnalyzer,
    TemplateContentChecker,
    ReportGenerator,
    PointMatchResult,
    TemplateComplianceReport,
    ReferenceTemplate,
    TemplatePoint,
)
from src.pdf_extractor import (
    PDFExtractor,
    CandidateChunk,
    EXPLICIT_ABSENCE_PATTERN,
    extract_template_from_pdf,
    extract_chunks_from_pdf,
)

__all__ = [
    "ComplianceAnalyzer",
    "TemplateContentChecker",
    "ReportGenerator",
    "PointMatchResult",
    "TemplateComplianceReport",
    "ReferenceTemplate",
    "TemplatePoint",
    "PDFExtractor",
    "CandidateChunk",
    "EXPLICIT_ABSENCE_PATTERN",
    "extract_template_from_pdf",
    "extract_chunks_from_pdf",
]

