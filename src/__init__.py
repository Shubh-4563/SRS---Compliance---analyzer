"""
SRS Compliance Analyzer - Version 0.4
Core pipeline package with PDF extraction, semantic section matching, semantic rule matching, and compliance rule verification.
"""

from .analyzer import (
    Requirement,
    ReferenceSection,
    ReferenceRule,
    DocumentSection,
    SectionMatchResult,
    RuleMatchResult,
    RuleCheckResult,
    RequirementAudit,
    ComplianceReport,
    ReferenceStandard,
    SemanticSectionMatcher,
    SemanticMatcher,
    SemanticRuleMatcher,
    RequirementExtractor,
    ComplianceAnalyzer,
    ReportGenerator,
)

from .pdf_extractor import (
    ScannedPDFError,
    PDFExtractionResult,
    PDFExtractor,
    extract_text_from_pdf,
    extract_srs_from_pdf,
    extract_standard_from_pdf,
)

__all__ = [
    "Requirement",
    "ReferenceSection",
    "ReferenceRule",
    "DocumentSection",
    "SectionMatchResult",
    "RuleMatchResult",
    "RuleCheckResult",
    "RequirementAudit",
    "ComplianceReport",
    "ReferenceStandard",
    "SemanticSectionMatcher",
    "SemanticMatcher",
    "SemanticRuleMatcher",
    "RequirementExtractor",
    "ComplianceAnalyzer",
    "ReportGenerator",
    "ScannedPDFError",
    "PDFExtractionResult",
    "PDFExtractor",
    "extract_text_from_pdf",
    "extract_srs_from_pdf",
    "extract_standard_from_pdf",
]

