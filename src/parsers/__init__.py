"""
V2 Parsers Package Initialization.
"""
from src.parsers.pdf_parser import TargetPDFParser
from src.parsers.requirement_decomposer import RequirementDecomposer

__all__ = [
    "TargetPDFParser",
    "RequirementDecomposer",
]
