"""
V2 Pipeline Orchestration Layer (Phase 2.5.3).
Provides CompliancePipeline connecting template ingestion, requirement decomposition,
target parsing, hybrid retrieval, NLI evidence verification, and compliance aggregation
into a unified, deterministic workflow.
"""
import os
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

from src.pdf_extractor import PDFExtractor, ScannedPDFError, ReferenceTemplate
from src.parsers.pdf_parser import TargetPDFParser
from src.parsers.requirement_decomposer import RequirementDecomposer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.verification.nli_verifier import NLIVerifier
from src.verification.evidence_verifier import EvidenceVerifier
from src.verification.compliance_aggregator import ComplianceAggregator
from src.models.compliance import DocumentComplianceReport
from src.models.document import TargetDocument
from src.models.requirement import DecomposedReferenceTemplate
from src.models.verification import SubRequirementVerificationResult


class CompliancePipeline:
    """
    End-to-end orchestration pipeline for reference-guided document compliance verification.
    Coordinates:
    1. Input path and PDF integrity validation.
    2. Reference template ingestion and atomic requirement decomposition.
    3. Target document parsing (with optional page scope filtering).
    4. Dual-engine hybrid retrieval (BM25 + Dense embeddings with RRF).
    5. NLI-based evidence verification per atomic sub-requirement.
    6. Hierarchical compliance aggregation and audit report generation.
    """

    def __init__(
        self,
        extractor: Optional[PDFExtractor] = None,
        parser: Optional[TargetPDFParser] = None,
        decomposer: Optional[RequirementDecomposer] = None,
        retriever: Optional[HybridRetriever] = None,
        verifier: Optional[EvidenceVerifier] = None,
        aggregator: Optional[ComplianceAggregator] = None,
        local_files_only: bool = True,
        rrf_k: int = 60,
        default_top_k: int = 5,
        min_substantive_chars: int = 50,
    ):
        """
        Initializes the pipeline with optional dependency injection for testability.
        Model-heavy components (retriever, verifier) are initialized lazily if not injected.
        """
        self.extractor = extractor or PDFExtractor()
        self.parser = parser or TargetPDFParser()
        self.decomposer = decomposer or RequirementDecomposer()
        self.aggregator = aggregator or ComplianceAggregator()
        self.local_files_only = local_files_only
        self.rrf_k = rrf_k
        self.default_top_k = default_top_k
        self.min_substantive_chars = min_substantive_chars

        self._retriever = retriever
        self._verifier = verifier

    @property
    def retriever(self) -> HybridRetriever:
        """Returns injected or lazily-initialized HybridRetriever."""
        if self._retriever is None:
            self._retriever = HybridRetriever(rrf_k=self.rrf_k)
        return self._retriever

    @property
    def verifier(self) -> EvidenceVerifier:
        """Returns injected or lazily-initialized EvidenceVerifier."""
        if self._verifier is None:
            nli = NLIVerifier(local_files_only=self.local_files_only)
            self._verifier = EvidenceVerifier(nli_verifier=nli)
        return self._verifier

    def run(
        self,
        reference_path: Union[str, Path],
        target_path: Union[str, Path],
        target_page: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> DocumentComplianceReport:
        """
        Executes end-to-end reference-guided compliance verification.

        Args:
            reference_path: File path to the reference standard or template PDF.
            target_path: File path to the target document PDF to audit.
            target_page: Optional 1-based page number to restrict target extraction.
            top_k: Number of candidate evidence blocks to retrieve per sub-requirement (default: 5).

        Returns:
            DocumentComplianceReport populated with hierarchical compliance findings.

        Raises:
            FileNotFoundError: If reference or target files do not exist.
            ValueError: If inputs are invalid, empty, or corrupted.
            ScannedPDFError: If reference or target document is an image-only / scanned PDF.
        """
        effective_top_k = top_k if top_k is not None else self.default_top_k

        # ----------------------------------------------------------------------
        # 1. Input Path and Pre-Flight Validation
        # ----------------------------------------------------------------------
        ref_path = Path(reference_path).resolve()
        tgt_path = Path(target_path).resolve()

        if not ref_path.exists():
            raise FileNotFoundError(f"Reference template file not found at: {ref_path}")
        if not ref_path.is_file():
            raise ValueError(f"Reference template path is not a file: {ref_path}")
        if ref_path.stat().st_size == 0:
            raise ValueError(f"Reference template file '{ref_path.name}' is empty (0 bytes).")

        if not tgt_path.exists():
            raise FileNotFoundError(f"Target document file not found at: {tgt_path}")
        if not tgt_path.is_file():
            raise ValueError(f"Target document path is not a file: {tgt_path}")
        if tgt_path.stat().st_size == 0:
            raise ValueError(f"Target document file '{tgt_path.name}' is empty (0 bytes).")

        if target_page is not None:
            if not isinstance(target_page, int) or isinstance(target_page, bool):
                raise ValueError(f"target_page must be an integer, got {type(target_page).__name__}")
            if target_page < 1:
                raise ValueError(f"Invalid target_page={target_page}. Page numbers must be >= 1.")

        # ----------------------------------------------------------------------
        # 2. Reference Template Extraction & Requirement Decomposition
        # ----------------------------------------------------------------------
        try:
            raw_template = self.extractor.extract_template(str(ref_path))
        except ScannedPDFError:
            raise
        except Exception as exc:
            raise ValueError(f"Failed to extract reference template '{ref_path.name}': {exc}") from exc

        if not raw_template or not getattr(raw_template, "points", None):
            raise ValueError(f"Reference template '{ref_path.name}' contains no requirement sections or points.")

        dec_template = self.decomposer.decompose_template(raw_template)
        if not dec_template.points or dec_template.total_sub_requirements == 0:
            raise ValueError(f"Reference template '{ref_path.name}' yielded zero verifiable atomic sub-requirements.")

        # ----------------------------------------------------------------------
        # 3. Target Document Parsing & Integrity Check
        # ----------------------------------------------------------------------
        try:
            target_doc = self.parser.parse_document(str(tgt_path), target_page=target_page)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Failed to parse target document '{tgt_path.name}': {exc}") from exc

        # Validate target_page bounds against parsed document page count
        if target_page is not None and target_page > target_doc.total_pages:
            raise ValueError(
                f"Invalid target_page={target_page}. Target document '{tgt_path.name}' "
                f"has only {target_doc.total_pages} page(s) (allowed: 1..{target_doc.total_pages})."
            )

        # Check for empty content
        if not target_doc.blocks:
            raise ValueError(f"Target document '{tgt_path.name}' contains no extractable content blocks.")

        # Check for scanned / image-only content in target document
        total_substantive_chars = sum(len(b.text.strip()) for b in target_doc.substantive_blocks)
        if total_substantive_chars < self.min_substantive_chars:
            raise ScannedPDFError(
                filepath=str(tgt_path),
                char_count=total_substantive_chars,
                page_count=target_doc.total_pages,
                message=(
                    f"Scanned or image-only PDF detected ('{tgt_path.name}'). "
                    f"Text extraction returned only {total_substantive_chars} character(s) across {target_doc.total_pages} page(s). "
                    f"Optical Character Recognition (OCR) is required to process this document."
                ),
            )

        # ----------------------------------------------------------------------
        # 4. Target Document Indexing for Hybrid Retrieval
        # ----------------------------------------------------------------------
        self.retriever.index_document(target_doc, include_headings=False)

        # ----------------------------------------------------------------------
        # 5. Candidate Evidence Retrieval & NLI Verification
        # ----------------------------------------------------------------------
        verification_results: List[SubRequirementVerificationResult] = []
        for point in dec_template.points:
            for sr in point.sub_requirements:
                candidates = self.retriever.retrieve_candidates_for_subrequirement(
                    sr, top_k=effective_top_k
                )
                res = self.verifier.verify_subrequirement(sr, candidates)
                verification_results.append(res)

        # ----------------------------------------------------------------------
        # 6. Compliance Aggregation and Scoring Rollup
        # ----------------------------------------------------------------------
        report = self.aggregator.aggregate(
            template=dec_template,
            verification_results=verification_results,
            target_document_path=str(tgt_path),
            target_document_name=tgt_path.name,
            total_pages=target_doc.total_pages,
            target_page=target_page,
        )

        return report

