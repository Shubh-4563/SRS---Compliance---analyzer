# SRS Compliance Analyzer — Project Context & Architectural Source of Truth

Last Updated: 2026-11-10
Status: Living Documentation (Phase 2.6.2.2 Complete, Phase 2.6.2.3 Ready)
Repository Root: `/Users/shubhamsharma/Downloads/AI - SRS - Compliance `


## 1. Executive Summary

The SRS Compliance Analyzer is an automated system engineering and natural language processing tool designed to evaluate System Requirements Specification (SRS) documents against reference standards, regulatory frameworks, and procedural templates.

In mission-critical software and safety-critical industries (aerospace, automotive, medical devices, enterprise infrastructure), compliance verification is traditionally an expensive, labor-intensive, and error-prone manual auditing process. The SRS Compliance Analyzer solves this problem by providing:

* Deterministic requirement decomposition: Breaks down unstructured or semi-structured template documents into atomic, verifiable sub-requirements with typed claims.
* Hybrid multi-stage candidate retrieval: Combines BM25 keyword search with dense semantic embeddings (`all-MiniLM-L6-v2`) via Reciprocal Rank Fusion (RRF) to identify high-probability evidence passages in target documents.
* Cross-encoder Natural Language Inference (NLI) verification: Employs transformer-based NLI (`cross-encoder/nli-deberta-v3-small`) to evaluate entailment, neutral, and contradiction relationships between requirement claims and retrieved evidence blocks.
* Multi-level deterministic compliance aggregation: Evaluates granular sub-requirements, aggregates parent requirement sections, and determines document-level compliance verdicts with strict mathematical coverage formulas.
* Transparent audit provenance: Guarantees end-to-end traceability from raw PDF bounding boxes and page numbers to executive audit reports.


## 2. System Architecture: V1 vs V2

The project has evolved through two major architectural generations:

### V1 Architecture (Legacy Baseline)
* Components: `src/parsers/template_parser.py`, `src/parsers/target_parser.py`, `src/compliance/point_checker.py`.
* Parsing: Monolithic text extraction without block-level structural bounding boxes or typography awareness.
* Matching: Regex patterns, keyword heuristics, and basic semantic cosine similarity thresholds without cross-encoder NLI verification.
* Aggregation: Flat heuristic scoring without structured parent-child hierarchies or formal status taxonomies.
* Current Status: Fully functional and preserved under backward-compatibility guarantees; tested by baseline regression tests in `tests/test_compliance.py`.

### V2 Architecture (Production Pipeline)
* Core Pipeline: `src/pipeline/compliance_pipeline.py`.
* Parsing & Chunking: `src/parsers/pdf_parser.py` implementing `TargetPDFParser` to extract layout-aware `DocumentBlock` objects with coordinates, font characteristics, and structural types.
* Decomposition: `src/parsers/template_parser.py` decomposing parent sections into atomic `SubRequirement` specifications.
* Hybrid Retrieval: `src/retrieval/hybrid_retriever.py` fusing lexical BM25 and dense semantic search via Reciprocal Rank Fusion (RRF).
* Evidence Verification: `src/verification/evidence_verifier.py` evaluating evidence via DeBERTa-v3 cross-encoder NLI with specialized linguistic refinement passes.
* Hierarchical Aggregation: `src/verification/compliance_aggregator.py` computing parent compliance states and document compliance verdicts.
* Presentation: `src/reporting/formatter.py` generating terminal, markdown, and JSON compliance reports.
* Benchmark System: `src/benchmarks/` providing a declarative JSON Schema Draft 2020-12, typed data models, and a three-layer validation engine.


## 3. End-to-End Pipeline Workflow

The V2 compliance evaluation runs through a seven-stage deterministic workflow:

1. Input Ingestion & Pre-flight Validation:
   The pipeline receives the reference template PDF (specifying required standards) and the candidate target SRS PDF. Files are verified for existence, non-emptiness, and readability.

2. Document Parsing & Layout Block Extraction:
   `TargetPDFParser` processes the target document using PyMuPDF (`fitz`), grouping contiguous text spans into `DocumentBlock` objects. Headers, numbered lists, bullet points, and disclaimer callouts are classified into `BlockType` enums.

3. Requirement Parsing & Atomic Decomposition:
   `TemplateParser` parses the reference document into `ParentRequirement` objects, each containing one or more atomic `SubRequirement` items. Each sub-requirement contains a typed claim, mandatory/optional flag, and expected evidence category.

4. Hybrid Candidate Retrieval:
   For every atomic sub-requirement claim, `HybridRetriever` runs BM25 lexical search and MiniLM dense semantic search across all document blocks. Candidate ranks are fused using Reciprocal Rank Fusion (RRF) with smoothing parameter k=60 to produce the Top-K candidate evidence passages.

5. NLI Evidence Verification:
   `EvidenceVerifier` evaluates each candidate passage against the sub-requirement claim using a cross-encoder NLI model. Candidate scores are processed through domain-specific refinement rules to assign an atomic status: `FULFILLED`, `EXPLICIT_OMISSION`, `CONTRADICTED`, `MISSING`, or `NEEDS_REVIEW`.

6. Hierarchical Compliance Aggregation:
   `ComplianceAggregator` rolls up atomic verification results into parent section compliance statuses, recalculates mandatory and optional coverage percentages, and assigns the document-level compliance verdict.

7. Multi-Format Report Formatting:
   `V2ReportFormatter` serializes the final `DocumentComplianceReport` into terminal summary output, GitHub Flavored Markdown reports, and machine-readable JSON documents.


## 4. Retrieval Subsystem Architecture

The retrieval subsystem (`src/retrieval/`) addresses the challenge of vocabulary mismatch and structural variation between template requirements and target document text.

### Lexical Retrieval (`BM25SearchEngine`)
* Implementation: Rank-BM25 over whitespace and punctuation-tokenized text.
* Role: Captures exact identifier matches, domain-specific acronyms (e.g., SRS, GUI, API, ISO, IEEE), section numbers, and technical terminology.
* Normalization: Case-insensitive tokenization with stopword filtering.

### Dense Semantic Retrieval (`DenseSearchEngine`)
* Implementation: Sentence-Transformers using `sentence-transformers/all-MiniLM-L6-v2`.
* Role: Captures semantic paraphrasing, conceptual synonyms, and indirect requirement fulfillments where exact terminology differs.
* Indexing: Pre-computed block embeddings normalized for cosine similarity computation.

### Reciprocal Rank Fusion (RRF)
* Formula: `RRF_Score = 1.0 / (k + rank_bm25) + 1.0 / (k + rank_dense)`
* Smoothing Parameter: `k = 60` (standard information retrieval setting).
* Top-K Selection: Ranks all candidate blocks and selects the top candidate passages (default: Top-3) for NLI verification.
* Provenance Preservation: Each candidate retains its source `block_id`, `page_number`, bounding box coordinates, and raw text.


## 5. Verification Subsystem Architecture

The verification subsystem (`src/verification/`) evaluates whether retrieved candidate evidence actually fulfills, contradicts, or explicitly omits a requirement.

### Cross-Encoder NLI Engine
* Base Model: `cross-encoder/nli-deberta-v3-small` (or `cross-encoder/nli-deberta-v3-base`).
* Input Format: Text pair `(premise = evidence_text, hypothesis = claim_text)`.
* Output Distribution: Softmax probabilities over `[entailment, neutral, contradiction]`.

### Specialized Domain Refinements (Phase 2.3.1)
* Boundary Clause Isolation: Isolates negative exclusionary clauses in requirement scopes (e.g., "The subsystem does not control secondary hydraulic actuators") so that they do not trigger false contradictions against positive claims.
* Meta-Discourse Sensitivity Resolution: Recognizes introductory and rhetorical patterns ("This document describes...", "Section 3 outlines...") as valid substantive evidence rather than procedural commentary.
* False Contradiction Resolution: Distinguishes between genuine factual opposition and partial coverage or implementation variances.
* Explicit Omission Classification: Detects explicit omission disclaimers (e.g., "This requirement is intentionally omitted per project scope") via `is_explicit_omission` block attributes and maps them to `EXPLICIT_OMISSION` status.
* Numeric Target Validation: Verifies that requirements tagged with `expected_evidence_type = NUMERIC_TARGET` contain concrete numerical figures and matching engineering units.


## 6. Aggregation & Scoring Architecture

The aggregation subsystem (`src/verification/compliance_aggregator.py`) converts granular atomic verification results into executive audit metrics.

### Atomic Verification Statuses
* `FULFILLED`: The target document contains clear, entailing evidence supporting the claim.
* `EXPLICIT_OMISSION`: The target document contains an explicit, documented statement that the requirement is omitted or out of scope.
* `CONTRADICTED`: The target document contains statements that directly conflict with or violate the requirement.
* `MISSING`: The target document contains no relevant or entailing evidence for the requirement.
* `NEEDS_REVIEW`: The evidence is ambiguous, contains conflicting signals, or falls into confidence edge cases requiring human inspection.

### Mandatory and Optional Coverage Metrics
* Mandatory Total Constraint: Every valid benchmark pair must contain at least one mandatory atomic sub-requirement (`mandatory_total >= 1`).
* Mandatory Coverage Percentage: `round((mandatory_fulfilled / mandatory_total) * 100.0, 1)`
* Optional Coverage Percentage: If `optional_total > 0`, `round((optional_fulfilled / optional_total) * 100.0, 1)`; if `optional_total == 0`, evaluates to `None`.

### Parent Status Aggregation Rules
For parents containing at least one mandatory sub-requirement:
1. If any mandatory child has `NEEDS_REVIEW`, parent status is `NEEDS_REVIEW`.
2. Otherwise, if all mandatory children are `FULFILLED` and none are `CONTRADICTED`, parent status is `FULLY_COMPLIANT`.
3. Otherwise, if at least one mandatory child is `FULFILLED`, parent status is `PARTIALLY_COMPLIANT`.
4. Otherwise, parent status is `NON_COMPLIANT`.

For optional-only parent sections (no mandatory children):
1. If any optional child has `NEEDS_REVIEW`, parent status is `NEEDS_REVIEW`.
2. Otherwise, if all optional children are `FULFILLED`, parent status is `FULLY_COMPLIANT`.
3. Otherwise, if at least one optional child is `FULFILLED`, parent status is `PARTIALLY_COMPLIANT`.
4. Otherwise, parent status is `NON_COMPLIANT`.

### Document Compliance Verdict Rules
* `CONDITIONALLY_COMPLIANT`: If any mandatory sub-requirement has status `NEEDS_REVIEW`.
* `COMPLIANT`: If mandatory coverage is 100.0% and no mandatory sub-requirement has status `CONTRADICTED`.
* `NON_COMPLIANT`: If mandatory coverage is below 100.0% or any mandatory sub-requirement has status `CONTRADICTED`.


## 7. CLI & Reporting Interface

The command-line interface (`main.py`) provides the primary user entrypoint for batch evaluation and reporting:

* Command Syntax: `python3 main.py --template <path> --target <path> [options]`
* Options:
  * `--output-dir <path>`: Directory where report artifacts are saved.
  * `--format {terminal,markdown,json,all}`: Output report formats.
  * `--dense-model <name>`: Custom sentence-transformers embedding model.
  * `--nli-model <name>`: Custom cross-encoder NLI model.
  * `--verbose`: Enables detailed debugging logs and candidate retrieval dumps.
* Report Artifacts:
  * Terminal Output: ANSI-colored executive summary with status counts, coverage percentages, and highlighted non-compliant sections.
  * Markdown Report (`compliance_report.md`): Formatted audit document ready for PR attachment or human review.
  * JSON Report (`compliance_report.json`): Machine-readable data structure conforming to the compliance report schema.


## 8. Benchmark System Architecture (Phase 2.6)

Phase 2.6 establishes a declarative ground-truth benchmark suite for validating pipeline accuracy across model upgrades.

### Schema Architecture (`benchmark_manifest.schema.json`)
* Specification Standard: JSON Schema Draft 2020-12.
* Structure:
  * `schema_version`: Semantic version string (e.g., `2.0.0`).
  * `metadata`: Metadata including `pair_id`, `domain`, `reference_document`, and `target_document` (with SHA-256 digests and page counts).
  * `parent_requirements`: Hierarchical array of parent sections and atomic child requirements.
  * `ground_truth.retrieval`: Expected evidence presence (`EXISTS`, `EXPLICIT_OMISSION`, `ABSENT`) and explicit evidence block annotations (`block_id`, `page`, optional `excerpt`).
  * `ground_truth.verification`: Expected verification status (`FULFILLED`, `EXPLICIT_OMISSION`, `CONTRADICTED`, `MISSING`, `NEEDS_REVIEW`) and expected confidence band.
  * `expected_document_compliance`: Expected mandatory/optional totals, fulfilled counts, status counts, coverage percentages, and document verdict.

### Typed Python Models (`src/models/benchmark.py`)
* Dataclasses: `BenchmarkManifest`, `BenchmarkMetadata`, `BenchmarkDocumentMetadata`, `BenchmarkParentRequirement`, `BenchmarkSubRequirement`, `BenchmarkGroundTruth`, `BenchmarkRetrievalGroundTruth`, `BenchmarkVerificationGroundTruth`, `BenchmarkEvidenceAnnotation`, `BenchmarkExpectedCompliance`, `BenchmarkExpectedStatusCounts`.
* Serialization: Strict `to_dict()` and `from_dict()` methods with zero silent type coercion and direct key access for schema-required fields.

### Three-Layer Validator (`src/benchmarks/manifest_validator.py`)
* Layer A (JSON Schema Validation): Validates raw dictionary data against `benchmark_manifest.schema.json` using `jsonschema.Draft202012Validator`.
* Layer B (Cross-Field Business Logic): Validates ID uniqueness, `{parent_id}.` sub-requirement ID prefix hierarchy, non-empty parents, parent mandatory consistency, evidence presence compatibility, status conservation, coverage math, parent aggregation waterfall, and document verdict consistency.
* Layer C (Document Integrity & Grounding): Resolves reference and target PDFs, verifies 64-character SHA-256 hashes, validates total page counts, executes `TargetPDFParser` to build a live block map, verifies block ID existence, checks page number consistency, and confirms whitespace-normalized excerpt text grounding.


## 9. Current Development Status

The project is currently at the boundary of Phase 2.6.2.2 and Phase 2.6.2.3:

* Phase 2.5 (V2 Pipeline & CLI Integration): Complete, tested, committed (`4e671b1`), and pushed to GitHub main.
* Phase 2.6.1 (Ground Truth Gap Analysis): Complete; evaluated existing test packs Pair A and Pair B.
* Phase 2.6.2.1 (Schema Specification): Complete; Draft 2020-12 JSON Schema finalized and approved.
* Phase 2.6.2.2 (Declarative Validator Implementation): Complete and committed (`ed4c95f`).
  * Created `src/benchmarks/schemas/benchmark_manifest.schema.json`.
  * Created `src/models/benchmark.py`.
  * Updated `src/models/__init__.py`.
  * Created `src/benchmarks/manifest_validator.py` and `src/benchmarks/__init__.py`.
  * Created `tests/test_v2_benchmark_validator.py` (32 tests passing).
* Phase 2.6.2.3 (Declarative Manifest Migration for Pair A and Pair B): Complete in local workspace.
  * Created `data/benchmarks/pair_a_lab_report.json`: Pair A manifest with 10 sub-requirements, 100% verified across Layers A, B, and C.
  * Created `data/benchmarks/pair_b_srs.json`: Pair B manifest with 12 sub-requirements, 100% verified across Layers A, B, and C.
  * Created `tests/test_benchmark_manifests.py`: 6 tests covering end-to-end manifest validation, ground truth verification, and tamper detection.
  * Total test count increased from 138 to 144 tests passing.
* Phase 2.6.3 (Automated Benchmark Evaluation CLI & Accuracy Runner): Ready to begin upon user approval.


## 10. Test Suite & Quality Assurance

The test suite provides 100% automated coverage across all subsystems:

### Test Modules
* `tests/test_compliance.py`: V1 baseline point checker, template extraction, and regression packs (33 tests).
* `tests/test_v2_retrieval.py`: BM25 tokenization, dense semantic encoding, RRF fusion, and candidate provenance (5 tests).
* `tests/test_v2_verification.py`: Cross-encoder NLI, boundary clauses, meta-discourse, false contradiction resolution, and numeric targets (11 tests).
* `tests/test_v2_aggregation.py`: Parent compliance aggregation, document verdicts, status conservation, and coverage formulas (38 tests).
* `tests/test_v2_reporting.py`: Terminal, markdown, and JSON report formatters, Unicode preservation, and escaping (19 tests).
* `tests/test_v2_benchmark_validator.py`: Layer A, Layer B, Layer C, and end-to-end validator unit tests (32 tests).
* `tests/test_benchmark_manifests.py`: Production Pair A and Pair B manifest validation, ground truth verification, and tamper detection (6 tests).

### Verification Execution
* Execution Command: `python3 -m unittest discover -s tests -p "test_*.py" -v`
* Result: 144 tests ran in ~18 seconds with 0 failures, 0 errors, and 0 skips.


## 11. Engineering Invariants & Guardrails

All future development must strictly maintain the following invariants:

* Invariant 1 (Mandatory Requirement Minimum): Every benchmark manifest and compliance pipeline run must contain at least one mandatory atomic sub-requirement (`mandatory_total >= 1`).
* Invariant 2 (Status Conservation): The sum of atomic sub-requirement status counts must exactly equal the total number of atomic sub-requirements.
* Invariant 3 (Hierarchical Traceability): Every atomic sub-requirement ID must be prefixed with its parent section ID (`{parent_id}.{sub_num}`).
* Invariant 4 (Evidence Presence Taxonomy):
  * `EXISTS`: Requires at least one expected evidence block annotation.
  * `EXPLICIT_OMISSION`: Requires at least one disclaimer evidence block annotation.
  * `ABSENT`: Requires `expected_evidence` list to be strictly empty.
* Invariant 5 (Document Grounding Integrity): Every referenced `block_id` must physically exist in the parsed target PDF, its declared page number must match the block page number, and any declared excerpt must be a normalized substring of the block text.
* Invariant 6 (Strict Mathematical Coverage): Coverage percentages must strictly equal `round((fulfilled / total) * 100.0, 1)`.
* Invariant 7 (No Markdown Tables Rule): All project documentation, generated reports, and communication must use structured bullet lists, code blocks, and indented text rather than markdown tables.


## 12. Known Technical Debt & Low-Priority Backlog

* Google Drive Sync Artefacts: The repository is located inside a Google Drive synced directory (`AI - SRS - Compliance`), resulting in periodic generation of `.gdoc` metadata files. These are now safely excluded via `.gitignore`.
* PyTorch MacOS Deprecation Notices: Running on macOS with Python 3.13 emits minor runtime notices regarding `torch.distributed.elastic` and `torch.jit.script` deprecation in PyTorch 2.6. These do not affect inference execution.
* Model Cold-Start Latency: The first execution of the dense embedding model and cross-encoder NLI model downloads and caches weights in `~/.cache/huggingface/`, which requires network access during initialization.
* Legacy Code Harmonization: Legacy V1 classes in `src/compliance/` remain for backwards compatibility; future phases may deprecate them once V2 benchmark parity is fully established.


## 13. Immediate Next Steps: Phase 2.6.3 Plan

With Phase 2.6.2.3 complete, the authoritative declarative benchmarks for Pair A and Pair B are established. The next phase is **Phase 2.6.3: Automated Benchmark Evaluation CLI & Accuracy Runner**:

1. Implement Benchmark Evaluator Runner:
   * Create `src/benchmarks/benchmark_runner.py` (or CLI `scripts/evaluate_manifest_benchmark.py`).
   * Runs the V2 `CompliancePipeline` against the benchmark pair reference and target documents.
   * Compares actual pipeline outputs against ground truth from the declarative manifest.

2. Metrics & Accuracy Evaluation:
   * Retrieval Metrics: Precision@K, Recall@K, Mean Reciprocal Rank (MRR) for expected evidence blocks.
   * Verification Metrics: Classification Accuracy, Macro-F1 score across statuses (`FULFILLED`, `EXPLICIT_OMISSION`, `CONTRADICTED`, `MISSING`, `NEEDS_REVIEW`).
   * Aggregation Metrics: Exact match on Parent Compliance Statuses and Document Verdict.

3. Detailed Diagnostics & Gap Reporting:
   * Generates diagnostic reports pinpointing discrepancies between ground truth annotations and model predictions (retrieval misses, NLI confidence edge cases, contradictory evidence conflicts).


