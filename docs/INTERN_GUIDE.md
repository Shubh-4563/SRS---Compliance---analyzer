# AI-SRS Compliance Analyzer — Intern Architecture & Onboarding Guide

Target Audience: Incoming Software Engineers, AI/ML Interns, and Project Stakeholders  
Repository: https://github.com/Shubh-4563/SRS---Compliance---analyzer.git  
Test Status: 151/151 tests passing  
PDF Guide: `AI_SRS_Compliance_Intern_Guide.pdf`


## 1. What Does This Project Do? (Plain English)

When software and systems engineering teams build complex applications (such as monitoring dashboards, medical devices, or aerospace control software), they write a System Requirements Specification (SRS) or a technical report.

Companies and regulatory bodies have strict checklists and reference standards (e.g. IEEE, ISO, safety guidelines, lab rubrics) that these documents must follow.

Traditionally, auditing an SRS is a painful, slow manual process:
* An engineer or auditor sits with a 20-page document and a checklist.
* They read line-by-line to find proof for every requirement.
* They check whether numbers match, whether features were omitted, or if anything contradicts the rules.
* They manually calculate coverage percentages and write an audit report.

The AI-SRS Compliance Analyzer automates this entire audit process:
* Input 1 (Reference Template PDF): The checklist, standard, or rubric containing required sections and rules.
* Input 2 (Target Document PDF): The candidate SRS, lab report, or technical specification being audited.
* Automated Output: An executive compliance report with an overall verdict (COMPLIANT, CONDITIONALLY_COMPLIANT, NON_COMPLIANT), mandatory coverage percentage, and line-by-line evidence citations with exact page numbers and quotes.


## 2. End-to-End System Architecture

The system executes a deterministic 5-stage pipeline orchestrated by `src/pipeline.py`:

```
+---------------------------------------------------------------------------------+
|                             END-TO-END PIPELINE FLOW                            |
+---------------------------------------------------------------------------------+
|  Reference Template PDF                     Target Document PDF                 |
|            |                                         |                          |
|            v                                         v                          |
|  [1. Template Ingestion]                    [1. Target PDF Parsing]             |
|  (PDFExtractor & TemplateParser)            (TargetPDFParser)                   |
|            |                                         |                          |
|            v                                         v                          |
|  [2. Atomic Decomposition]                  [Structured DocumentBlocks]         |
|  (Splits into SubRequirements)              (p1_b1, p1_b2, p1_b3 with bbox)     |
|            \                                         /                          |
|             \                                       /                           |
|              v                                     v                            |
|     +-----------------------------------------------------+                     |
|     | [3. Hybrid Candidate Retrieval (HybridRetriever)]    |                     |
|     | - BM25 Lexical Keyword Search                       |                     |
|     | - Dense Semantic Embeddings (all-MiniLM-L6-v2)      |                     |
|     | - Reciprocal Rank Fusion (RRF: k=60)                |                     |
|     +-----------------------------------------------------+                     |
|                                |                                                |
|                                v                                                |
|     +-----------------------------------------------------+                     |
|     | [4. NLI Evidence Verification (EvidenceVerifier)]   |                     |
|     | - Cross-Encoder Transformer (nli-deberta-v3-small)  |                     |
|     | - Premise = Candidate Block, Hypothesis = Claim    |                     |
|     | - Refinements: Boundary Clauses, Meta-Discourse     |                     |
|     | - Classifies: FULFILLED, CONTRADICTED, OMISSION, ...|                     |
|     +-----------------------------------------------------+                     |
|                                |                                                |
|                                v                                                |
|     +-----------------------------------------------------+                     |
|     | [5. Compliance Aggregation (ComplianceAggregator)]  |                     |
|     | - Rolls up sub-results to parent sections           |                     |
|     | - Computes mandatory & optional coverage %          |                     |
|     | - Computes document verdict (COMPLIANT / NON)       |                     |
|     +-----------------------------------------------------+                     |
|                                |                                                |
|                                v                                                |
|     [Terminal ANSI Summary] [compliance_report.md] [compliance_report.json]     |
+---------------------------------------------------------------------------------+
```


## 3. Deep Dive into the 5 Pipeline Stages

### Stage 1: Document Parsing & Layout Chunking (`src/parsers/pdf_parser.py`)
Instead of extracting messy raw text strings, `TargetPDFParser` analyzes the visual and typography layout of the PDF.
* It groups contiguous spans of text into cohesive `DocumentBlock` objects.
* It assigns each block a deterministic ID like `p1_b3` (Page 1, Block 3).
* It classifies blocks into semantic types: `HEADING`, `PARAGRAPH`, `LIST_ITEM`, `TABLE_ROW`, or `DISCLAIMER`.

```python
@dataclass
class DocumentBlock:
    block_id: str                          # e.g., 'p1_b3'
    text: str                              # Clean text content
    block_type: BlockType                  # PARAGRAPH, HEADING, DISCLAIMER
    page_number: int                       # 1-based source page
    section_heading: Optional[str] = None  # Parent section title
    is_explicit_omission: bool = False     # True if text states 'not included'
```

### Stage 2: Requirement Decomposition (`src/parsers/requirement_decomposer.py`)
Human requirements in a checklist often contain multiple testable claims (e.g. "State the purpose and describe what it is intended to measure").
* The `RequirementDecomposer` breaks each parent point into atomic `SubRequirement` claims.
* Each sub-requirement is tagged with an expected evidence type:
  * `FACTUAL_STATEMENT`: Descriptive prose or factual affirmations.
  * `NUMERIC_TARGET`: Concrete numbers and units (e.g., "70 degrees C", "less than 200 ms").
  * `SAFETY_CONSTRAINT`: Hazard mitigation, fault warnings, or safety precautions.
  * `INTERFACE_SPECIFICATION`: User roles, hardware gateways, or communication protocols.

```python
@dataclass
class SubRequirement:
    sub_id: str                            # e.g., 'B1.1', 'B1.2'
    parent_point_id: str                   # e.g., 'B1'
    claim_text: str                        # Atomic verifiable hypothesis
    expected_evidence_type: EvidenceType   # FACTUAL_STATEMENT, NUMERIC_TARGET...
    is_mandatory: bool = True              # Mandatory vs Optional flag
    decomposition_rationale: str = ""
```

### Stage 3: Hybrid Retrieval with Reciprocal Rank Fusion (`src/retrieval/`)
How do we find the relevant paragraph in a 10-page document?
* Lexical Search (`BM25SearchEngine`): Finds exact keywords, section codes, and acronyms (e.g. "GUI", "ISO 26262", "70 degrees C").
* Dense Semantic Search (`DenseSearchEngine` via `all-MiniLM-L6-v2`): Finds concepts and synonyms even when wording differs (e.g. "cooling rate" matches "water temperature decrease").
* Reciprocal Rank Fusion (RRF): Merges both rankings with smoothing parameter k=60 so the best candidate passages rise to the top:
  `RRF_Score = 1.0 / (60 + rank_bm25) + 1.0 / (60 + rank_dense)`

### Stage 4: AI Evidence Verification via NLI (`src/verification/evidence_verifier.py`)
Keyword matching can be fooled easily. The analyzer uses a Natural Language Inference (NLI) cross-encoder transformer (`nli-deberta-v3-small`).
It evaluates the candidate block as the premise and the requirement claim as the hypothesis:
* `FULFILLED`: Entailment probability is high (the document proves the claim).
* `CONTRADICTED`: Contradiction probability is high (the document says the opposite).
* `EXPLICIT_OMISSION`: The document contains a disclaimer stating the requirement is out of scope.
* `MISSING`: No candidate evidence entails the requirement.
* `NEEDS_REVIEW`: Ambiguous or conflicting signals flagged for human auditor review.

### Stage 5: Hierarchical Compliance Aggregation (`src/verification/compliance_aggregator.py`)
Aggregates atomic sub-requirements up to parent sections and determines the document verdict:
* Mandatory Coverage Percentage: `round((mandatory_fulfilled / mandatory_total) * 100.0, 1)`
* Document Verdict Rules:
  * `CONDITIONALLY_COMPLIANT`: If any mandatory sub-requirement has status `NEEDS_REVIEW`.
  * `COMPLIANT`: If mandatory coverage is 100.0% and no mandatory requirement is `CONTRADICTED`.
  * `NON_COMPLIANT`: If mandatory coverage is below 100.0% or any mandatory requirement is `CONTRADICTED`.


## 4. The Declarative Benchmark Suite (Phase 2.6)

To ensure that future changes never degrade compliance accuracy, the project includes an automated benchmark system:
* Production Manifests:
  * `data/benchmarks/pair_a_lab_report.json`: Laboratory Science cooling experiment benchmark (10 sub-requirements).
  * `data/benchmarks/pair_b_srs.json`: Software Engineering Equipment Monitoring Dashboard benchmark (12 sub-requirements).
* Three-Layer Validator (`src/benchmarks/manifest_validator.py`):
  * Layer A: JSON Schema Draft 2020-12 structural validation.
  * Layer B: Business logic invariants (mandatory_total >= 1, ID hierarchy, status conservation).
  * Layer C: Physical document integrity, SHA-256 hash checks, and word-for-word text excerpt grounding.
* Automated Accuracy Runner (`src/benchmarks/benchmark_runner.py`):
  * Automatically executes the pipeline on both benchmark pairs and compares predictions against declared ground truth.
  * Measures Hit@1..5, MRR, Accuracy (100% on both pairs), and logs zero discrepancies.


## 5. Step-by-Step Developer Guide: How to Run & Test

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/Shubh-4563/SRS---Compliance---analyzer.git
cd SRS---Compliance---analyzer

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run the Full Test Suite
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
# Expected: Ran 151 tests in ~25-30s ... OK (100% passing)
```

### Step 3: Run the Live Pipeline on Pre-Built Pairs
```bash
# Run Pair B (Technical Specification vs Template)
python3 main.py \
  --template data/test_packs/B_reference_technical_template.pdf \
  --target data/test_packs/B_target_technical_document.pdf \
  --format all

# Run Pair A (Laboratory Report vs Template)
python3 main.py \
  --template data/test_packs/A_reference_lab_report_template.pdf \
  --target data/test_packs/A_target_lab_report.pdf \
  --format all
```

Outputs generated:
* Terminal ANSI summary.
* `compliance_report.md` (formatted Markdown audit document).
* `compliance_report.json` (machine-readable data with exact quotes and page numbers).

### Step 4: Run the Automated Benchmark Accuracy Runner
```bash
python3 scripts/evaluate_manifest_benchmark.py --all
# Evaluates Pair A and Pair B against ground truth, outputting 100% accuracy.
```

### Step 5: Test a New Document from Ma'am or Customer
```bash
python3 main.py \
  --template path/to/reference_checklist.pdf \
  --target path/to/candidate_document.pdf \
  --output-dir reports/my_audit/ \
  --format all
```


## 6. Future Development Roadmap for the Next Intern

High-priority directions for the next phase of development:
* OCR Integration (Tesseract / EasyOCR): Add support for scanned paper PDFs or photos of requirements documents (currently supports digital PDFs with selectable text).
* Large Document Scaling (Pair C & D): Benchmark against multi-page industrial SRS documents (50+ pages) to stress-test hybrid retrieval candidate ranking.
* Interactive Web UI (Streamlit / FastHTML): Build a drag-and-drop web dashboard where users can upload two PDFs and view interactive compliance findings.
* Model Optimization: Test quantized or ONNX models (e.g. MiniLM-L6 ONNX) to decrease CPU inference latency for faster local audits.

