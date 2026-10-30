# SRS Compliance Analyzer

> **AI-powered Software Requirements Specification (SRS) verification system**  
> Combines deterministic rule checks with `sentence-transformers` semantic embeddings to automate SRS quality analysis against reference standards.

---

## Features

| Version | Capability |
|---|---|
| **Phase 1** | **Reference Template Content Checker** — Extract required points/sections from template PDF; compare target PDF (whole document or `--page N`) using `all-MiniLM-L6-v2`; classify as COVERED, PARTIAL, or MISSING with nearest text evidence |
| **v0.4** | PDF input support — extract text, sections, and requirements from digital PDFs; graceful scanned-PDF detection |
| **v0.3** | Semantic requirement-to-rule matching — assigns each requirement the best-matching compliance rule |
| **v0.2** | Semantic section alignment — maps SRS sections to standard sections using `all-MiniLM-L6-v2` cosine similarity |
| **v0.1** | Deterministic R1–R4 rule checks: unique IDs, `shall` enforcement, measurability, vague-language detection |

### What the analyser checks (R1–R4)

| Rule | Name | What it checks |
|---|---|---|
| **R1** | Unique Requirement ID | Every requirement has a unique ID (`REQ-001`, `REQ-FDM-001`, …) |
| **R2** | Clear Behaviour | Statement uses `shall` (not `should`, `may`, `will`) |
| **R3** | Measurable Criteria | Quantitative threshold present (`50 ms`, `20 Hz`, `0.5 s`) |
| **R4** | Avoid Vague Language | No subjective terms (`quickly`, `fast`, `easy`, `adequate`, `user-friendly`) |

---

## Project Structure

```
AI - SRS - Compliance/
├── data/
│   ├── reference_standard.json      # Default standard: SEC-1–4 and R1–R4
│   ├── reference_standard.pdf       # PDF version of the reference standard
│   ├── sample_srs.txt               # Baseline 5-requirement SRS (v0.1 benchmark)
│   ├── sample_srs.pdf               # PDF version of sample_srs.txt
│   ├── avionics_sample_srs.md       # Extended avionics SRS (multi-section)
│   ├── avionics_sample_srs.pdf      # PDF version of the avionics SRS
│   └── scanned_sample_srs.pdf       # Simulated scanned/image-only PDF (0 text)
├── src/
│   ├── __init__.py                  # Package exports
│   ├── analyzer.py                  # Core engine — rules, semantic matching, reports
│   └── pdf_extractor.py             # PDF text extraction & scanned-PDF detection (v0.4)
├── poc/
│   ├── srs_compliance_analyzer.py   # Early proof-of-concept (TF-IDF based)
│   └── run_poc.py                   # POC runner
├── scripts/
│   └── generate_sample_pdfs.py      # Regenerate test PDF fixtures (requires ReportLab)
├── tests/
│   └── test_compliance.py           # 21-test suite across v0.1–v0.4
├── demo_pdf_support.py              # v0.4 PDF demo
├── demo_rule_matching.py            # v0.3 rule-matching demo
├── demo_semantic_matching.py        # v0.2 section-alignment demo
├── main.py                          # CLI entry point
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Installation

**Python 3.10+** is recommended.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/srs-compliance-analyzer.git
cd srs-compliance-analyzer

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note on model weights:** `sentence-transformers` downloads `all-MiniLM-L6-v2` (~22 MB) from HuggingFace on first run and caches it in `~/.cache/huggingface`. No weights are stored inside this repository.

---

## Usage

### Reference-Template Content Compliance Check (Phase 1)
```bash
# Check default target (sample_srs.pdf) against reference template (reference_standard.pdf)
python3 main.py

# Check custom target PDF against a reference template PDF
python3 main.py --template data/sample_srs.pdf --target data/avionics_sample_srs.pdf

# Check only a specific page of the target PDF (1-based index)
python3 main.py --template data/sample_srs.pdf --target data/avionics_sample_srs.pdf --page 1

# Configure cosine similarity decision thresholds
python3 main.py --template data/sample_srs.pdf --target data/avionics_sample_srs.pdf --covered-threshold 0.75 --partial-threshold 0.45

# Export Markdown and JSON reports
python3 main.py --template data/reference_standard.pdf --target data/sample_srs.pdf --output-md report.md --output-json report.json
```

### Legacy Quality Rules Audit (v0.1–v0.4 R1–R4 Style Linter)
```bash
# Run deterministic R1-R4 rules and section alignment
python3 main.py --legacy-rules --srs data/sample_srs.txt --standard data/reference_standard.json
```

### Adjust the section-alignment similarity threshold
```bash
python3 main.py --srs data/avionics_sample_srs.md --threshold 0.6
```

### Try a scanned / image-only PDF (no extractable text)
```bash
python3 main.py --srs data/scanned_sample_srs.pdf
# → Exits with code 2 and prints an OCR advisory. No crash.
```

### Save reports
```bash
python3 main.py --srs data/avionics_sample_srs.md --output-json report.json --output-md report.md
```

### Run demos
```bash
python3 main.py --demo          # v0.4 PDF input demo
python3 main.py --demo-rules    # v0.3 rule-matching demo
python3 main.py --demo-sections # v0.2 section-alignment demo
python3 main.py --demo-all      # All three demos
```

---

## Running Tests

```bash
python3 -m unittest tests/test_compliance.py -v
```

Expected output:
```
Ran 21 tests in ~27s
OK
```

| Test Class | Tests | Covers |
|---|---|---|
| `TestVersion01Rules` | 5 | R1–R4 deterministic checks + sample_srs.txt benchmark |
| `TestVersion02SemanticSectionMatching` | 3 | Section alignment, pairwise similarity |
| `TestVersion03SemanticRuleMatching` | 7 | Rule loading, paraphrase, compliant/vague/irrelevant inputs, batch, end-to-end |
| `TestVersion04PDFSupport` | 6 | Text extraction, avionics sections/requirements, standard parsing, scanned detection, PDF-to-PDF audit |

---

## Regenerating Test PDFs

The PDF fixtures in `data/` are pre-generated and committed. To regenerate them:

```bash
pip install reportlab Pillow
python3 scripts/generate_sample_pdfs.py
```

---

## Backward Compatibility

- `sample_srs.txt` and `sample_srs.pdf` always produce **3 Compliant / 2 Non-Compliant (60.0%)**.
- All v0.1 R1–R4 rule checks are deterministic and unchanged.
- Section alignment threshold and vague-word lists are unchanged from v0.2.
- Semantic rule matching from v0.3 runs on every requirement in every audit.

---

## Roadmap

- **v0.5** — OCR integration (Tesseract / PaddleOCR) for scanned PDFs
- **v0.6** — NLI / DeBERTa for formal regulatory entailment checking
- **v0.7** — Retrieval-Augmented Generation (RAG) over avionics standards (DO-178C, ARP4754A)

---

## License

This project is released for educational and research purposes.
