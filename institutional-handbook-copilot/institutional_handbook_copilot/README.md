# Institutional Handbook Copilot
**Version-Aware RAG with Conflict Detection**

An enterprise-grade, privacy-first Retrieval-Augmented Generation (RAG) system for university handbooks, academic regulations, employee manuals, and compliance guidelines. It understands document version chronology, automatically detects contradictory policies across revisions, prioritizes the newer version, and explicitly cites both the current and superseded regulations.

---

## Key Features

1. **Strictly Grounded Retrieval**: Answers only from uploaded institutional PDFs—no hallucinations.
2. **Chronological Version Inference**: Automatically detects publication dates, revision numbers, and academic years from document metadata, headers, and filenames.
3. **Automated Conflict Detection**: Identifies contradicting numbers, percentages, limits, eligibility criteria, and deadlines across editions (e.g. 75% attendance in 2025 vs 80% in 2026).
4. **Transparent Resolution**: Answers with the current valid policy, clearly flags `"This contradicts an older version."`, and renders dual citations for the current and historical policies.
5. **Privacy & Offline First**: Zero paid cloud APIs. Runs locally on CPU using `all-MiniLM-L6-v2`, `FAISS`, and local Ollama `llama3.2:3b`.

---

## Project Structure

```text
institutional_handbook_copilot/
│
├── app.py                     # Streamlit web user interface
├── core.py                    # RAG pipeline, version detection, conflict engine
├── demo_test.py               # Automated verification and evaluation test suite
├── generate_sample_docs.py    # Sample 2025 & 2026 handbook generator
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation & run guide
├── sample_docs/               # Sample 2025 and 2026 test documents
│   ├── handbook_2025.pdf
│   └── handbook_2026.pdf
├── data/                      # Uploaded PDFs storage
└── cache/                     # Embedding cache
```

---

## Installation & Setup

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Pull Ollama Model (Local LLM)
```bash
ollama pull llama3.2:3b
```

### 3. Launch Streamlit UI
```bash
python -m streamlit run app.py
```

### 4. Run Automated Demo & Verification Test
```bash
python demo_test.py
```
