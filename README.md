# 📘 Institutional Handbook Copilot
### Version-Aware RAG with Conflict Detection & Dual Citations

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20App-blue?style=for-the-badge&logo=google-chrome)](https://ais-pre-h5sru4mtkvoqqk74e63lmd-641540487682.asia-southeast1.run.app)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![FAISS + MiniLM](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-orange?style=for-the-badge)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Local LLM](https://img.shields.io/badge/LLM-Ollama%20llama3.2%3A3b-black?style=for-the-badge)](https://ollama.com)

---

## 🔗 Live Interactive Demo

* **Shared Application Preview URL:** [https://ais-pre-h5sru4mtkvoqqk74e63lmd-641540487682.asia-southeast1.run.app](https://ais-pre-h5sru4mtkvoqqk74e63lmd-641540487682.asia-southeast1.run.app)
* **Development Preview URL:** [https://ais-dev-h5sru4mtkvoqqk74e63lmd-641540487682.asia-southeast1.run.app](https://ais-dev-h5sru4mtkvoqqk74e63lmd-641540487682.asia-southeast1.run.app)

---

## 📌 Problem Statement

Institutions, universities, government bodies, and enterprises publish multi-version PDFs over successive academic years or fiscal cycles:
* University handbooks & student codes of conduct
* Academic progression & backlog regulations
* Compliance policies & grant guidelines
* Human resources & employee benefits manuals

Standard naive RAG systems silently retrieve whatever text has the highest embedding similarity, often blending superseded regulations with active ones or hallucinating outdated requirements.

**Institutional Handbook Copilot** solves this by:
1. **Accurate answers grounded strictly in uploaded documents** (never hallucinating outside knowledge).
2. **Page-level citations** for all factual assertions.
3. **Automated document version detection & chronological ranking**.
4. **Deterministic contradiction detection** across handbook revisions (detecting changed percentages, numerical limits, deadlines, and eligibility criteria).
5. **Conflict resolution prioritization:** When a newer edition contradicts an older edition, the Copilot answers using the newest policy, explicitly flags `⚠️ This contradicts an older version.`, and shows side-by-side citations for both the current policy and the older superseded policy.
6. **Never silently hiding or masking the older historical policy.**

---

## 🏛️ Core Architecture

```text
Uploaded PDFs (100+ pages supported)
    │
    ▼
[Text Extraction (PyMuPDF / Native Stream Extractor)]
    │ ── Page-aware chunking preserving page number & document name
    │
    ▼
[Version Inference Engine]
    │ ── Priority Hierarchy: Effective Date > Publication Date > Explicit Version > Academic Year > Filename
    │
    ▼
[Dense Embeddings & Vector Index]
    │ ── all-MiniLM-L6-v2 + L2 Normalization + FAISS Flat Inner Product
    │
    ▼
[Semantic Retrieval (8–12 Candidates)]
    │
    ▼
[Deterministic Conflict Detection Engine]
    │ ── Analyzes percentages (75% vs 80%), limits (4 backlogs vs 2), dates, and rule polarities
    │ ── Identifies contradictions between newer and older versions matching query topic
    │
    ▼
[Answer Synthesis (Ollama llama3.2:3b / Grounded Fallback)]
    │ ── Strict Grounded Prompting (answers with newest policy)
    │ ── Formats explicit contradiction warning
    │
    ▼
[Final Dual-Citation & Conflict Card Output]
```

---

## ⚡ Conflict Detection in Action

### Example 1: Attendance Requirement
* **Query:** `"What is the minimum attendance requirement?"`
* **Older Document:** `handbook_2025.pdf` — Page 1 (Minimum attendance: 75%)
* **Newer Document:** `handbook_2026.pdf` — Page 1 (Minimum attendance: 80%)

**Copilot Output:**
```text
The current minimum attendance requirement is 80%. (The older 2025 handbook specified 75%.)

⚠️ This contradicts an older version.

Current policy:
handbook_2026.pdf (2026) — Page 1

Older policy:
handbook_2025.pdf (2025) — Page 1
```

### Example 2: Backlog Limit for Promotion
* **Query:** `"How many backlogs are allowed for promotion?"`
* **Older Document:** `handbook_2025.pdf` — Page 1 (Up to 4 backlogs allowed)
* **Newer Document:** `handbook_2026.pdf` — Page 1 (Up to 2 backlogs allowed)

**Copilot Output:**
```text
The current policy allows up to 2 backlogs for promotion. (The older 2025 handbook permitted 4 backlogs.)

⚠️ This contradicts an older version.

Current policy:
handbook_2026.pdf (2026) — Page 1

Older policy:
handbook_2025.pdf (2025) — Page 1
```

### Example 3: Grounded Rejection (Out of Scope)
* **Query:** `"What is the cafeteria menu?"`
* **Copilot Output:**
```text
I could not find this information in the provided documents.
```

---

## 📂 Project Structure

```text
institutional_handbook_copilot/
│
├── app.py                     # Streamlit interactive web application
├── core.py                    # RAG pipeline, version detection, conflict engine
├── demo_test.py               # Automated verification and evaluation test suite
├── generate_sample_docs.py    # Generates 2025 & 2026 sample PDF handbooks
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── sample_docs/               # Sample versioned handbooks
│   ├── handbook_2025.pdf      # (75% attendance, 4 backlogs)
│   └── handbook_2026.pdf      # (80% attendance, 2 backlogs)
├── data/                      # Uploaded institutional PDF storage
└── cache/                     # Embedding cache
```

---

## 🚀 Installation & Local Execution

### 1. Install Python Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. (Optional) Run Local LLM via Ollama
```bash
ollama pull llama3.2:3b
ollama serve
```
*(Note: If Ollama is not running, the application automatically runs its deterministic grounded synthesis engine—zero setup friction, zero paid APIs).*

### 3. Launch the Streamlit Web Application
```bash
python -m streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Automated Demo & Test Suite

Run the test suite:
```bash
python demo_test.py
```

### Verified Test Output:
```text
======================================================================
INSTITUTIONAL HANDBOOK COPILOT — AUTOMATED DEMO TEST SUITE
======================================================================
[Step 1] Verified sample handbooks:
  • Older: sample_docs/handbook_2025.pdf
  • Newer: sample_docs/handbook_2026.pdf

[Step 2] Ingesting and indexing documents...
  • Ingested handbook_2025.pdf: Detected Version='Effective August 1, 2024', Chunks=3
  • Ingested handbook_2026.pdf: Detected Version='Effective August 1, 2025', Chunks=3
----------------------------------------------------------------------
RUNNING TEST_CASE_1: Conflict detection on numerical percentage (Attendance 75% -> 80%)
Query: "What is the minimum attendance requirement?"
----------------------------------------------------------------------
[Application Output]:
The current minimum attendance requirement is 80%. (The older Effective August 1, 2024 handbook specified 75%.)

⚠️ This contradicts an older version.

Current policy:
handbook_2026.pdf (Effective August 1, 2025) — Page 1

Older policy:
handbook_2025.pdf (Effective August 1, 2024) — Page 1

[Conflict Detected]: True
[Citations Provided]:
  - Current Policy: handbook_2026.pdf (vEffective August 1, 2025) Page 1
  - Older Conflicting Policy: handbook_2025.pdf (vEffective August 1, 2024) Page 1

✅ TEST_CASE_1 PASSED
----------------------------------------------------------------------
RUNNING TEST_CASE_2: Conflict detection on quantitative limit (Backlogs 4 -> 2)
Query: "How many backlogs are allowed for promotion?"
----------------------------------------------------------------------
[Application Output]:
The current policy allows up to 2 backlogs for promotion. (The older Effective August 1, 2024 handbook permitted 4 backlogs.)

⚠️ This contradicts an older version.

Current policy:
handbook_2026.pdf (Effective August 1, 2025) — Page 1

Older policy:
handbook_2025.pdf (Effective August 1, 2024) — Page 1

[Conflict Detected]: True
[Citations Provided]:
  - Current Policy: handbook_2026.pdf (vEffective August 1, 2025) Page 1
  - Older Conflicting Policy: handbook_2025.pdf (vEffective August 1, 2024) Page 1

✅ TEST_CASE_2 PASSED
----------------------------------------------------------------------
RUNNING TEST_CASE_3: Grounded rejection of unsupported query
Query: "What is the cafeteria menu?"
----------------------------------------------------------------------
[Application Output]:
I could not find this information in the provided documents.

[Conflict Detected]: False
[Citations Provided]:

✅ TEST_CASE_3 PASSED
======================================================================
ALL TESTS PASSED SUCCESSFULLY! (3/3)
======================================================================
```

---

## 🛡️ Acceptance Criteria Checklist

| Requirement | Status | Implementation |
|---|---|---|
| Upload one or multiple 100+ page PDFs | ✅ Complete | Page-aware chunker with overlap and metadata retention |
| Automatic version detection | ✅ Complete | Multi-tier regex: Effective Date > Pub Date > Version > AY > Filename |
| Dense semantic retrieval | ✅ Complete | MiniLM-L6-v2 + FAISS cosine similarity (8–12 candidates) |
| Grounded factual answers | ✅ Complete | Strict prompt constraint & zero hallucination policy |
| Detect contradictory policies | ✅ Complete | Deterministic entity & numerical divergence engine |
| Prioritize newer policy | ✅ Complete | Enforces highest version score as primary answer |
| Disclose older conflicting policy | ✅ Complete | Explicit `"⚠️ This contradicts an older version."` notice |
| Dual page citations | ✅ Complete | Displays current edition & historical edition page numbers |
| Reject unsupported questions | ✅ Complete | `"I could not find this information in the provided documents."` |
| Privacy & zero paid APIs | ✅ Complete | Runs locally on CPU with Ollama / deterministic synthesis |
| Automated verification test | ✅ Complete | `demo_test.py` passes all 3 scenarios (3/3) |
| Live Web Application Preview | ✅ Complete | Deployed on Google Cloud Run |

---

## 📄 License
This project is licensed under the Apache-2.0 License.
