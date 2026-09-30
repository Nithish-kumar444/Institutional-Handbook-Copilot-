# 📘 Institutional Handbook Copilot

## A Local Flask-Based Citation-Aware RAG System for Institutional Document Intelligence

Institutional Handbook Copilot is a **locally hosted AI-powered document
assistant** designed for large institutional PDFs such as university
handbooks, academic regulations, compliance policies, and grant
guidelines.

The system allows users to ask natural-language questions and receive
**grounded answers with page-level citations**. It uses
Retrieval-Augmented Generation (RAG), semantic search, FAISS vector
retrieval, and a locally running Ollama LLM.

The application is designed to run on a local machine through **Flask**,
with the web interface accessible through:

``` text
http://127.0.0.1:5000
```

------------------------------------------------------------------------

## 🎯 Problem Statement

Large institutional PDFs can contain hundreds of pages of information.
Students, faculty, and employees may spend significant time manually
searching these documents for specific rules, policies, requirements,
and procedures.

The problem becomes more difficult when multiple versions of a handbook
contain different policies.

For example:

``` text
2025 Handbook
Minimum attendance requirement: 75%

2026 Handbook
Minimum attendance requirement: 80%
```

A conventional document chatbot may retrieve both statements without
understanding which policy is current.

Institutional Handbook Copilot is designed to address this problem by
combining:

-   Semantic document retrieval
-   Page-aware PDF processing
-   Version detection
-   Policy conflict detection
-   Newer-version prioritization
-   Grounded local LLM generation
-   Page-level and document-level citations
-   Hallucination-resistant fallback behavior

------------------------------------------------------------------------

# 🚀 Key Features

-   Natural-language question answering
-   Local Flask web application
-   Upload and process institutional PDFs
-   Semantic search using `all-MiniLM-L6-v2`
-   FAISS vector retrieval
-   Page-aware PDF processing
-   Multiple PDF / handbook version support
-   Automatic version detection
-   Policy conflict detection
-   Newer-version priority
-   Explicit conflict warnings
-   Page-level citations
-   Source-document citations
-   Grounded answers
-   Unsupported-question fallback
-   Local LLM inference with Ollama
-   No paid cloud LLM API required
-   Localhost deployment
-   Evidence-based document question answering

------------------------------------------------------------------------

# 🏗️ System Architecture

``` text
                  Institutional PDF(s)
                         │
                         ▼
                ┌──────────────────┐
                │ PyMuPDF Extraction│
                └─────────┬────────┘
                          │
                          ▼
                ┌──────────────────┐
                │ Page-Aware        │
                │ Chunking          │
                └─────────┬────────┘
                          │
                          ▼
                ┌──────────────────┐
                │ MiniLM Embeddings │
                │ all-MiniLM-L6-v2  │
                └─────────┬────────┘
                          │
                          ▼
                ┌──────────────────┐
                │ FAISS Vector      │
                │ Index             │
                └─────────┬────────┘
                          │
                          ▼
                    User Question
                          │
                          ▼
                ┌──────────────────┐
                │ Query Embedding   │
                └─────────┬────────┘
                          │
                          ▼
                ┌──────────────────┐
                │ Semantic Retrieval│
                └─────────┬────────┘
                          │
                          ▼
                Relevant PDF Chunks
                          │
                    ┌─────┴─────┐
                    ▼           ▼
             Version Detection  Conflict Detection
                    │           │
                    └─────┬─────┘
                          ▼
                Current Policy Selection
                          │
                          ▼
                ┌──────────────────┐
                │ Local Ollama LLM │
                └─────────┬────────┘
                          │
                          ▼
              Answer + Warning + Citations
                          │
                          ▼
                ┌──────────────────┐
                │ Flask Web Interface│
                │ localhost:5000     │
                └──────────────────┘
```

------------------------------------------------------------------------

# 🧠 RAG Pipeline

The system follows these steps:

1.  Upload one or more institutional PDFs.
2.  Extract text page by page using PyMuPDF.
3.  Clean and normalize extracted text.
4.  Split the text into overlapping chunks.
5.  Store page, document, version, and chunk metadata.
6.  Generate embeddings using `all-MiniLM-L6-v2`.
7.  Store embeddings in a FAISS index.
8.  Convert the user's question into an embedding.
9.  Retrieve the most relevant document chunks.
10. Compare relevant policies across document versions.
11. Detect possible policy conflicts.
12. Select the latest applicable policy when version information is
    available.
13. Send only the retrieved evidence to the local Ollama model.
14. Generate a concise grounded answer.
15. Display page and source-document citations.
16. Show a conflict warning when contradictory versions are detected.
17. Return a fallback response when the required information is not
    found.

------------------------------------------------------------------------

# 🔍 Version and Conflict Detection

A major feature of the project is the ability to work with evolving
institutional documents.

The system can attempt to identify document versions using:

-   Effective date
-   Publication date
-   Revision date
-   Explicit version number
-   Academic year
-   Filename
-   Document title

### Example

``` text
Handbook_2025.pdf
Page 38
Minimum attendance requirement: 75%

Handbook_2026.pdf
Page 42
Minimum attendance requirement: 80%
```

If the user asks:

``` text
What is the minimum attendance requirement?
```

The system can produce:

``` text
The current minimum attendance requirement is 80%.

⚠️ This contradicts an older version.

Current policy:
Handbook_2026.pdf — Page 42

Older conflicting policy:
Handbook_2025.pdf — Page 38
```

The older policy is not silently discarded. It remains visible so that
the user can understand the policy change.

------------------------------------------------------------------------

# 📚 Example Questions

The application can answer questions such as:

### Academic

1.  What is the grading system used by the institution?
2.  What are the requirements for obtaining a degree?
3.  What happens if a student misses a final examination?
4.  What are the rules for course registration?
5.  What are the academic probation requirements?

### Student Life

6.  What are the housing rules?
7.  What are the rules regarding student organizations?
8.  What student conduct policies are mentioned in the handbook?
9.  What health-related resources are available to students?
10. What are the rules for extracurricular activities?

### Administrative

11. What are the financial obligations of students?
12. What are the rules for taking a leave of absence?
13. How can a student return after a leave of absence?
14. What are the examination policies?
15. What are the academic complaint procedures?

### Advanced RAG Tests

16. What is the minimum attendance requirement?
17. Has the attendance requirement changed between handbook versions?
18. How many backlogs are allowed for promotion?
19. What changed between the previous and current handbook?
20. What is the cafeteria menu?

The final question can be used as an **unsupported-question test**. If
the information does not exist in the uploaded documents, the system
should not invent an answer.

Expected behavior:

``` text
I could not find this information in the provided documents.
```

------------------------------------------------------------------------

# 🛡️ Grounding and Hallucination Control

The system is designed to reduce hallucinations by:

-   Restricting the LLM to retrieved document evidence
-   Using semantic retrieval before generation
-   Providing source citations
-   Providing page numbers
-   Detecting document-version conflicts
-   Prioritizing the newer applicable policy
-   Returning a fallback when information is unavailable

The model should not invent:

-   Rules
-   Dates
-   Requirements
-   Percentages
-   Policies
-   Procedures
-   Institutional decisions

For important institutional decisions, users should still verify the
answer against the cited official document.

------------------------------------------------------------------------

# 🖥️ Web Interface

The application provides a local web interface with sections such as:

``` text
University Mode

[ General Chat ]

[ 1. Harvard Chat ]

[ 2. Upload University PDF ]
```

The knowledge-base section allows users to select and process PDFs.

Example:

``` text
Knowledge Base

Choose Files: Harvard.pdf

[ Harvard ]

[ Process University Documents ]
```

The retrieval section displays information about the semantic search
system:

``` text
Retrieval

Semantic Search

MiniLM embeddings + FAISS
Top results: 4
```

The interface can also display:

``` text
Knowledge Chunks: 526
Documents:       1
Pages:           114
```

These values depend on the documents processed by the application.

------------------------------------------------------------------------

# 🛠️ Technology Stack

  Component              Technology
  ---------------------- -----------------------------
  Web Framework          Flask
  Frontend               HTML, CSS, JavaScript
  Programming Language   Python
  PDF Processing         PyMuPDF
  Embeddings             Sentence Transformers
  Embedding Model        `all-MiniLM-L6-v2`
  Vector Search          FAISS CPU
  LLM Runtime            Ollama
  Local LLM              Llama / Qwen 3B-class model
  Numerical Processing   NumPy
  HTTP Communication     Requests
  Deployment             Localhost
  Host                   `127.0.0.1`
  Port                   `5000`

------------------------------------------------------------------------

# 📁 Project Structure

A typical project structure is:

``` text
institutional_handbook_copilot/
│
├── CHATBOT-LLama-2-main
│   └── CHATBOT-LLama-2-main
│       └── harvard_docs
│       │   └── harvard_pdf
│       └── Static
│           └── index.html
│       └── templates
│       └── License
│       └── app.py
│       └── chat_history.json
│   └── app.py
│
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── data/
│   └── uploaded PDFs
│
├── cache/
│   ├── embeddings
│   └── indexes
│
└── sample_docs/
    ├── Handbook_2025.pdf
    └── Handbook_2026.pdf
```

The exact structure may differ depending on the implementation.

------------------------------------------------------------------------

# 💻 System Requirements

Recommended:

-   Python 3.10 or newer
-   8 GB RAM or more
-   Windows, Linux, or macOS
-   Ollama installed locally
-   Internet connection for the initial model and embedding-model
    download
-   Sufficient disk space for PDF files, models, and indexes

A dedicated GPU is not required for the basic CPU-based implementation.

------------------------------------------------------------------------

# 📦 Installation

## 1. Clone or Download the Project

``` bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd institutional_handbook_copilot
```

Or download the project ZIP and extract it.

------------------------------------------------------------------------

## 2. Create a Virtual Environment

### Windows

``` bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

``` bash
python3 -m venv venv
source venv/bin/activate
```

------------------------------------------------------------------------

## 3. Install Python Dependencies

``` bash
python -m pip install -r requirements.txt
```

Example `requirements.txt`:

``` text
Flask
PyMuPDF
sentence-transformers
faiss-cpu
numpy
requests
```

------------------------------------------------------------------------

# 🤖 Ollama Setup

Install Ollama on the local computer.

After installation, download a compatible local model.

For example:

``` bash
ollama pull llama3.2:3b
```

Or, if the project configuration uses Qwen:

``` bash
ollama pull qwen2.5:3b
```

Check installed models:

``` bash
ollama list
```

The exact model name must match the model configured in `app.py`.

------------------------------------------------------------------------

# ▶️ Running the Application

Start Ollama first.

Then run the Flask application:

``` bash
python app.py
```

The terminal should show a local Flask address similar to:

``` text
http://127.0.0.1:5000
```

Open the address in a web browser.

``` text
http://127.0.0.1:5000
```

------------------------------------------------------------------------

# 📄 Using the Application

## Step 1 --- Upload a PDF

Select an institutional handbook or policy document.

Example:

``` text
Harvard.pdf
```

------------------------------------------------------------------------

## Step 2 --- Process the Document

Click:

``` text
Process University Documents
```

The system extracts the document content and creates the semantic
knowledge base.

------------------------------------------------------------------------

## Step 3 --- Ask a Question

Example:

``` text
What is the grading system used by the institution?
```

------------------------------------------------------------------------

## Step 4 --- Retrieval

The question is converted into an embedding.

FAISS then searches the indexed document chunks and retrieves the most
relevant evidence.

------------------------------------------------------------------------

## Step 5 --- Local Generation

The retrieved evidence is passed to the local Ollama model.

The model generates a grounded response.

------------------------------------------------------------------------

## Step 6 --- Review Sources

The answer should contain information such as:

``` text
Answer:
...

Source:
Harvard.pdf

Page:
42
```

If multiple versions conflict, the system should display the conflict
information.

------------------------------------------------------------------------

# 🔬 Example RAG Flow

``` text
Question
   │
   ▼
"What is the grading system?"
   │
   ▼
MiniLM Query Embedding
   │
   ▼
FAISS Similarity Search
   │
   ▼
Top Relevant Chunks
   │
   ▼
Version / Conflict Analysis
   │
   ▼
Relevant Evidence
   │
   ▼
Ollama Local LLM
   │
   ▼
Grounded Answer
   │
   ▼
Page-Level Citation
```

------------------------------------------------------------------------

# 🧪 Demonstration Tests

## Test 1 --- Normal Question

### Question

``` text
What is the grading system used by the institution?
```

### Expected behavior

The system should retrieve the relevant grading section and answer using
the retrieved document evidence.

------------------------------------------------------------------------

## Test 2 --- Policy Question

### Question

``` text
What is the minimum attendance requirement?
```

The answer should include the relevant policy and page citation.

If multiple handbook versions contain different values, the system
should show the conflict.

------------------------------------------------------------------------

## Test 3 --- Version Conflict

### Documents

``` text
Handbook_2025.pdf
Handbook_2026.pdf
```

### Question

``` text
What is the minimum attendance requirement?
```

### Expected behavior

``` text
Current policy:
Handbook_2026.pdf — Page XX

⚠️ Older conflicting policy:
Handbook_2025.pdf — Page XX
```

The exact answer depends on the contents of the uploaded documents.

------------------------------------------------------------------------

## Test 4 --- Unsupported Question

### Question

``` text
What is the cafeteria menu?
```

If the information is not present:

``` text
I could not find this information in the provided documents.
```

The system should not generate an unsupported answer.

------------------------------------------------------------------------

# ⚡ Performance Optimization

The application is designed for a normal student laptop using local
resources.

Optimization techniques include:

-   Cached embedding model
-   Cached FAISS index
-   CPU-compatible FAISS
-   Local Ollama inference
-   Limited retrieval context
-   Short generated answers
-   Page-aware chunking
-   No paid cloud inference

The first execution can take longer because the embedding model and
local LLM need to be downloaded and initialized.

------------------------------------------------------------------------

# 🔐 Privacy and Cost

A major objective of the project is **local AI processing**.

The main workflow is:

``` text
PDF
 ↓
Local Flask Application
 ↓
Local Embeddings
 ↓
Local FAISS
 ↓
Local Ollama
 ↓
Local Answer
```

No paid cloud LLM API is required for the core system.

This makes the architecture suitable for institutional documents where
keeping document content on the local machine is desirable.

------------------------------------------------------------------------

# ⚠️ Limitations

System performance depends on:

-   PDF text quality
-   Semantic retrieval quality
-   Chunking strategy
-   Correct version detection
-   Similarity between conflicting sections
-   Clarity of policy language
-   Local hardware performance

Scanned image-only PDFs may require OCR.

Documents without clear version information may require manual version
labeling.

The system should not be treated as a replacement for official
institutional policy documents.

------------------------------------------------------------------------

# 🔮 Future Improvements

Possible future improvements include:

-   OCR for scanned PDFs
-   Hybrid BM25 + semantic search
-   Cross-encoder reranking
-   Sentence-level citations
-   Table extraction
-   Automatic policy-topic classification
-   Policy-change timeline
-   Side-by-side version comparison
-   Document change tracking
-   Multi-language support
-   Authentication
-   Role-based access control
-   Persistent database storage
-   Docker deployment
-   Optional cloud deployment

------------------------------------------------------------------------

# 🎯 Project Objective

Institutional Handbook Copilot transforms long institutional documents
into a searchable, explainable, and version-aware knowledge system.

Instead of only answering:

> **What does the handbook say?**

the system is designed to answer:

> **What is the current policy, what did the older version say, and did
> the policy change?**

------------------------------------------------------------------------

# 📊 Evaluation Metrics

The RAG system can be evaluated using:

### Retrieval Metrics

-   Recall@K
-   Precision@K
-   Mean Reciprocal Rank (MRR)

### Answer Metrics

-   Answer relevance
-   Context relevance
-   Faithfulness
-   Citation accuracy

### System Metrics

-   Retrieval latency
-   Response generation time
-   End-to-end response time

------------------------------------------------------------------------

# 🧩 Core Project Components

``` text
RAG
 +
Semantic Search
 +
FAISS Retrieval
 +
Version Awareness
 +
Conflict Detection
 +
Page-Level Citations
 +
Local LLM Inference
 +
Flask Web Application
```

Together, these components create a transparent document
question-answering system for large and evolving institutional
documents.

------------------------------------------------------------------------

# 🏆 Project Highlights

### Local AI

Runs the core AI pipeline locally using Ollama.

### RAG

Answers are generated using retrieved institutional document evidence.

### Semantic Search

Uses MiniLM embeddings instead of relying only on keyword matching.

### Version Awareness

Handles multiple versions of institutional documents.

### Conflict Detection

Identifies contradictory policy statements across versions.

### Citation-Aware Answers

Provides document and page references for verification.

### Hallucination Resistance

Returns a not-found response when the required information is not
supported by the documents.

### Flask Deployment

The complete application is accessible through a local Flask server.

``` text
http://127.0.0.1:5000
```

------------------------------------------------------------------------

# 👨‍💻 Project Summary

**Project:** Institutional Handbook Copilot

**Category:** Artificial Intelligence / Machine Learning / Generative AI
/ NLP

**Core Technology:** Retrieval-Augmented Generation (RAG)

**Web Framework:** Flask

**Vector Search:** FAISS

**Embedding Model:** `all-MiniLM-L6-v2`

**Local LLM:** Ollama

**PDF Processing:** PyMuPDF

**Deployment:** Localhost

**Primary Goal:** Intelligent, citation-aware question answering over
large institutional documents.

------------------------------------------------------------------------

# 📌 Final Run Command

``` bash
python app.py
```

Then open:

``` text
http://127.0.0.1:5000
```

------------------------------------------------------------------------

## 📜 License

Add the license appropriate for your project and repository.

For example:

``` text
MIT License
```

if you choose to release the project under the MIT License.
