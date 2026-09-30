
# NextGen University AI Assistant

Fast local university document assistant.

## Architecture

PDF
↓
PyMuPDF
↓
Text chunks
↓
MiniLM semantic embeddings
↓
FAISS vector search
↓
Top relevant chunks
↓
Ollama Llama 3.2 3B
↓
Grounded answer + source pages

## Features

- Harvard University mode
- Upload any university PDF
- Semantic retrieval
- FAISS vector database
- Ollama local LLM
- Page citations
- Multiple PDF support
- No cloud LLM
- Fast short answers

## Harvard

Put Harvard PDF files into:

harvard_docs/

Then run:

streamlit run app.py

## Other University

Open:

Upload University PDF

Upload one or more PDFs.

Enter the university name.

Click:

Process University Documents

Then ask questions.

## Ollama

Recommended model:

llama3.2:3b

Check:

ollama list

If not installed:

ollama pull llama3.2:3b
