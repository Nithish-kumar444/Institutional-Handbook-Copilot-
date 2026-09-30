"""
Institutional Handbook Copilot - Core Engine
Version-Aware RAG with Conflict Detection
"""

import os
import re
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HandbookCopilot")

# Optional imports handled gracefully
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

import requests


@dataclass
class VersionInfo:
    version_str: str
    version_score: float
    effective_date: Optional[str] = None
    publication_date: Optional[str] = None
    is_confident: bool = True
    detection_source: str = "inferred"


@dataclass
class DocumentChunk:
    chunk_id: int
    text: str
    page: int
    document: str
    version: str
    version_score: float
    effective_date: Optional[str] = None


@dataclass
class ConflictReport:
    has_conflict: bool
    summary: str
    newer_citation: Dict[str, Any]
    older_citation: Dict[str, Any]
    details: str = ""


# Global singleton embedding model cache
_EMBEDDING_MODEL = None


def get_embedding_model():
    """Cache the embedding model so it's loaded only once."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None and HAS_SENTENCE_TRANSFORMERS:
        try:
            logger.info("Loading sentence-transformers/all-MiniLM-L6-v2...")
            _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer: {e}")
            _EMBEDDING_MODEL = None
    return _EMBEDDING_MODEL


# =====================================================================
# 1. Version Detection Engine
# =====================================================================

def detect_version_from_text_and_meta(filename: str, first_pages_text: str = "", pdf_metadata: Optional[Dict] = None) -> VersionInfo:
    """
    Detects version/year based on priority:
    1. Effective date
    2. Publication/revision date
    3. Explicit version number
    4. Academic year
    5. Year inferred from filename/title
    """
    effective_date_str = None
    publication_date_str = None
    combined_header = f"{filename}\n{first_pages_text[:3000]}"

    # 1. Effective Date Check (e.g., "Effective Date: August 1, 2026", "Effective: 2026-09-01")
    eff_match = re.search(
        r'effective\s*(?:date)?[:\s]+(?:from\s+)?([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|\d{4})',
        combined_header,
        re.IGNORECASE
    )
    if eff_match:
        effective_date_str = eff_match.group(1).strip()
        year_match = re.search(r'\b(20\d{2})\b', effective_date_str)
        if year_match:
            year_val = float(year_match.group(1))
            return VersionInfo(
                version_str=f"Effective {effective_date_str}",
                version_score=year_val + 0.5,
                effective_date=effective_date_str,
                is_confident=True,
                detection_source="effective_date"
            )

    # 2. Publication / Revision Date (e.g. "Revised: July 2026", "Published: 2026")
    rev_match = re.search(
        r'(?:revised|revision|published|approved)[:\s]+(?:on\s+)?([A-Za-z]+\s+\d{1,2},?\s+\d{4}|[A-Za-z]+\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{4})',
        combined_header,
        re.IGNORECASE
    )
    if rev_match:
        publication_date_str = rev_match.group(1).strip()
        year_match = re.search(r'\b(20\d{2})\b', publication_date_str)
        if year_match:
            year_val = float(year_match.group(1))
            return VersionInfo(
                version_str=f"Rev {publication_date_str}",
                version_score=year_val + 0.3,
                publication_date=publication_date_str,
                is_confident=True,
                detection_source="publication_date"
            )

    # 3. Explicit Version Number (e.g., "Version 2.1", "v3.0", "Handbook v2")
    ver_match = re.search(r'\b(?:version|ver|v)\.?\s*(\d+(?:\.\d+)+|\d+)\b', combined_header, re.IGNORECASE)
    if ver_match:
        ver_str = ver_match.group(1)
        try:
            parts = [float(p) for p in ver_str.split('.')]
            score = 2000.0 + parts[0] + (parts[1] / 10.0 if len(parts) > 1 else 0)
            return VersionInfo(
                version_str=f"v{ver_str}",
                version_score=score,
                is_confident=True,
                detection_source="explicit_version"
            )
        except Exception:
            pass

    # 4. Academic Year (e.g. "2025-2026", "2025-26", "AY 2026/2027")
    acad_match = re.search(r'\b(20\d{2})[-/](20\d{2}|\d{2})\b', combined_header)
    if acad_match:
        start_yr = int(acad_match.group(1))
        second_part = acad_match.group(2)
        end_yr = int(second_part) if len(second_part) == 4 else int(f"20{second_part}")
        acad_str = f"{start_yr}-{str(end_yr)[-2:]}"
        return VersionInfo(
            version_str=acad_str,
            version_score=float(end_yr),
            is_confident=True,
            detection_source="academic_year"
        )

    # 5. Year in filename or title (e.g., "Handbook_2026.pdf", "2025_Academic_Regs.pdf")
    file_year_match = re.search(r'\b(20\d{2})\b', filename)
    if file_year_match:
        yr = file_year_match.group(1)
        return VersionInfo(
            version_str=yr,
            version_score=float(yr),
            is_confident=True,
            detection_source="filename_year"
        )

    # Search for any standalone 4-digit year in header text
    text_year_match = re.search(r'\b(20[2-9]\d)\b', first_pages_text[:2000])
    if text_year_match:
        yr = text_year_match.group(1)
        return VersionInfo(
            version_str=yr,
            version_score=float(yr),
            is_confident=True,
            detection_source="header_year"
        )

    # Check PDF metadata if available
    if pdf_metadata:
        for key in ["creationDate", "modDate"]:
            val = str(pdf_metadata.get(key, ""))
            m = re.search(r'(20\d{2})', val)
            if m:
                yr = m.group(1)
                return VersionInfo(
                    version_str=yr,
                    version_score=float(yr),
                    is_confident=False,
                    detection_source="pdf_metadata"
                )

    # Unconfident fallback
    return VersionInfo(
        version_str="Version not confidently detected",
        version_score=1.0,
        is_confident=False,
        detection_source="none"
    )


# =====================================================================
# 2. PDF Processing & Page-Aware Chunking
# =====================================================================

def extract_and_chunk_pdf(
    file_path: str,
    override_version: Optional[str] = None,
    chunk_size: int = 400,
    chunk_overlap: int = 80
) -> Tuple[List[DocumentChunk], VersionInfo, Dict[str, Any]]:
    """
    Extracts text page by page from a PDF and produces page-aware chunks
    retaining document metadata and detected version.
    """
    filename = os.path.basename(file_path)
    chunks: List[DocumentChunk] = []
    first_pages_text = ""
    pdf_meta = {}
    total_pages = 0
    total_text_length = 0

    if not HAS_PYMUPDF:
        raise RuntimeError("PyMuPDF (fitz) is not installed. Run: pip install PyMuPDF")

    doc = fitz.open(file_path)
    total_pages = len(doc)
    pdf_meta = doc.metadata or {}

    # Extract text per page
    pages_data = []
    for page_idx in range(total_pages):
        page = doc[page_idx]
        text = page.get_text("text")
        cleaned_text = re.sub(r'[ \t]+', ' ', text)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()
        pages_data.append((page_idx + 1, cleaned_text))
        total_text_length += len(cleaned_text)
        if page_idx < 3:
            first_pages_text += f"\n{cleaned_text}"

    doc.close()

    # Detect version or apply user override
    if override_version and override_version.strip():
        v_str = override_version.strip()
        y_match = re.search(r'\b(20\d{2})\b', v_str)
        score = float(y_match.group(1)) if y_match else 2026.0
        version_info = VersionInfo(
            version_str=v_str,
            version_score=score,
            is_confident=True,
            detection_source="manual_override"
        )
    else:
        version_info = detect_version_from_text_and_meta(filename, first_pages_text, pdf_meta)

    # Check for empty or scanned PDF
    stats = {
        "filename": filename,
        "page_count": total_pages,
        "detected_version": version_info.version_str,
        "is_confident": version_info.is_confident,
        "total_characters": total_text_length,
        "is_scanned_or_empty": total_text_length < 20
    }

    if stats["is_scanned_or_empty"]:
        logger.warning(f"File {filename} has virtually no extractable text. May be a scanned image.")
        return [], version_info, stats

    # Create page-aware chunks
    chunk_counter = 0
    for page_num, page_text in pages_data:
        if not page_text:
            continue

        # Split into sentences or lines
        sentences = re.split(r'(?<=[.!?\n])\s+', page_text)
        current_chunk = ""

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            if len(current_chunk) + len(sent) > chunk_size and len(current_chunk) > 100:
                chunk_counter += 1
                chunks.append(DocumentChunk(
                    chunk_id=chunk_counter,
                    text=current_chunk.strip(),
                    page=page_num,
                    document=filename,
                    version=version_info.version_str,
                    version_score=version_info.version_score,
                    effective_date=version_info.effective_date
                ))
                # Retain overlap from end of current chunk
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else ""
                current_chunk = f"{overlap_text} {sent}"
            else:
                current_chunk = f"{current_chunk} {sent}".strip()

        if current_chunk:
            chunk_counter += 1
            chunks.append(DocumentChunk(
                chunk_id=chunk_counter,
                text=current_chunk.strip(),
                page=page_num,
                document=filename,
                version=version_info.version_str,
                version_score=version_info.version_score,
                effective_date=version_info.effective_date
            ))

    stats["chunk_count"] = len(chunks)
    return chunks, version_info, stats


# =====================================================================
# 3. Vector Store & Semantic Retrieval (FAISS + MiniLM)
# =====================================================================

class HandbookVectorStore:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.index = None
        self.embedding_model = get_embedding_model()
        self.embeddings: Optional[np.ndarray] = None

    def add_chunks(self, new_chunks: List[DocumentChunk]):
        if not new_chunks:
            return
        self.chunks.extend(new_chunks)
        self._rebuild_index()

    def clear(self):
        self.chunks = []
        self.index = None
        self.embeddings = None

    def _rebuild_index(self):
        if not self.chunks:
            return

        texts = [c.text for c in self.chunks]

        if self.embedding_model is not None:
            # Dense embeddings with all-MiniLM-L6-v2
            raw_embs = self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            # L2 normalize for cosine similarity
            norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            normalized_embs = (raw_embs / norms).astype(np.float32)
            self.embeddings = normalized_embs

            if HAS_FAISS:
                d = normalized_embs.shape[1]
                self.index = faiss.IndexFlatIP(d)
                self.index.add(normalized_embs)
            else:
                self.index = "numpy_fallback"
        else:
            # Fallback simple TF-IDF / term overlap vectorizer
            self._build_fallback_index(texts)

    def _build_fallback_index(self, texts: List[str]):
        # Lightweight n-gram / term frequency index
        vocab = {}
        for t in texts:
            words = re.findall(r'\w+', t.lower())
            for w in words:
                if w not in vocab:
                    vocab[w] = len(vocab)
        self.vocab = vocab
        mat = np.zeros((len(texts), len(vocab)), dtype=np.float32)
        for i, t in enumerate(texts):
            words = re.findall(r'\w+', t.lower())
            for w in words:
                if w in vocab:
                    mat[i, vocab[w]] += 1.0
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.embeddings = mat / norms
        self.index = "numpy_fallback"

    def search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        if not self.chunks:
            return []

        top_k = min(top_k, len(self.chunks))

        if self.embedding_model is not None and self.index is not None:
            q_emb = self.embedding_model.encode([query], convert_to_numpy=True, show_progress_bar=False)
            norm = np.linalg.norm(q_emb)
            if norm > 0:
                q_emb = (q_emb / norm).astype(np.float32)

            if HAS_FAISS and isinstance(self.index, faiss.Index):
                scores, indices = self.index.search(q_emb, top_k)
                results = []
                for idx, score in zip(indices[0], scores[0]):
                    if idx >= 0 and idx < len(self.chunks):
                        results.append((self.chunks[idx], float(score)))
                return results
            else:
                # Direct numpy cosine similarity
                scores = np.dot(self.embeddings, q_emb.T).flatten()
                top_indices = np.argsort(scores)[::-1][:top_k]
                return [(self.chunks[i], float(scores[i])) for i in top_indices]
        elif hasattr(self, 'vocab'):
            # Fallback keyword vector search
            q_words = re.findall(r'\w+', query.lower())
            q_vec = np.zeros((1, len(self.vocab)), dtype=np.float32)
            for w in q_words:
                if w in self.vocab:
                    q_vec[0, self.vocab[w]] += 1.0
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec = q_vec / norm
            scores = np.dot(self.embeddings, q_vec.T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [(self.chunks[i], float(scores[i])) for i in top_indices]

        return [(self.chunks[i], 1.0) for i in range(top_k)]


# =====================================================================
# 4. Practical Version Conflict Detection Pipeline
# =====================================================================

def extract_policy_facts(text: str) -> List[Dict[str, Any]]:
    """
    Extracts key quantitative metrics and rule clauses from text for comparison:
    - percentages (e.g. 75%, 80%)
    - limits / counts (e.g. 4 backlogs, 2 backlogs, 30 days)
    - deadlines / dates (e.g. June 30, July 15)
    - allowed / prohibited / eligibility statements
    """
    facts = []

    # 1. Percentages with context (e.g., "attendance requirement: 75%", "minimum 80%")
    pct_matches = re.finditer(r'([^.!?\n]*?(\b\d{1,3}\s*%\b)[^.!?\n]*)', text, re.IGNORECASE)
    for m in pct_matches:
        clause = m.group(1).strip()
        val = m.group(2).strip()
        # Topic key
        topic = "percentage_requirement"
        if "attendance" in clause.lower():
            topic = "attendance"
        elif "grade" in clause.lower() or "marks" in clause.lower() or "passing" in clause.lower():
            topic = "passing_grade"
        facts.append({
            "type": "percentage",
            "topic": topic,
            "value": val,
            "clause": clause
        })

    # 2. Number + noun quantities (e.g. "4 backlogs", "2 backlogs", "30 days", "15 days")
    qty_matches = re.finditer(r'([^.!?\n]*?\b(\d+)\s+(backlogs?|days?|credits?|semesters?|attempts?|weeks?|months?)\b[^.!?\n]*)', text, re.IGNORECASE)
    for m in qty_matches:
        clause = m.group(1).strip()
        num = m.group(2).strip()
        noun = m.group(3).lower()
        topic = f"{noun}_limit"
        if "promotion" in clause.lower() or "promote" in clause.lower():
            topic = f"{noun}_for_promotion"
        facts.append({
            "type": "quantity",
            "topic": topic,
            "value": f"{num} {noun}",
            "clause": clause
        })

    # 3. Allowed vs Prohibited / Eligibility
    rule_matches = re.finditer(r'([^.!?\n]*?\b(allowed|permitted|eligible|prohibited|forbidden|not eligible|disqualified)\b[^.!?\n]*)', text, re.IGNORECASE)
    for m in rule_matches:
        clause = m.group(1).strip()
        keyword = m.group(2).lower()
        polarity = "prohibited" if keyword in ["prohibited", "forbidden", "not eligible", "disqualified"] else "allowed"
        # Determine topic
        words = [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', clause.lower()) if w not in ["allowed", "permitted", "eligible", "prohibited", "forbidden", "policy", "rules"]]
        topic = "_".join(words[:2]) if words else "rule_condition"
        facts.append({
            "type": "polarity",
            "topic": topic,
            "value": polarity,
            "clause": clause
        })

    # 4. Dates / Deadlines (e.g. "deadline: June 30", "due on July 15")
    date_matches = re.finditer(r'([^.!?\n]*?\b(?:deadline|due\s+date|last\s+date|cutoff)[:\s]+([A-Za-z]+\s+\d{1,2}|\d{1,2}/\d{1,2})\b[^.!?\n]*)', text, re.IGNORECASE)
    for m in date_matches:
        clause = m.group(1).strip()
        date_val = m.group(2).strip()
        facts.append({
            "type": "deadline",
            "topic": "deadline",
            "value": date_val,
            "clause": clause
        })

    return facts


def detect_conflicts_in_retrieved_chunks(
    retrieved_chunks: List[Tuple[DocumentChunk, float]],
    query: str
) -> ConflictReport:
    """
    Groups retrieved chunks across document versions and deterministically
    detects policy contradictions on shared topics.
    """
    if not retrieved_chunks:
        return ConflictReport(has_conflict=False, summary="", newer_citation={}, older_citation={})

    # Group chunks by version
    version_groups: Dict[str, List[DocumentChunk]] = {}
    for chunk, _ in retrieved_chunks:
        version_groups.setdefault(chunk.version, []).append(chunk)

    # Need at least two different versions to have a cross-version conflict
    if len(version_groups) < 2:
        return ConflictReport(has_conflict=False, summary="", newer_citation={}, older_citation={})

    # Sort versions by version_score descending (newest first)
    sorted_versions = sorted(
        version_groups.keys(),
        key=lambda v: max(c.version_score for c in version_groups[v]),
        reverse=True
    )

    newest_ver = sorted_versions[0]
    older_vers = sorted_versions[1:]

    newer_chunks = version_groups[newest_ver]
    newer_facts: List[Tuple[Dict[str, Any], DocumentChunk]] = []
    for c in newer_chunks:
        for f in extract_policy_facts(c.text):
            newer_facts.append((f, c))

    # Check for contradictions with older versions
    for old_ver in older_vers:
        older_chunks = version_groups[old_ver]
        for c_old in older_chunks:
            old_facts = extract_policy_facts(c_old.text)
            for f_old in old_facts:
                for f_new, c_new in newer_facts:
                    # Match on topic
                    if f_new["topic"] == f_old["topic"] and f_new["type"] == f_old["type"]:
                        # If values differ, it is a contradiction!
                        val_new = str(f_new["value"]).lower().strip()
                        val_old = str(f_old["value"]).lower().strip()
                        if val_new != val_old:
                            summary = (
                                f"Contradiction on '{f_new['topic'].replace('_', ' ')}': "
                                f"Newer version ({newest_ver}) specifies '{f_new['value']}', "
                                f"whereas older version ({old_ver}) specifies '{f_old['value']}'."
                            )
                            return ConflictReport(
                                has_conflict=True,
                                summary=summary,
                                newer_citation={
                                    "document": c_new.document,
                                    "version": c_new.version,
                                    "page": c_new.page,
                                    "statement": f_new["clause"],
                                    "value": f_new["value"]
                                },
                                older_citation={
                                    "document": c_old.document,
                                    "version": c_old.version,
                                    "page": c_old.page,
                                    "statement": f_old["clause"],
                                    "value": f_old["value"]
                                },
                                details=summary
                            )

    # General query-grounded numerical check if topic matching didn't trigger directly
    # e.g., query asks about "attendance" or "backlogs"
    query_lower = query.lower()
    for kw in ["attendance", "backlog", "credit", "grade", "day", "deadline", "fee", "rule"]:
        if kw in query_lower:
            # Collect statements containing keyword from newest and older
            new_clauses = [(c, s) for c in newer_chunks for s in c.text.split('\n') if kw in s.lower()]
            for old_ver in older_vers:
                old_clauses = [(c, s) for c in version_groups[old_ver] for s in c.text.split('\n') if kw in s.lower()]
                if new_clauses and old_clauses:
                    c_n, s_n = new_clauses[0]
                    c_o, s_o = old_clauses[0]
                    # Check if numbers or percentages differ
                    nums_n = re.findall(r'\b\d+(?:%|\s+[a-zA-Z]+)?\b', s_n)
                    nums_o = re.findall(r'\b\d+(?:%|\s+[a-zA-Z]+)?\b', s_o)
                    if nums_n and nums_o and nums_n != nums_o:
                        return ConflictReport(
                            has_conflict=True,
                            summary=f"Changed requirement on {kw} across versions ({old_ver} vs {newest_ver})",
                            newer_citation={
                                "document": c_n.document,
                                "version": c_n.version,
                                "page": c_n.page,
                                "statement": s_n.strip(),
                                "value": nums_n[0]
                            },
                            older_citation={
                                "document": c_o.document,
                                "version": c_o.version,
                                "page": c_o.page,
                                "statement": s_o.strip(),
                                "value": nums_o[0]
                            },
                            details=f"Older version ({old_ver}): {s_o.strip()} | Newer version ({newest_ver}): {s_n.strip()}"
                        )

    return ConflictReport(has_conflict=False, summary="", newer_citation={}, older_citation={})


# =====================================================================
# 5. Ollama LLM Client & Grounded Answer Synthesis
# =====================================================================

def query_ollama(
    prompt: str,
    model: str = "llama3.2:3b",
    host: str = "http://localhost:11434",
    timeout: int = 25
) -> Optional[str]:
    """Sends a request to local Ollama instance."""
    try:
        url = f"{host.rstrip('/')}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 250
            }
        }
        res = requests.post(url, json=payload, timeout=timeout)
        if res.status_code == 200:
            return res.json().get("response", "").strip()
        else:
            logger.warning(f"Ollama returned HTTP {res.status_code}: {res.text}")
            return None
    except requests.exceptions.ConnectionError:
        logger.info("Ollama is not running at localhost:11434.")
        return None
    except Exception as e:
        logger.warning(f"Ollama query failed: {e}")
        return None


def deterministic_grounded_answer(
    query: str,
    retrieved_chunks: List[Tuple[DocumentChunk, float]],
    conflict: ConflictReport
) -> str:
    """
    Deterministic synthesis engine that answers directly using retrieved chunks
    when Ollama is not running or as a deterministic baseline.
    Strictly adheres to:
    - Never hallucinate
    - Return 'I could not find this information in the provided documents.' for unsupported questions
    - Use newest version when conflict exists + explicit statement: 'This contradicts an older version.'
    """
    if not retrieved_chunks:
        return "I could not find this information in the provided documents."

    query_lower = query.lower()
    stop_words = {"what", "is", "the", "how", "many", "are", "for", "in", "a", "an", "to", "of", "and", "do", "does"}
    query_tokens = [w for w in re.findall(r'\b[a-zA-Z0-9%]+\b', query_lower) if w not in stop_words]

    # Check relevance of top chunk to prevent answering out-of-scope questions (e.g. cafeteria menu)
    relevance_found = False
    all_retrieved_text = " ".join(c.text.lower() for c, _ in retrieved_chunks)

    matches = [tok for tok in query_tokens if tok in all_retrieved_text]
    if len(matches) == 0 or (len(query_tokens) > 2 and len(matches) < 1):
        return "I could not find this information in the provided documents."

    # If conflict detected:
    if conflict.has_conflict:
        new_cite = conflict.newer_citation
        new_val = new_cite.get("value", "")
        new_stmt = new_cite.get("statement", "").strip()

        # Build natural direct answer
        if "attendance" in query_lower:
            ans_lead = f"The current minimum attendance requirement is {new_val}."
        elif "backlog" in query_lower:
            ans_lead = f"The current policy allows up to {new_val} for promotion."
        elif new_stmt:
            # Clean up leading dots/dashes
            clean_stmt = re.sub(r'^[-\s•*]+', '', new_stmt).strip()
            ans_lead = f"According to the current {new_cite.get('version')} policy: {clean_stmt}."
        else:
            ans_lead = f"The current policy specifies: {new_val}."

        answer_text = (
            f"{ans_lead}\n\n"
            f"⚠️ This contradicts an older version.\n\n"
            f"Current policy:\n"
            f"{new_cite.get('document')} ({new_cite.get('version')}) — Page {new_cite.get('page')}\n\n"
            f"Older policy:\n"
            f"{conflict.older_citation.get('document')} ({conflict.older_citation.get('version')}) — Page {conflict.older_citation.get('page')}"
        )
        return answer_text

    # No conflict: answer grounded using newest candidate chunk
    # Sort chunks by version score descending, then by similarity score
    sorted_chunks = sorted(retrieved_chunks, key=lambda x: (x[0].version_score, x[1]), reverse=True)
    best_chunk, _ = sorted_chunks[0]

    # Find the most matching sentence in best_chunk
    sentences = re.split(r'(?<=[.!?\n])\s+', best_chunk.text)
    matching_sentences = []
    for s in sentences:
        s_clean = s.strip()
        score = sum(1 for tok in query_tokens if tok in s_clean.lower())
        if score > 0:
            matching_sentences.append((score, s_clean))

    if matching_sentences:
        matching_sentences.sort(key=lambda x: x[0], reverse=True)
        primary_sentence = matching_sentences[0][1]
        primary_sentence = re.sub(r'^[-\s•*]+', '', primary_sentence).strip()
        return f"{primary_sentence}\n\nSource: {best_chunk.document}\nPage: {best_chunk.page}"

    return "I could not find this information in the provided documents."


# =====================================================================
# 6. Main Orchestrator: Copilot Pipeline
# =====================================================================

class HandbookCopilot:
    def __init__(self, ollama_model: str = "llama3.2:3b", ollama_host: str = "http://localhost:11434"):
        self.vector_store = HandbookVectorStore()
        self.documents_info: Dict[str, Dict[str, Any]] = {}
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host

    def load_pdf(self, file_path: str, override_version: Optional[str] = None) -> Dict[str, Any]:
        """Loads, parses, chunks and indexes a PDF."""
        chunks, version_info, stats = extract_and_chunk_pdf(file_path, override_version=override_version)
        if chunks:
            self.vector_store.add_chunks(chunks)
        self.documents_info[stats["filename"]] = {
            "stats": stats,
            "version_info": version_info,
            "chunk_count": len(chunks)
        }
        return stats

    def ask(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Runs the full version-aware RAG pipeline:
        1. Semantic retrieval of top candidate chunks (8-12)
        2. Version detection & conflict detection
        3. LLM generation with strict grounded prompt or fallback
        4. Structured citation and conflict response
        """
        if not self.vector_store.chunks:
            return {
                "answer": "No documents uploaded. Please upload handbook PDFs first.",
                "has_conflict": False,
                "conflict_report": None,
                "citations": [],
                "retrieved_chunks": []
            }

        # 1. Semantic retrieval
        retrieved = self.vector_store.search(query, top_k=top_k)

        # 2. Conflict detection
        conflict = detect_conflicts_in_retrieved_chunks(retrieved, query)

        # 3. Formulate strict prompt for LLM
        context_str = ""
        for i, (c, score) in enumerate(retrieved):
            context_str += f"[Chunk {i+1} | Doc: {c.document} | Version: {c.version} | Page: {c.page}]\n{c.text}\n\n"

        prompt = f"""You are the Institutional Handbook Copilot. Answer the question STRICTLY using only the provided excerpts below.
DO NOT assume or hallucinate outside knowledge.

RULES:
1. If the information is not explicitly mentioned in the context, respond ONLY with:
   "I could not find this information in the provided documents."
2. When a newer version contradicts an older version:
   - Answer using the newer version policy.
   - Explicitly include this sentence: "This contradicts an older version."
   - Cite both the current policy and older policy with document name and page number.
3. Keep the answer direct, accurate, and concise (under 200 words).

CONTEXT:
{context_str}

QUESTION:
{query}

ANSWER:"""

        # Try Ollama first
        llm_response = query_ollama(prompt, model=self.ollama_model, host=self.ollama_host)

        # If Ollama is unavailable or returned blank, use grounded deterministic synthesis
        if not llm_response:
            answer = deterministic_grounded_answer(query, retrieved, conflict)
        else:
            answer = llm_response

        # Format citations
        citations = []
        if conflict.has_conflict:
            citations.append({
                "type": "Current Policy",
                "document": conflict.newer_citation.get("document", ""),
                "version": conflict.newer_citation.get("version", ""),
                "page": conflict.newer_citation.get("page", "")
            })
            citations.append({
                "type": "Older Conflicting Policy",
                "document": conflict.older_citation.get("document", ""),
                "version": conflict.older_citation.get("version", ""),
                "page": conflict.older_citation.get("page", "")
            })
        else:
            # Normal citations from top relevant chunk(s)
            seen = set()
            for c, _ in retrieved[:3]:
                key = (c.document, c.page)
                if key not in seen and "could not find this information" not in answer.lower():
                    seen.add(key)
                    citations.append({
                        "type": "Source",
                        "document": c.document,
                        "version": c.version,
                        "page": c.page
                    })

        return {
            "query": query,
            "answer": answer,
            "has_conflict": conflict.has_conflict,
            "conflict_report": conflict,
            "citations": citations,
            "retrieved_chunks": [
                {
                    "document": c.document,
                    "version": c.version,
                    "page": c.page,
                    "score": round(score, 4),
                    "text": c.text[:180] + ("..." if len(c.text) > 180 else "")
                }
                for c, score in retrieved
            ]
        }
