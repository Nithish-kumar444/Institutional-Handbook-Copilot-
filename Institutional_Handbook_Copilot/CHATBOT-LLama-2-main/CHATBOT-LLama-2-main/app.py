import os
import re
import json
import hashlib
import difflib
import threading

import requests
import fitz  # PyMuPDF
import numpy as np
import faiss
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
BASE = os.path.dirname(os.path.abspath(__file__))
OLLAMA = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2:3b"
HARVARD_FOLDER = os.path.join(BASE, "harvard_docs")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 4
MIN_SCORE = 0.2  # chunks less relevant than this are ignored (raise it to be stricter)
GREETINGS = {"hi", "hii", "hello", "hey", "good morning", "good afternoon", "good evening", "thanks", "thank you", "bye"}
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
TIMEOUT = 300

# ---- version-conflict settings ----
KEY_SENTENCES = 4      # how many of the newest handbook's most relevant sentences get compared with older versions
SENT_MIN = 0.20        # a sentence must be at least this relevant to the question to be compared
ALIGN_MIN = 0.55       # an older sentence must be at least this similar to count as "the same clause"
SIMILAR_MIN = 0.72     # embedding similarity needed to trust the rule-based verdict without the LLM
OVERLAP_MIN = 0.35     # word overlap needed for the same purpose
MAX_CONFLICTS = 3      # conflicts shown per answer
MAX_JUDGE_CALLS = 3    # LLM confirmations per question (only used for ambiguous pairs)
JUDGE_TIMEOUT = 40

# ---- upload behaviour ----
RELATED_SIM = 0.80     # a clause counts as "already in the loaded handbook" at this similarity
RELATED_MIN = 0.30     # share of shared clauses needed to treat a new PDF as another version of the same handbook
HISTORY_TURNS = 6      # previous chat messages sent to the model
BANNER_END = "Answer (based on the newer policy):"

app = Flask(__name__)
model_lock = threading.Lock()
kb_lock = threading.Lock()
_embedder = None
KB = {"harvard": None, "harvard_sig": None, "custom": None, "custom_name": "Uploaded University"}


# ---------------- EMBEDDINGS ----------------
def embedder():
    global _embedder
    with model_lock:
        if _embedder is None:
            _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def embed(texts):
    vecs = embedder().encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")


# ---------------- PDF -> CHUNKS / SENTENCES ----------------
def extract_pages(data, source):
    pages = []
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            for n, page in enumerate(doc, 1):
                text = " ".join(page.get_text("text").split())
                if len(text) >= 20:
                    pages.append({"source": source, "page": n, "text": text})
    except Exception as e:
        print("PDF error:", source, e)
    return pages


def make_chunks(pages):
    chunks = []
    for p in pages:
        cur, length, fresh, n = [], 0, 0, 0

        def emit():
            nonlocal n
            chunks.append({"source": p["source"], "page": p["page"], "chunk": n, "text": " ".join(cur)})
            n += 1

        for w in p["text"].split():
            cur.append(w)
            length += len(w) + 1
            fresh += 1
            if length >= CHUNK_SIZE:
                emit()
                keep, l = [], 0
                for prev in reversed(cur):
                    if l + len(prev) + 1 > CHUNK_OVERLAP:
                        break
                    keep.insert(0, prev)
                    l += len(prev) + 1
                cur, length, fresh = keep, l, 0
        if cur and fresh:
            emit()
    return chunks


SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\"'\u201c\u2022])")


def make_sentences(pages):
    """Sentence-level units. These are what versions are compared on."""
    out = []
    for p in pages:
        for s in SENT_SPLIT.split(p["text"]):
            s = s.strip()
            if len(s) < 25:
                continue
            while len(s) > 500:  # run-on text (tables, lists): cut at a word boundary
                cut = s.rfind(" ", 0, 320)
                cut = cut if cut > 0 else 320
                out.append({"source": p["source"], "page": p["page"], "text": s[:cut].strip()})
                s = s[cut:].strip()
            if len(s) >= 25:
                out.append({"source": p["source"], "page": p["page"], "text": s})
    return out


def norm(text):
    return re.sub(r"[^a-z0-9%$ ]+", "", re.sub(r"\s+", " ", text.lower())).strip()


# ---------------- VERSIONED KNOWLEDGE BASE ----------------
# kb = {"versions": [version, ...], "seq": int}
# version = label, rank (higher = newer), seq (upload order), files, hashes, pages,
#           chunks + emb + index (for answering), sents + s_emb + s_index + s_norm (for comparing versions)
V_NUM = re.compile(r"(?<![a-z0-9])v(?:ersion)?\s*(\d+(?:\.\d+)?)(?![a-z0-9])", re.I)
YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
OLD_RE = re.compile(r"\b(original|old|older|previous|prior|legacy|superseded|archived?)\b", re.I)
NEW_RE = re.compile(r"\b(updated?|new|newer|newest|latest|revised|amended)\b", re.I)


def parse_marker(text):
    """Find a version hint in a file name or label -> (label, kind) or (None, None)."""
    t = re.sub(r"[_\-]+", " ", text or "")
    m = V_NUM.search(t)
    if m:
        return "v" + m.group(1), float(m.group(1))
    m = YEAR.search(t)
    if m:
        return m.group(1), float(m.group(1))
    if OLD_RE.search(t):
        return "Original", "old"
    if NEW_RE.search(t):
        return "Updated", "new"
    return None, None


def resolve_rank(kind, ranks):
    if isinstance(kind, float):
        return kind
    if kind == "old":
        return (min(ranks) - 1) if ranks else 1.0
    return (max(ranks) + 1) if ranks else (2.0 if kind == "new" else 1.0)


def vkey(v):
    return (v["rank"], v["seq"])


def ordered(kb):
    return sorted(kb["versions"], key=vkey) if kb else []


def make_version(label, rank, seq, files, hashes, pages):
    chunks = make_chunks(pages)
    if not chunks:
        return None
    emb = embed([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(emb.shape[1])  # normalized vectors: inner product = cosine
    index.add(emb)
    sents = make_sentences(pages)
    s_emb = s_index = None
    if sents:
        s_emb = embed([s["text"] for s in sents])
        s_index = faiss.IndexFlatIP(s_emb.shape[1])
        s_index.add(s_emb)
    return {"label": label, "rank": rank, "seq": seq, "files": list(files), "hashes": set(hashes), "pages": pages,
            "chunks": chunks, "emb": emb, "index": index,
            "sents": sents, "s_emb": s_emb, "s_index": s_index, "s_norm": {norm(s["text"]) for s in sents}}


def relatedness(new, kb):
    """Share of the new document's clauses that already exist (almost verbatim) in a loaded version."""
    if not new.get("s_index"):
        return 0.0
    best = 0.0
    for old in kb["versions"]:
        if old.get("s_index"):
            sims, _ = old["s_index"].search(new["s_emb"], 1)
            best = max(best, float((sims[:, 0] >= RELATED_SIM).mean()))
    return best


def add_documents(kb, files, explicit_label=None):
    """files = [(filename, bytes)]. Each file lands in a handbook version:
       - explicit_label (form field 'version') puts every file of the request into that version
       - otherwise a marker in the file name decides (v2, 2024, original, updated ...)
       - unmarked files of one request form one new version (v1, v2, ... in upload order)
       An unmarked PDF that shares almost nothing with what is loaded is a different document, so it
       replaces the knowledge base instead of being compared with it.
       Identical files (same SHA-256) are skipped. Returns (kb, info)."""
    kb = kb or {"versions": [], "seq": 0}
    info = {"added": [], "skipped": [], "failed": [], "replaced": False, "new_versions": []}
    seen = {h for v in kb["versions"] for h in v["hashes"]}
    groups = {}  # label key -> {"label", "kind", "files": [(name, data, hash)]}
    for name, data in files:
        h = hashlib.sha256(data).hexdigest()
        if h in seen:
            info["skipped"].append(name)
            continue
        seen.add(h)
        if explicit_label:
            label, kind = explicit_label, parse_marker(explicit_label)[1]
        else:
            label, kind = parse_marker(os.path.splitext(name)[0])
        key = (label or "").lower()
        groups.setdefault(key, {"label": label, "kind": kind, "files": []})["files"].append((name, data, h))

    for g in groups.values():
        pages, names, hashes = [], [], []
        for name, data, h in g["files"]:
            got = extract_pages(data, name)
            if got:
                pages += got
                names.append(name)
                hashes.append(h)
            else:
                info["failed"].append(name)
        if not pages:
            continue
        existing = next((v for v in kb["versions"] if g["label"] and v["label"].lower() == g["label"].lower()), None)
        if existing:  # more files for a version we already have: rebuild it
            new = make_version(existing["label"], existing["rank"], existing["seq"], existing["files"] + names,
                               existing["hashes"] | set(hashes), existing["pages"] + pages)
            kb["versions"][kb["versions"].index(existing)] = new
        else:
            ver = make_version(None, 0.0, 0, names, hashes, pages)
            if kb["versions"] and not g["label"] and not info["new_versions"] and relatedness(ver, kb) < RELATED_MIN:
                kb["versions"], info["replaced"] = [], True  # unrelated PDF: start a fresh knowledge base
            label = g["label"]
            if not label:
                n = len(kb["versions"]) + 1
                while any(v["label"].lower() == f"v{n}" for v in kb["versions"]):
                    n += 1
                label = f"v{n}"
            kb["seq"] += 1
            ver.update(label=label, rank=resolve_rank(g["kind"], [v["rank"] for v in kb["versions"]]), seq=kb["seq"])
            kb["versions"].append(ver)
            info["new_versions"].append(label)
        info["added"] += names
    return kb, info


UNI_RE = re.compile(r"\b((?:[A-Z][\w&.'\-]+\s+){1,4}(?:University|College|Institute(?:\s+of\s+[A-Z][\w&.'\-]+(?:\s+[A-Z][\w&.'\-]+){0,2})?)"
                    r"|University\s+of\s+[A-Z][\w&.'\-]+(?:\s+[A-Z][\w&.'\-]+){0,2})")


def guess_university_name(pages):
    """Best-effort institution name from the first pages of an uploaded PDF (used when the name box is empty)."""
    found = {}
    for p in pages[:3]:
        for m in UNI_RE.finditer(p["text"][:3000]):
            name = re.sub(r"^(?:The|Welcome To|Welcome to|Of)\s+", "", m.group(1).strip())
            if len(name.split()) >= 2:
                found[name] = found.get(name, 0) + 1
    return max(found, key=found.get) if found else None


def stats(kb):
    if not kb or not kb["versions"]:
        return None
    vs = ordered(kb)
    return {
        "chunks": sum(len(v["chunks"]) for v in vs),
        "documents": sum(len(v["files"]) for v in vs),
        "pages": sum(len({(p["source"], p["page"]) for p in v["pages"]}) for v in vs),
        "versions": [{"label": v["label"], "rank": v["rank"], "files": v["files"], "chunks": len(v["chunks"]),
                      "newest": v is vs[-1]} for v in vs],
    }


def search_chunks(v, qvec, top_k=TOP_K):
    scores, ids = v["index"].search(qvec, min(top_k, len(v["chunks"])))
    out = []
    for s, i in zip(scores[0], ids[0]):
        if i >= 0:
            item = dict(v["chunks"][i])
            item.update(score=float(s), version=v["label"], rank=v["rank"])
            out.append(item)
    return out


# ---------------- CLAUSE COMPARISON (rule-based first, LLM only when unsure) ----------------
SECTION_REF = re.compile(r"\b(?:section|sec|article|chapter|clause|appendix|table|figure|page|pp?|policy|rule)\.?\s*\d+(?:\.\d+)*|\u00a7\s*\d+(?:\.\d+)*")
NUM_RE = re.compile(r"\d+(?:\.\d+)?%?")
WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
           "ninety": 90}
WORDNUM_RE = re.compile(r"\b(" + "|".join(WORDNUM) + r")\b(?=\s+(?:business |working |calendar )?(?:day|week|month|hour|year|credit|semester|time|attempt|absence|copy|copies|percent))")
MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
MONTH_A = re.compile(r"\b" + MON + r"\.?\s+\d{1,2}\b")
MONTH_B = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?" + MON + r"\b")
NEUTRAL = re.compile(r"\b(?:no|not)\s+(?:later|earlier|more|less|fewer|greater|longer|shorter|only)\s+than\b|\bnot only\b|\bno one\b")
MODAL = {
    "oblig": re.compile(r"\b(must|shall|required?|requires?|mandatory|compulsory|obligatory|need to|needs to|have to|has to)\b"),
    "option": re.compile(r"\b(may|optional|can|could|permitted|allowed|allows?|eligible|encouraged|recommended)\b"),
    "prohib": re.compile(r"\b(not|no|never|cannot|prohibited|forbidden|banned?|disallowed|barred|without)\b|n't"),
}
GAINED = {"oblig": "now required", "option": "now optional/allowed", "prohib": "now prohibited/negated"}
LOST = {"oblig": "no longer required", "option": "no longer optional/allowed", "prohib": "no longer prohibited/negated"}


def facts(text):
    """Numbers, percentages, day counts and dates in a clause (section/page references are ignored)."""
    t = SECTION_REF.sub(" ", text.lower())
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)
    t = re.sub(r"\s*(?:%|per\s?cent|percent)\b", "%", t)
    t = WORDNUM_RE.sub(lambda m: str(WORDNUM[m.group(1)]), t)
    out = set(NUM_RE.findall(t))
    for rx in (MONTH_A, MONTH_B):
        out |= {"m:" + m.group(1) for m in rx.finditer(t)}
    return out


def modality(text):
    t = NEUTRAL.sub(" ", text.lower())
    return {k for k, rx in MODAL.items() if rx.search(t)}


def words(text):
    return {w for w in re.findall(r"[a-z]{3,}", text.lower())}


def show_facts(items):
    return ", ".join(sorted(x[2:].title() if x.startswith("m:") else x for x in items)) or "(none)"


def compare_clauses(new_t, old_t, sim):
    """Cheap, deterministic comparison of two aligned clauses."""
    fn, fo = facts(new_t), facts(old_t)
    mn, mo = modality(new_t), modality(old_t)
    parts = []
    if fn != fo:
        parts.append(f"Values changed: {show_facts(fo - fn)} \u2192 {show_facts(fn - fo)}")
    if mn != mo:
        parts.append("Rule changed: " + ", ".join([GAINED[k] for k in sorted(mn - mo)] + [LOST[k] for k in sorted(mo - mn)]))
    wn, wo = words(new_t), words(old_t)
    overlap = len(wn & wo) / max(1, len(wn | wo))
    ratio = difflib.SequenceMatcher(None, norm(old_t), norm(new_t)).ratio()
    return {"signal": bool(parts), "change": "; ".join(parts), "ratio": ratio,
            "similar": sim >= SIMILAR_MIN and overlap >= OVERLAP_MIN}


def ollama_json(model, system, user):
    try:
        r = requests.post(f"{OLLAMA}/api/chat", timeout=JUDGE_TIMEOUT, json={
            "model": model, "stream": False, "format": "json", "keep_alive": "1h",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "num_predict": 160, "num_ctx": 2048}})
        r.raise_for_status()
        txt = r.json().get("message", {}).get("content", "")
        try:
            return json.loads(txt)
        except Exception:
            m = re.search(r"\{.*\}", txt, re.S)
            return json.loads(m.group(0)) if m else None
    except Exception as e:
        print("Judge error:", e)
        return None


JUDGE_SYSTEM = (
    "You compare two versions of an institutional handbook. Decide whether the OLD and NEW passages state different, "
    "incompatible rules, values, deadlines or conditions about the same subject. Rewording that keeps the same rule is NOT a conflict. "
    'Reply with JSON only: {"conflict": true or false, "topic": "short subject", '
    '"old": "what the old passage says (max 20 words)", "new": "what the new passage says (max 20 words)"}'
)


def decide_conflict(question, new_t, old_t, sim, model, budget):
    """Returns a change description if the clauses contradict, else None."""
    c = compare_clauses(new_t, old_t, sim)
    if c["signal"] and c["similar"]:
        return c["change"]  # e.g. 75% -> 80%, "may" -> "must": no LLM needed
    if (c["signal"] or c["ratio"] < 0.85) and sim >= 0.6 and budget["n"] > 0:
        budget["n"] -= 1
        j = ollama_json(model, JUDGE_SYSTEM, f"QUESTION: {question}\nOLD: {old_t}\nNEW: {new_t}")
        if isinstance(j, dict) and j.get("conflict") in (True, "true", "True"):
            topic = str(j.get("topic") or "").strip()
            old_s, new_s = str(j.get("old") or "").strip(), str(j.get("new") or "").strip()
            desc = f"Older says: {old_s}. Newer says: {new_s}." if old_s and new_s else (c["change"] or "The rule was reworded materially.")
            return f"{topic}: {desc}" if topic else desc
    return None


def ref(v, i):
    s = v["sents"][i]
    return {"source": s["source"], "page": s["page"], "label": v["label"], "text": s["text"]}


def find_conflicts(question, qvec, newv, oldvs, model):
    """Compare the newest relevant clauses with the matching clauses of every older version."""
    if not newv.get("s_index"):
        return []
    k = min(KEY_SENTENCES * 2, len(newv["sents"]))
    sc, ids = newv["s_index"].search(qvec, k)
    keys = [(int(i), float(s)) for s, i in zip(sc[0], ids[0]) if i >= 0 and s >= SENT_MIN][:KEY_SENTENCES]
    if not keys:
        return []
    kvecs = newv["s_emb"][[i for i, _ in keys]]
    budget, done, found = {"n": MAX_JUDGE_CALLS}, set(), []
    for old in reversed(oldvs):  # nearest older version first
        if not old.get("s_index"):
            continue
        sims, oids = old["s_index"].search(kvecs, 1)
        for (nid, rel), sim, oid in zip(keys, sims[:, 0], oids[:, 0]):
            sim, oid = float(sim), int(oid)
            if nid in done or oid < 0 or sim < ALIGN_MIN:
                continue
            new_t, old_t = newv["sents"][nid]["text"], old["sents"][oid]["text"]
            if norm(new_t) in old["s_norm"] or norm(old_t) in newv["s_norm"]:
                continue  # the same clause still exists unchanged
            change = decide_conflict(question, new_t, old_t, sim, model, budget)
            if change:
                done.add(nid)
                found.append({"rel": rel, "sim": sim, "change": change, "new": ref(newv, nid), "old": ref(old, oid)})
    found.sort(key=lambda c: -c["rel"])
    return found[:MAX_CONFLICTS]


def diff_versions(newv, oldv, limit=200):
    """Scan two whole versions for contradicting clauses (rule-based only, no LLM)."""
    if not newv.get("s_index") or not oldv.get("s_index"):
        return []
    out = []
    for start in range(0, len(newv["sents"]), 512):
        sims, oids = oldv["s_index"].search(newv["s_emb"][start:start + 512], 1)
        for j, (sim, oid) in enumerate(zip(sims[:, 0], oids[:, 0])):
            sim, oid, nid = float(sim), int(oid), start + j
            if oid < 0 or sim < ALIGN_MIN:
                continue
            new_t, old_t = newv["sents"][nid]["text"], oldv["sents"][oid]["text"]
            if norm(new_t) in oldv["s_norm"] or norm(old_t) in newv["s_norm"]:
                continue
            c = compare_clauses(new_t, old_t, sim)
            if c["signal"] and c["similar"]:
                out.append({"similarity": round(sim, 3), "change": c["change"], "newer": ref(newv, nid), "older": ref(oldv, oid)})
    out.sort(key=lambda x: -x["similarity"])
    return out[:limit]


# ---------------- HARVARD FOLDER ----------------
def harvard_files():
    if not os.path.isdir(HARVARD_FOLDER):
        return []
    return sorted(f for f in os.listdir(HARVARD_FOLDER) if f.lower().endswith(".pdf"))


def harvard_signature():
    parts = []
    for f in harvard_files():
        s = os.stat(os.path.join(HARVARD_FOLDER, f))
        parts.append(f"{f}|{s.st_size}|{s.st_mtime}")
    return hashlib.md5("\n".join(parts).encode()).hexdigest() if parts else "empty"


def prepare_harvard(force=False):
    sig = harvard_signature()
    if sig == "empty":
        return None
    with kb_lock:
        if force or KB["harvard"] is None or KB["harvard_sig"] != sig:
            files = []
            for f in harvard_files():
                with open(os.path.join(HARVARD_FOLDER, f), "rb") as fh:
                    files.append((f, fh.read()))
            kb, _ = add_documents(None, files)  # name a file ..._v2.pdf / ..._updated.pdf to make it a newer version
            KB["harvard"] = kb if kb["versions"] else None
            KB["harvard_sig"] = sig
        return KB["harvard"]


# ---------------- OLLAMA ----------------
def ollama_models():
    try:
        r = requests.get(f"{OLLAMA}/api/tags", timeout=3)
        if r.status_code == 200:
            return True, [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        pass
    return False, []


def clean_history(raw):
    out = []
    for m in (raw or [])[-HISTORY_TURNS:]:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            text = m["content"]
            if m["role"] == "assistant":  # keep old-version values out of later prompts
                text = text.split(BANNER_END)[-1]
                text = re.sub(r"^\u2139\ufe0f.*\n?", "", text, flags=re.M)
            out.append({"role": m["role"], "content": text.strip()[:700]})
    return out


def build_payload(question, retrieved, university, model, version_note="", history=None):
    context = "\n".join(
        f"SOURCE {i}\nDocument: {r['source']}\nPage: {r['page']}\n\n{r['text']}\n"
        for i, r in enumerate(retrieved, 1)
    )
    system = f"""You are a university information assistant for {university}.
Answer using ONLY the document context supplied by the user.
Rules:
1. Never use outside knowledge and never invent courses, fees, deadlines, policies, requirements or contacts.
2. If the context does not support an answer, say exactly: "I could not find this information in the provided university documents."
3. Be direct, concise and use simple language. Use a bullet list if a list is asked for.
4. Mention the page number when the answer comes from a specific page.
5. If several sources answer it, combine them carefully. Do not discuss your retrieval process."""
    if version_note:
        system += "\n6. " + version_note
    user = f"DOCUMENT CONTEXT:\n\n{context}\n\nUSER QUESTION:\n\n{question}\n\nGive a short, accurate answer based only on the document context."
    return {
        "model": model,
        "messages": [{"role": "system", "content": system}] + (history or [])[-4:] + [{"role": "user", "content": user}],
        "stream": True,
        "keep_alive": "1h",
        "options": {"temperature": 0.1, "num_predict": 400, "num_ctx": 4096},
    }


def build_general_payload(question, history, model):
    system = ("You are a friendly, helpful AI assistant for students. Answer clearly and concisely in simple language. "
              "You have no access to any university's documents in this mode: if asked about a specific university's fees, "
              "deadlines, rules or requirements, say you cannot verify them here and suggest the Upload University PDF mode.")
    return {
        "model": model,
        "messages": [{"role": "system", "content": system}] + history + [{"role": "user", "content": question}],
        "stream": True,
        "keep_alive": "1h",
        "options": {"temperature": 0.6, "num_predict": 500, "num_ctx": 4096},
    }


def stream_chat(payload):
    """Yield NDJSON token lines from Ollama, turning failures into readable messages."""
    try:
        with requests.post(f"{OLLAMA}/api/chat", json=payload, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if raw:
                    tok = json.loads(raw).get("message", {}).get("content", "")
                    if tok:
                        yield line(type="token", text=tok)
    except requests.exceptions.ConnectionError:
        yield line(type="token", text="Cannot connect to Ollama. Make sure Ollama is running.")
    except requests.exceptions.Timeout:
        yield line(type="token", text="Ollama took too long to respond. Try a shorter question.")
    except Exception as e:
        print("Ollama error:", e)
        yield line(type="token", text=f"Ollama error: {e}. Check that the selected model is installed.")


def line(**kw):
    return json.dumps(kw) + "\n"


# ---------------- ANSWER PLAN: which version answers, and what conflicts ----------------
def clip(text, n=320):
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "\u2026"


def cite(r):
    return f"{r['source']} [{r['label']}], p.{r['page']}"


def retrieval_query(question, history):
    """A short follow-up ("what about fees?") is searched together with the previous question."""
    prev = [m["content"] for m in history if m["role"] == "user"]
    return f"{prev[-1]} {question}" if prev and len(question.split()) <= 6 else question


def plan_answer(question, kb, model, query=None):
    vs = ordered(kb)
    qvec = embed([query or question])
    multi = len(vs) > 1
    answering, retrieved = None, []
    for v in reversed(vs):  # newest version first
        hits = [h for h in search_chunks(v, qvec) if h["score"] >= MIN_SCORE]
        if hits:
            answering, retrieved = v, hits
            break
    if not answering:
        return None
    plan = {"multi": multi, "answering": answering, "retrieved": retrieved, "conflicts": [], "banner": "", "note": ""}
    if not multi:
        return plan

    older = [v for v in vs if vkey(v) < vkey(answering)]
    plan["conflicts"] = find_conflicts(question, qvec, answering, older, model) if older else []
    parts = []
    if answering is not vs[-1]:
        parts.append(f"\u2139\ufe0f Only an older version ({answering['label']}) covers this. The newer version ({vs[-1]['label']}) "
                     "has no matching section, so treat this as possibly outdated.\n")
        plan["note"] = f"The passages come from an older handbook version ({answering['label']}); the newest version has no matching section. Say so."
    if plan["conflicts"]:
        lines = ["\u26a0\ufe0f VERSION CONFLICT: this contradicts an older version of the handbook.",
                 f"The answer below follows the newer policy ({answering['label']}).", ""]
        for n, c in enumerate(plan["conflicts"], 1):
            lines += [f"{n}. NEWER \u2192 {cite(c['new'])}", f"   \u201c{clip(c['new']['text'])}\u201d",
                      f"   OLDER \u2192 {cite(c['old'])} (superseded)", f"   \u201c{clip(c['old']['text'])}\u201d",
                      f"   What changed: {c['change']}", ""]
        lines.append(BANNER_END)
        parts.append("\n".join(lines))
        plan["note"] = ("A different, older handbook version conflicts with this context and is flagged to the user separately. "
                        "Answer with the current rule only and do not mention the older values.")
        # make sure the conflicting newer clauses are really in the context the model sees
        for c in plan["conflicts"]:
            if not any(c["new"]["text"] in r["text"] for r in retrieved):
                retrieved.append({"source": c["new"]["source"], "page": c["new"]["page"], "text": c["new"]["text"],
                                  "score": c["sim"], "version": answering["label"]})
    plan["banner"] = "\n".join(parts)
    return plan


def source_rows(plan):
    tag = plan["multi"]
    rows = [{"source": f"{r['source']} [{plan['answering']['label']}]" if tag else r["source"], "page": r["page"],
             "score": r["score"], "text": r["text"][:700]} for r in plan["retrieved"]]
    for c in plan["conflicts"]:
        rows.append({"source": f"{c['old']['source']} [{c['old']['label']}] (superseded)", "page": c["old"]["page"],
                     "score": c["sim"], "text": c["old"]["text"][:700]})
    return rows


# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html", default_model=DEFAULT_MODEL, top_k=TOP_K)


@app.route("/status")
def status():
    ok, models = ollama_models()
    return jsonify({
        "ollama": ok,
        "models": models,
        "default_model": DEFAULT_MODEL,
        "harvard_docs": len(harvard_files()),
        "harvard": stats(KB["harvard"]),
        "custom": stats(KB["custom"]),
        "custom_name": KB["custom_name"],
    })


@app.route("/prepare", methods=["POST"])
def prepare():
    force = bool((request.get_json(silent=True) or {}).get("force"))
    if not harvard_files():
        return jsonify({"ok": False, "message": "No PDFs found. Put your Harvard PDF inside the harvard_docs folder."})
    kb = prepare_harvard(force)
    if not kb:
        return jsonify({"ok": False, "message": "No readable text found in the Harvard PDFs."})
    return jsonify({"ok": True, "stats": stats(kb)})


@app.route("/upload", methods=["POST"])
def upload():
    """Read the uploaded university PDF(s); chat answers then come only from them (Harvard is not involved).
       - a different university PDF replaces the previous one
       - an updated copy of the same handbook is added as the newer version (conflicts are flagged in answers)
       Optional form fields: name, version (label, e.g. "2024 Updated"), replace=1 (start over)."""
    files = [(f.filename or "document.pdf", f.read()) for f in request.files.getlist("files")]
    if not files:
        return jsonify({"ok": False, "message": "Choose at least one PDF first."})
    label = (request.form.get("version") or "").strip() or None
    replace = (request.form.get("replace") or "").lower() in ("1", "true", "yes", "on")
    typed = (request.form.get("name") or "").strip()
    with kb_lock:
        base = None if replace else KB["custom"]
        kb, info = add_documents(base, files, label)
        if not kb["versions"]:
            return jsonify({"ok": False, "message": "No readable text was found in the uploaded PDF."})
        fresh = base is None or info["replaced"]
        KB["custom"] = kb
        if typed:
            KB["custom_name"] = typed
        elif fresh:
            new = next((v for v in kb["versions"] if info["new_versions"] and v["label"] == info["new_versions"][0]), None)
            KB["custom_name"] = (guess_university_name(new["pages"]) if new else None) or "Uploaded University"
    st = stats(kb)
    newest = ordered(kb)[-1]["label"]
    if not info["added"]:
        msg = "These files are already loaded."
    elif fresh:
        msg = (f"Read {st['pages']} pages ({st['chunks']} knowledge chunks). "
               f"Answers now come only from this university PDF.")
        if info["replaced"]:
            msg = "This looks like a different document, so it replaced the previous one. " + msg
    else:
        msg = (f"Added as a newer version ({', '.join(info['new_versions']) or newest}). "
               f"Answers will use the newest policy ({newest}) and flag anything that contradicts an older version.")
    return jsonify({"ok": True, "stats": st, "name": KB["custom_name"], "message": msg,
                    "added": info["added"], "skipped_duplicates": info["skipped"], "unreadable": info["failed"]})


@app.route("/reset", methods=["POST"])
def reset():
    with kb_lock:
        KB["custom"], KB["custom_name"] = None, "Uploaded University"
    return jsonify({"ok": True})


@app.route("/versions")
def versions():
    mode = request.args.get("mode", "custom")
    return jsonify({"ok": True, "stats": stats(KB["harvard"] if mode == "harvard" else KB["custom"])})


@app.route("/diff")
def diff():
    """List every clause that contradicts between the newest version and each older one."""
    mode = request.args.get("mode", "custom")
    kb = prepare_harvard() if mode == "harvard" else KB["custom"]
    vs = ordered(kb)
    if len(vs) < 2:
        return jsonify({"ok": False, "message": "Load at least two handbook versions first."})
    newest = vs[-1]
    return jsonify({"ok": True, "newer": newest["label"], "comparisons": [
        {"older": o["label"], "conflicts": diff_versions(newest, o)} for o in reversed(vs[:-1])]})


@app.route("/chat", methods=["POST"])
def chat():
    d = request.get_json(force=True)
    question = (d.get("question") or "").strip()
    model = d.get("model") or DEFAULT_MODEL
    mode = d.get("mode") or "general"
    history = clean_history(d.get("history"))
    if mode == "harvard":
        kb, university = prepare_harvard(), "Harvard University"
        empty_msg = "No Harvard documents are loaded yet. Put PDFs in the harvard_docs folder and click Reload Harvard Documents."
    elif mode == "custom":  # only the uploaded university PDF is used here
        kb, university = KB["custom"], KB["custom_name"]
        empty_msg = "No university PDF is loaded yet. Choose a PDF in the sidebar and click Process University Documents first."
    else:
        kb, university, empty_msg = None, "", ""

    def generate():
        if not question:
            return
        if mode == "general":  # plain chat, no documents
            yield from stream_chat(build_general_payload(question, history, model))
            yield line(type="done")
            return
        if question.lower().strip(" !.?") in GREETINGS:
            yield line(type="token", text=f"Hello! I can answer questions about {university} using its documents. What would you like to know?")
            yield line(type="done")
            return
        if not kb:
            yield line(type="token", text=empty_msg)
            yield line(type="done")
            return
        plan = plan_answer(question, kb, model, retrieval_query(question, history))
        if not plan:
            yield line(type="token", text="I could not find relevant information in the provided university documents.")
            yield line(type="done")
            return
        yield line(type="sources", sources=source_rows(plan))
        if plan["banner"]:
            yield line(type="token", text=plan["banner"] + "\n")
        yield from stream_chat(build_payload(question, plan["retrieved"], university, model, plan["note"], history))
        yield line(type="done")

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


def warm_up():
    try:
        embedder()  # Harvard documents are prepared only when Harvard mode is opened
    except Exception as e:
        print("Warm-up error:", e)


if __name__ == "__main__":
    os.makedirs(HARVARD_FOLDER, exist_ok=True)
    threading.Thread(target=warm_up, daemon=True).start()
    app.run(debug=True, use_reloader=False)
