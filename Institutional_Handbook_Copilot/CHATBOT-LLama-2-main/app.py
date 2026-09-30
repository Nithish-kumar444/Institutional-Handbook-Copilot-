import os
import json
import hashlib
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


# ---------------- PDF -> CHUNKS -> FAISS ----------------
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


def build_kb(pages):
    chunks = make_chunks(pages)
    if not chunks:
        return None
    emb = embed([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(emb.shape[1])  # normalized vectors: inner product = cosine
    index.add(emb)
    return {"chunks": chunks, "index": index}


def stats(kb):
    if not kb:
        return None
    c = kb["chunks"]
    return {
        "chunks": len(c),
        "documents": len({x["source"] for x in c}),
        "pages": len({(x["source"], x["page"]) for x in c}),
    }


def retrieve(query, kb, top_k=TOP_K):
    if not kb:
        return []
    scores, ids = kb["index"].search(embed([query]), min(top_k, len(kb["chunks"])))
    out = []
    for s, i in zip(scores[0], ids[0]):
        if i >= 0:
            item = dict(kb["chunks"][i])
            item["score"] = float(s)
            out.append(item)
    return out


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
            pages = []
            for f in harvard_files():
                with open(os.path.join(HARVARD_FOLDER, f), "rb") as fh:
                    pages += extract_pages(fh.read(), f)
            KB["harvard"] = build_kb(pages)
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


def build_payload(question, retrieved, university, model):
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
    user = f"DOCUMENT CONTEXT:\n\n{context}\n\nUSER QUESTION:\n\n{question}\n\nGive a short, accurate answer based only on the document context."
    return {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": True,
        "keep_alive": "1h",
        "options": {"temperature": 0.1, "num_predict": 400, "num_ctx": 4096},
    }


def line(**kw):
    return json.dumps(kw) + "\n"


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
    pages = []
    for f in request.files.getlist("files"):
        pages += extract_pages(f.read(), f.filename)
    kb = build_kb(pages)
    if not kb:
        return jsonify({"ok": False, "message": "No readable text was found in the uploaded PDF."})
    with kb_lock:
        KB["custom"] = kb
        KB["custom_name"] = (request.form.get("name") or "").strip() or "Uploaded University"
    return jsonify({"ok": True, "stats": stats(kb), "name": KB["custom_name"]})


@app.route("/chat", methods=["POST"])
def chat():
    d = request.get_json(force=True)
    question = (d.get("question") or "").strip()
    model = d.get("model") or DEFAULT_MODEL
    if d.get("mode") == "harvard":
        kb, university = prepare_harvard(), "Harvard University"
    else:
        kb, university = KB["custom"], KB["custom_name"]

    def generate():
        if not question:
            return
        if question.lower().strip(" !.?") in GREETINGS:
            yield line(type="token", text=f"Hello! I can answer questions about {university} using its documents. What would you like to know?")
            yield line(type="done")
            return
        if not kb:
            yield line(type="token", text="No documents are loaded yet. Load a knowledge base from the sidebar first.")
            yield line(type="done")
            return
        retrieved = [r for r in retrieve(question, kb) if r["score"] >= MIN_SCORE]
        if not retrieved:
            yield line(type="token", text="I could not find relevant information in the provided university documents.")
            yield line(type="done")
            return
        yield line(type="sources", sources=[
            {"source": r["source"], "page": r["page"], "score": r["score"], "text": r["text"][:700]} for r in retrieved
        ])
        try:
            with requests.post(f"{OLLAMA}/api/chat", json=build_payload(question, retrieved, university, model),
                               stream=True, timeout=TIMEOUT) as r:
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
        yield line(type="done")

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


def warm_up():
    try:
        embedder()
        prepare_harvard()
    except Exception as e:
        print("Warm-up error:", e)


if __name__ == "__main__":
    os.makedirs(HARVARD_FOLDER, exist_ok=True)
    threading.Thread(target=warm_up, daemon=True).start()
    app.run(debug=True, use_reloader=False)

