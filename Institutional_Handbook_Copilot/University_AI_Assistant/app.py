
import os
import io
import hashlib
import requests
import fitz
import numpy as np
import streamlit as st
import faiss

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"

DEFAULT_MODEL = "llama3.2:3b"

HARVARD_FOLDER = "harvard_docs"

CACHE_FOLDER = "cache"

# Small and fast semantic embedding model.
# It runs locally and does not require a cloud API.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Number of semantic results sent to Ollama.
TOP_K = 4

# Chunk configuration.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

# Ollama timeout.
OLLAMA_TIMEOUT = 120


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="University AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 16px;
        opacity: 0.75;
        margin-bottom: 20px;
    }

    .source-box {
        padding: 10px;
        border-radius: 8px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 8px;
    }

    .small-text {
        font-size: 13px;
        opacity: 0.75;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 University AI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Fast semantic RAG assistant powered by local Ollama and university documents'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "harvard_knowledge" not in st.session_state:
    st.session_state.harvard_knowledge = None

if "custom_knowledge" not in st.session_state:
    st.session_state.custom_knowledge = None

if "custom_name" not in st.session_state:
    st.session_state.custom_name = None

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():

    return SentenceTransformer(EMBEDDING_MODEL)


def get_embedding_model():

    if st.session_state.embedding_model is None:

        st.session_state.embedding_model = load_embedding_model()

    return st.session_state.embedding_model


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(pdf_bytes):

    pages = []

    try:

        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        for page_number, page in enumerate(document):

            text = page.get_text("text")

            if text and text.strip():

                clean_text = " ".join(
                    text.split()
                )

                if len(clean_text) >= 20:

                    pages.append(
                        {
                            "page": page_number + 1,
                            "text": clean_text
                        }
                    )

        document.close()

    except Exception as e:

        print("PDF extraction error:", e)

    return pages


# ============================================================
# CHUNK TEXT
# ============================================================

def create_chunks(
    documents,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    chunks = []

    for document in documents:

        text = document["text"]

        # ----------------------------------------------------
        # Prefer sentence-like boundaries where possible.
        # ----------------------------------------------------

        words = text.split()

        current_words = []

        current_length = 0

        chunk_number = 0

        for word in words:

            current_words.append(word)

            current_length += len(word) + 1

            if current_length >= chunk_size:

                chunk_text = " ".join(
                    current_words
                ).strip()

                if chunk_text:

                    chunks.append(
                        {
                            "text": chunk_text,
                            "source": document["source"],
                            "page": document["page"],
                            "chunk": chunk_number
                        }
                    )

                    chunk_number += 1

                # ------------------------------------------------
                # Keep overlap using approximately last words.
                # ------------------------------------------------

                overlap_words = []

                overlap_length = 0

                for previous_word in reversed(
                    current_words
                ):

                    if (
                        overlap_length
                        + len(previous_word)
                        + 1
                        <= overlap
                    ):

                        overlap_words.insert(
                            0,
                            previous_word
                        )

                        overlap_length += (
                            len(previous_word) + 1
                        )

                    else:

                        break

                current_words = overlap_words

                current_length = overlap_length

        # ----------------------------------------------------
        # Remaining text.
        # ----------------------------------------------------

        if current_words:

            chunk_text = " ".join(
                current_words
            ).strip()

            if chunk_text:

                chunks.append(
                    {
                        "text": chunk_text,
                        "source": document["source"],
                        "page": document["page"],
                        "chunk": chunk_number
                    }
                )

    return chunks


# ============================================================
# CREATE SEMANTIC EMBEDDINGS
# ============================================================

def create_embeddings(chunks):

    if not chunks:

        return None

    model = get_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    return embeddings


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_faiss_index(chunks):

    if not chunks:

        return None

    embeddings = create_embeddings(
        chunks
    )

    if embeddings is None:

        return None

    dimension = embeddings.shape[1]

    # Inner product with normalized vectors
    # is equivalent to cosine similarity.
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


# ============================================================
# CREATE KNOWLEDGE BASE
# ============================================================

def create_knowledge_base(documents):

    chunks = create_chunks(
        documents
    )

    if not chunks:

        return None

    index = build_faiss_index(
        chunks
    )

    if index is None:

        return None

    return {
        "chunks": chunks,
        "index": index
    }


# ============================================================
# SEMANTIC RETRIEVAL
# ============================================================

def retrieve_documents(
    query,
    knowledge,
    top_k=TOP_K
):

    if not knowledge:

        return []

    chunks = knowledge["chunks"]

    index = knowledge["index"]

    if not chunks or index is None:

        return []

    model = get_embedding_model()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    search_k = min(
        top_k,
        len(chunks)
    )

    scores, indices = index.search(
        query_embedding,
        search_k
    )

    results = []

    for score, index_value in zip(
        scores[0],
        indices[0]
    ):

        if index_value < 0:

            continue

        result = chunks[index_value].copy()

        result["score"] = float(
            score
        )

        results.append(
            result
        )

    return results


# ============================================================
# CHECK OLLAMA
# ============================================================

def check_ollama():

    try:

        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=3
        )

        if response.status_code == 200:

            data = response.json()

            models = data.get(
                "models",
                []
            )

            model_names = [
                model.get(
                    "name",
                    ""
                )
                for model in models
            ]

            return True, model_names

    except Exception:

        pass

    return False, []


# ============================================================
# OLLAMA ANSWER
# ============================================================

def ask_ollama(
    question,
    retrieved,
    university_name,
    model_name
):

    if not retrieved:

        return (
            "I could not find relevant information "
            "in the provided university documents."
        )

    # --------------------------------------------------------
    # Build compact context.
    # --------------------------------------------------------

    context_parts = []

    for number, item in enumerate(
        retrieved,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {number}
Document: {item["source"]}
Page: {item["page"]}

{item["text"]}
"""
        )

    context = "\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # Strong grounded system prompt.
    # --------------------------------------------------------

    system_prompt = f"""
You are a university information assistant for {university_name}.

Answer the user's question using ONLY the university document
context supplied below.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Do not invent courses, programs, fees, deadlines, policies,
   admission requirements, contact information, or regulations.
4. If the answer is not supported by the supplied context,
   say exactly:
   "I could not find this information in the provided university documents."
5. Answer directly and concisely.
6. Use simple language.
7. Do not discuss your retrieval process.
8. Do not mention information that is not supported by the sources.
9. If the answer comes from a specific page, mention the page number.
10. If several sources provide the answer, combine them carefully.
11. If the question asks for a list, provide a clear bullet list.
12. Do not say that you searched the internet.
13. The supplied documents are the authority for this answer.

University:
{university_name}
"""

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Give a short, accurate answer based only on the document context.
"""

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 250
        }
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get(
            "message",
            {}
        ).get(
            "content",
            ""
        )

        if not answer:

            return (
                "I could not generate an answer."
            )

        return answer.strip()

    except requests.exceptions.ConnectionError:

        return (
            "❌ Cannot connect to Ollama.\n\n"
            "Please make sure Ollama is running."
        )

    except requests.exceptions.Timeout:

        return (
            "❌ Ollama took too long to respond. "
            "Please try a shorter question."
        )

    except Exception as e:

        return (
            f"❌ Ollama error: {str(e)}"
        )


# ============================================================
# HARVARD FILE SIGNATURE
# ============================================================

def get_harvard_signature():

    if not os.path.exists(
        HARVARD_FOLDER
    ):

        return "empty"

    files = []

    for filename in sorted(
        os.listdir(HARVARD_FOLDER)
    ):

        if not filename.lower().endswith(
            ".pdf"
        ):

            continue

        path = os.path.join(
            HARVARD_FOLDER,
            filename
        )

        try:

            stat = os.stat(path)

            files.append(
                f"{filename}|{stat.st_size}|{stat.st_mtime}"
            )

        except Exception:

            pass

    if not files:

        return "empty"

    return hashlib.md5(
        "\n".join(files).encode()
    ).hexdigest()


# ============================================================
# LOAD HARVARD DOCUMENTS
# ============================================================

@st.cache_data(
    show_spinner=False
)
def load_harvard_documents():

    documents = []

    if not os.path.exists(
        HARVARD_FOLDER
    ):

        return documents

    for filename in sorted(
        os.listdir(
            HARVARD_FOLDER
        )
    ):

        if not filename.lower().endswith(
            ".pdf"
        ):

            continue

        path = os.path.join(
            HARVARD_FOLDER,
            filename
        )

        try:

            with open(
                path,
                "rb"
            ) as file:

                pdf_bytes = file.read()

            pages = extract_pdf_text(
                pdf_bytes
            )

            for page_data in pages:

                documents.append(
                    {
                        "source": filename,
                        "page": page_data["page"],
                        "text": page_data["text"]
                    }
                )

        except Exception as e:

            print(
                "Error reading:",
                filename,
                e
            )

    return documents


# ============================================================
# CACHE HARVARD KNOWLEDGE BASE
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def build_harvard_cached(
    signature
):

    raw_documents = load_harvard_documents()

    return create_knowledge_base(
        raw_documents
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Configuration"
    )

    ollama_ok, installed_models = check_ollama()

    if ollama_ok:

        st.success(
            "Ollama is running"
        )

        if installed_models:

            # Prefer llama3.2:3b when available.
            preferred_index = 0

            if DEFAULT_MODEL in installed_models:

                preferred_index = (
                    installed_models.index(
                        DEFAULT_MODEL
                    )
                )

            model_name = st.selectbox(
                "Ollama Model",
                installed_models,
                index=preferred_index
            )

        else:

            model_name = st.text_input(
                "Ollama Model",
                DEFAULT_MODEL
            )

    else:

        st.error(
            "Ollama is not running"
        )

        model_name = st.text_input(
            "Ollama Model",
            DEFAULT_MODEL
        )

        st.info(
            "Start Ollama before asking questions."
        )

    st.divider()

    st.header(
        "🎓 University Mode"
    )

    mode = st.radio(
        "Choose an option:",
        [
            "1. Harvard Chat",
            "2. Upload University PDF"
        ]
    )

    st.divider()

    st.header(
        "🧠 Retrieval"
    )

    st.success(
        "Semantic Search"
    )

    st.caption(
        "MiniLM embeddings + FAISS"
    )

    st.caption(
        f"Top results: {TOP_K}"
    )

    st.divider()

    st.header(
        "📚 Knowledge Base"
    )

    # --------------------------------------------------------
    # Harvard
    # --------------------------------------------------------

    if mode == "1. Harvard Chat":

        st.info(
            "Using Harvard documents stored in "
            "`harvard_docs/`."
        )

        if st.button(
            "🔄 Reload Harvard Documents"
        ):

            load_harvard_documents.clear()

            build_harvard_cached.clear()

            st.session_state.harvard_knowledge = None

            st.session_state.messages = []

            st.rerun()

    # --------------------------------------------------------
    # Other University
    # --------------------------------------------------------

    else:

        uploaded_files = st.file_uploader(
            "Upload university PDF(s)",
            type=["pdf"],
            accept_multiple_files=True
        )

        university_name_input = st.text_input(
            "University name",
            placeholder="Example: CMR University"
        )

        if uploaded_files:

            if st.button(
                "📖 Process University Documents"
            ):

                with st.spinner(
                    "Creating semantic knowledge base..."
                ):

                    all_pages = []

                    for uploaded_file in uploaded_files:

                        pdf_bytes = (
                            uploaded_file.getvalue()
                        )

                        pages = extract_pdf_text(
                            pdf_bytes
                        )

                        for page_data in pages:

                            all_pages.append(
                                {
                                    "source": uploaded_file.name,
                                    "page": page_data["page"],
                                    "text": page_data["text"]
                                }
                            )

                    knowledge = create_knowledge_base(
                        all_pages
                    )

                    if knowledge:

                        st.session_state.custom_knowledge = (
                            knowledge
                        )

                        st.session_state.custom_name = (
                            university_name_input.strip()
                            if university_name_input.strip()
                            else "Uploaded University"
                        )

                        st.session_state.messages = []

                        st.success(
                            f"Created {len(knowledge['chunks'])} "
                            "semantic knowledge chunks."
                        )

                    else:

                        st.error(
                            "No readable text was found "
                            "in the uploaded PDF."
                        )

    st.divider()

    if st.button(
        "🗑️ Clear Chat"
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# LOAD ACTIVE KNOWLEDGE BASE
# ============================================================

knowledge = None

university_name = "University"


# ============================================================
# HARVARD MODE
# ============================================================

if mode == "1. Harvard Chat":

    signature = get_harvard_signature()

    if signature == "empty":

        st.warning(
            "No Harvard PDF documents found."
        )

        st.info(
            "Put your Harvard PDF inside the "
            "`harvard_docs` folder."
        )

        st.stop()

    if st.session_state.harvard_knowledge is None:

        with st.spinner(
            "Preparing Harvard semantic knowledge base..."
        ):

            st.session_state.harvard_knowledge = (
                build_harvard_cached(
                    signature
                )
            )

    knowledge = (
        st.session_state.harvard_knowledge
    )

    university_name = "Harvard University"


# ============================================================
# CUSTOM UNIVERSITY MODE
# ============================================================

else:

    knowledge = (
        st.session_state.custom_knowledge
    )

    university_name = (
        st.session_state.custom_name
        if st.session_state.custom_name
        else "Uploaded University"
    )


# ============================================================
# MAIN HEADER
# ============================================================

if mode == "1. Harvard Chat":

    st.subheader(
        "🏛️ Harvard University Assistant"
    )

else:

    st.subheader(
        f"🏫 {university_name} Assistant"
    )


# ============================================================
# KNOWLEDGE STATUS
# ============================================================

if knowledge is None:

    if mode == "2. Upload University PDF":

        st.info(
            "Upload a university PDF from the sidebar "
            "and click **Process University Documents**."
        )

    st.stop()


chunks = knowledge.get(
    "chunks",
    []
)


if not chunks:

    st.warning(
        "No readable knowledge chunks are available."
    )

    st.stop()


# ============================================================
# KNOWLEDGE METRICS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "🧠 Knowledge Chunks",
        len(chunks)
    )

with col2:

    unique_sources = len(
        set(
            chunk["source"]
            for chunk in chunks
        )
    )

    st.metric(
        "📄 Documents",
        unique_sources
    )

with col3:

    unique_pages = len(
        set(
            (
                chunk["source"],
                chunk["page"]
            )
            for chunk in chunks
        )
    )

    st.metric(
        "📑 Pages",
        unique_pages
    )


st.divider()


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 Sources used"
            ):

                for source in message["sources"]:

                    st.markdown(
                        f"""
                        <div class="source-box">

                        <b>📄 {source["source"]}</b><br>

                        Page: <b>{source["page"]}</b><br>

                        Semantic relevance:
                        <b>{source["score"]:.3f}</b>

                        <br><br>

                        <span class="small-text">
                        {source["text"][:700]}...
                        </span>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# ============================================================
# USER QUESTION
# ============================================================

question = st.chat_input(
    "Ask a question about the university..."
)


if question:

    # --------------------------------------------------------
    # Add user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # --------------------------------------------------------
    # SEMANTIC RETRIEVAL
    # --------------------------------------------------------

    retrieved = retrieve_documents(
        question,
        knowledge,
        top_k=TOP_K
    )


    # --------------------------------------------------------
    # ANSWER
    # --------------------------------------------------------

    if not retrieved:

        answer = (
            "I could not find relevant information "
            "in the provided university documents."
        )

    else:

        with st.spinner(
            "Generating answer..."
        ):

            answer = ask_ollama(
                question=question,
                retrieved=retrieved,
                university_name=university_name,
                model_name=model_name
            )


    # --------------------------------------------------------
    # DISPLAY ANSWER
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            answer
        )

        if retrieved:

            with st.expander(
                "📚 Sources used"
            ):

                for source in retrieved:

                    st.markdown(
                        f"""
                        <div class="source-box">

                        <b>📄 {source["source"]}</b><br>

                        Page: <b>{source["page"]}</b><br>

                        Semantic relevance:
                        <b>{source["score"]:.3f}</b>

                        <br><br>

                        <span class="small-text">
                        {source["text"][:700]}...
                        </span>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


    # --------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": retrieved
        }
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Local University AI Assistant • "
    "MiniLM Semantic RAG + FAISS + Ollama • "
    "No cloud LLM required"
)
