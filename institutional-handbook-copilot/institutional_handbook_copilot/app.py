"""
Institutional Handbook Copilot — Version-Aware RAG with Conflict Detection
Streamlit Web Application
"""

import os
import sys
import tempfile
import streamlit as st

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core import (
    HandbookCopilot,
    extract_and_chunk_pdf,
    detect_conflicts_in_retrieved_chunks,
    deterministic_grounded_answer,
    query_ollama,
    HAS_PYMUPDF,
    HAS_SENTENCE_TRANSFORMERS,
    HAS_FAISS
)

# Page configuration
st.set_page_config(
    page_title="Institutional Handbook Copilot",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished institutional styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .conflict-card {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    .conflict-header {
        color: #B91C1C;
        font-weight: 700;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .policy-box-current {
        background-color: #F0FDF4;
        border-left: 4px solid #22C55E;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .policy-box-older {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .citation-tag {
        display: inline-block;
        background-color: #E2E8F0;
        color: #1E293B;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .chunk-card {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "copilot" not in st.session_state:
    st.session_state.copilot = HandbookCopilot()
if "uploaded_docs_meta" not in st.session_state:
    st.session_state.uploaded_docs_meta = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =====================================================================
# Sidebar: Document Management & Settings
# =====================================================================
with st.sidebar:
    st.header("📂 Document Repository")
    
    # Quick Demo Loader
    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_docs")
    sample_2025 = os.path.join(sample_dir, "handbook_2025.pdf")
    sample_2026 = os.path.join(sample_dir, "handbook_2026.pdf")
    
    if os.path.exists(sample_2025) and os.path.exists(sample_2026):
        if st.button("🚀 Load Sample 2025 & 2026 Handbooks", use_container_width=True, type="primary"):
            with st.spinner("Indexing sample handbooks..."):
                stats_25 = st.session_state.copilot.load_pdf(sample_2025)
                stats_26 = st.session_state.copilot.load_pdf(sample_2026)
                st.session_state.uploaded_docs_meta[stats_25["filename"]] = stats_25
                st.session_state.uploaded_docs_meta[stats_26["filename"]] = stats_26
                st.success("Loaded 2025 & 2026 sample handbooks with conflicting policies!")
    
    st.markdown("---")
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload Institutional PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload university handbooks, academic regulations, or compliance guidelines."
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            fname = uploaded_file.name
            if fname not in st.session_state.uploaded_docs_meta:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                with st.spinner(f"Parsing & chunking {fname}..."):
                    try:
                        stats = st.session_state.copilot.load_pdf(tmp_path)
                        # Normalize filename
                        stats["filename"] = fname
                        st.session_state.uploaded_docs_meta[fname] = stats
                        st.success(f"Indexed: {fname} (v{stats['detected_version']})")
                    except Exception as e:
                        st.error(f"Error processing {fname}: {e}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

    # Document Status List
    if st.session_state.uploaded_docs_meta:
        st.subheader("Indexed Handbooks")
        for doc_name, meta in list(st.session_state.uploaded_docs_meta.items()):
            ver_label = meta.get("detected_version", "Unknown")
            is_conf = meta.get("is_confident", True)
            
            with st.expander(f"📄 {doc_name} — v{ver_label}", expanded=False):
                st.write(f"**Pages:** {meta.get('page_count', 0)}")
                st.write(f"**Chunks:** {meta.get('chunk_count', 0)}")
                
                if not is_conf:
                    st.warning("⚠️ Version not confidently detected")
                    override_val = st.text_input(f"Override version for {doc_name}:", key=f"ver_{doc_name}")
                    if st.button(f"Update Version", key=f"btn_{doc_name}"):
                        meta["detected_version"] = override_val
                        st.success(f"Updated to {override_val}")
                        st.rerun()
                else:
                    st.caption(f"Status: Confidently detected ({ver_label})")

        if st.button("🗑️ Clear All Indexed Documents", use_container_width=True):
            st.session_state.copilot.vector_store.clear()
            st.session_state.uploaded_docs_meta = {}
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.subheader("⚙️ LLM & Retrieval Engine")
    ollama_model = st.text_input("Ollama Model", value="llama3.2:3b")
    ollama_host = st.text_input("Ollama Endpoint", value="http://localhost:11434")
    st.session_state.copilot.ollama_model = ollama_model
    st.session_state.copilot.ollama_host = ollama_host

    # Check Ollama connection
    st.caption("Engine Status:")
    st.caption(f"• PyMuPDF: {'Available' if HAS_PYMUPDF else 'Missing'}")
    st.caption(f"• MiniLM-L6-v2: {'Available' if HAS_SENTENCE_TRANSFORMERS else 'Fallback Bag-of-Words'}")
    st.caption(f"• FAISS Index: {'Active' if HAS_FAISS else 'NumPy Vector Store'}")


# =====================================================================
# Main Interface
# =====================================================================
st.markdown('<div class="main-title">📘 Institutional Handbook Copilot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Version-Aware RAG • Conflict Detection • Cited Answers</div>', unsafe_allow_html=True)

if not st.session_state.uploaded_docs_meta:
    st.info("👋 Welcome! Please upload your institutional handbooks (or click **'Load Sample 2025 & 2026 Handbooks'** in the left sidebar) to begin asking policy questions.")

# Quick Sample Prompts
st.markdown("**Try asking:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📌 What is the minimum attendance requirement?", use_container_width=True):
        st.session_state["user_query_input"] = "What is the minimum attendance requirement?"
with col2:
    if st.button("📌 How many backlogs are allowed for promotion?", use_container_width=True):
        st.session_state["user_query_input"] = "How many backlogs are allowed for promotion?"
with col3:
    if st.button("📌 What is the cafeteria menu?", use_container_width=True):
        st.session_state["user_query_input"] = "What is the cafeteria menu?"

# Question Input Box
query_val = st.session_state.get("user_query_input", "")
user_query = st.text_input(
    "Ask a question about the handbook:",
    value=query_val,
    placeholder="e.g. What is the minimum attendance requirement for examinations?",
    key="query_input"
)

if st.button("Submit Question", type="primary") or (user_query and user_query != st.session_state.get("last_run_query", "")):
    if user_query.strip():
        st.session_state["last_run_query"] = user_query.strip()
        with st.spinner("Searching versioned documents, analyzing policy changes, and formulating answer..."):
            result = st.session_state.copilot.ask(user_query.strip(), top_k=10)
            st.session_state.chat_history.insert(0, result)

# Display latest results
if st.session_state.chat_history:
    latest = st.session_state.chat_history[0]
    
    st.markdown("### Answer")
    
    # Conflict Warning Box if contradictory policies found
    if latest.get("has_conflict"):
        rep = latest["conflict_report"]
        st.markdown(f"""
        <div class="conflict-card">
            <div class="conflict-header">
                ⚠️ VERSION CONFLICT DETECTED
            </div>
            <div style="margin: 0.6rem 0; color: #7F1D1D; font-weight: 500;">
                {rep.summary}
            </div>
            <div class="policy-box-current">
                <strong>Current Policy (Newer):</strong><br>
                <strong>{rep.newer_citation.get('document')}</strong> (v{rep.newer_citation.get('version')}) — <strong>Page {rep.newer_citation.get('page')}</strong><br>
                <em>"{rep.newer_citation.get('statement')}"</em>
            </div>
            <div class="policy-box-older">
                <strong>Older Conflicting Policy:</strong><br>
                <strong>{rep.older_citation.get('document')}</strong> (v{rep.older_citation.get('version')}) — <strong>Page {rep.older_citation.get('page')}</strong><br>
                <em>"{rep.older_citation.get('statement')}"</em>
            </div>
            <div style="font-size: 0.85rem; color: #991B1B; margin-top: 0.5rem;">
                <em>Notice: The current newer policy was prioritized for the primary answer. The older superseded policy is preserved above for transparency.</em>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Primary Answer text
    st.markdown(f"""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.2rem; font-size: 1.05rem; line-height: 1.6;">
        {latest.get('answer')}
    </div>
    """, unsafe_allow_html=True)
    
    # Citations
    citations = latest.get("citations", [])
    if citations:
        st.markdown("#### Citations")
        cite_cols = st.columns(min(len(citations), 3))
        for idx, cite in enumerate(citations):
            col = cite_cols[idx % len(cite_cols)]
            with col:
                st.markdown(f"""
                <div style="background: white; border: 1px solid #CBD5E1; border-radius: 6px; padding: 0.6rem 0.8rem; margin-bottom: 0.5rem;">
                    <span class="citation-tag">{cite.get('type', 'Source')}</span><br>
                    <strong>{cite.get('document')}</strong><br>
                    <span style="color: #64748B; font-size: 0.9rem;">Version: {cite.get('version')} | Page: {cite.get('page')}</span>
                </div>
                """, unsafe_allow_html=True)

    # Retrieved Chunks Accordion
    with st.expander("🔍 View Retrieved Context Chunks (Semantic Matches)", expanded=False):
        for idx, ch in enumerate(latest.get("retrieved_chunks", [])):
            st.markdown(f"""
            <div class="chunk-card">
                <strong>Chunk #{idx+1}</strong> — Document: <code>{ch.get('document')}</code> | Version: <strong>{ch.get('version')}</strong> | Page: <strong>{ch.get('page')}</strong> | Similarity: <code>{ch.get('score')}</code>
                <div style="margin-top: 0.4rem; color: #334155;">
                    {ch.get('text')}
                </div>
            </div>
            """, unsafe_allow_html=True)
