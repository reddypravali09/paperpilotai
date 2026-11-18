from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from config.settings import (
    GROQ_API_KEY,
    DEFAULT_SLIDE_COUNT,
    MIN_SLIDE_COUNT,
    MAX_SLIDE_COUNT,
    MAX_CHAT_HISTORY_TURNS,
)
from utils.helpers import get_paper_hash
from core.pdf_processor import extract_text_from_pdf
from core.text_chunker import chunk_documents
from core.vector_store import add_documents, collection_exists

from agents.summarizer import create_summarizer
from agents.concept_extractor import create_concept_extractor
from agents.viva_agent import create_viva_agent
from agents.slide_generator import create_slide_generator
from agents.general_agent import create_general_agent
from agents.intent_router import route_query

from presentation.pptx_builder import build_presentation


# =========================================================
# PAGE CONFIG + THEME
# =========================================================

st.set_page_config(
    page_title="PaperPilot AI | Research Workspace",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS_PATH = Path(__file__).parent / "assets" / "style.css"
if CSS_PATH.exists():
    st.markdown(
        f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )

# Small reusable HTML helpers
def section_heading(eyebrow: str, title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="section-heading">
            <div class="eyebrow">{eyebrow}</div>
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_agent_metadata(result: dict):
    sources = result.get("sources", [])
    chunk_count = result.get("chunk_count", 0)
    verification = result.get("verification", {})

    a, b, c = st.columns(3)
    with a:
        st.metric("Retrieved chunks", chunk_count)
    with b:
        st.metric("Source pages", len(sources) if sources else 0)
    with c:
        status = "Verified" if verification.get("verified") else "Review"
        st.metric("Grounding status", status)

    if sources:
        st.caption("Referenced pages: " + ", ".join(map(str, sources)))
    if result.get("insufficient_retrieval"):
        st.warning(
            "The retrieved paper sections may not contain enough information "
            "to answer this reliably. Check the cited pages before relying on it."
        )
    if verification.get("status"):
        st.caption(verification["status"])


def require_paper() -> bool:
    if not st.session_state.paper_hash:
        st.warning("Upload and process a research paper first to use this tool.")
        return False
    return True


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "paper_hash": None,
    "paper_name": None,
    "page_count": 0,
    "chunk_count": 0,
    "paper_documents": [],
    "summary_result": None,
    "concept_result": None,
    "slide_result": None,
    "pptx_data": None,
    "viva_result": None,
    "chat_messages": [],
    "active_page": "Dashboard",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# AGENT INITIALIZATION
# =========================================================

@st.cache_resource
def load_agents():
    return {
        "summarize": create_summarizer(),
        "concepts": create_concept_extractor(),
        "slides": create_slide_generator(),
        "viva": create_viva_agent(),
        "general": create_general_agent(),
    }


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
            <div class="brand-mark">✦</div>
            <div>
                <div class="brand-name">PaperPilot <span>AI</span></div>
                <div class="brand-caption">RESEARCH INTELLIGENCE</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label">WORKSPACE</div>', unsafe_allow_html=True)
    pages = [
        "Dashboard",
        "Paper Summary",
        "Key Concepts",
        "Presentation Studio",
        "Viva Preparation",
        "AI Chat",
    ]
    page = st.radio(
        "Navigation",
        pages,
        index=pages.index(st.session_state.active_page)
        if st.session_state.active_page in pages else 0,
        label_visibility="collapsed",
        key="navigation_radio",
    )
    st.session_state.active_page = page

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">YOUR PAPER</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload research PDF",
        type=["pdf"],
        help="Upload a text-based research paper.",
        key="paper_uploader",
    )

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        current_hash = get_paper_hash(pdf_bytes)

        st.caption(f"Selected: {uploaded_file.name}")

        if st.session_state.paper_hash == current_hash:
            st.success("Paper ready")
        elif st.button("⚡ Process paper", use_container_width=True, type="primary"):
            try:
                with st.spinner("Reading paper and building knowledge index..."):
                    documents = extract_text_from_pdf(pdf_bytes)
                    chunks = chunk_documents(documents, paper_hash=current_hash)

                    if not chunks:
                        st.error("No text chunks could be generated from this PDF.")
                        st.stop()

                    if collection_exists(current_hash):
                        total_chunks = len(chunks)
                    else:
                        total_chunks = add_documents(current_hash, chunks)

                    st.session_state.paper_hash = current_hash
                    st.session_state.paper_name = uploaded_file.name
                    st.session_state.page_count = len(documents)
                    st.session_state.chunk_count = total_chunks
                    st.session_state.paper_documents = documents

                    # Results from a previous paper must not carry over.
                    for key in (
                        "summary_result", "concept_result", "slide_result",
                        "pptx_data", "viva_result"
                    ):
                        st.session_state[key] = None
                    st.session_state.chat_messages = []

                st.success("Paper processed successfully.")
                st.rerun()
            except ValueError as exc:
                st.error(f"PDF processing error: {exc}")
            except Exception as exc:
                st.error(f"Paper processing failed: {exc}")

    if st.session_state.paper_hash:
        st.markdown(
            f"""
            <div class="paper-status">
                <div class="status-dot"></div>
                <div>
                    <div class="status-title">PAPER INDEXED</div>
                    <div class="status-file">{st.session_state.paper_name}</div>
                    <div class="status-meta">{st.session_state.page_count} pages · {st.session_state.chunk_count} chunks</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sidebar-hint">Your research paper will appear here after processing.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-bottom">POWERED BY GROQ · CHROMADB · LANGCHAIN</div>', unsafe_allow_html=True)


# =========================================================
# API KEY + AGENTS
# =========================================================

if not GROQ_API_KEY:
    st.error(
        "Groq API key not found. Add `GROQ_API_KEY=...` to the `.env` file "
        "in the project root, then restart Streamlit."
    )
    st.stop()

try:
    agents = load_agents()
except Exception as exc:
    st.error(f"Unable to initialize AI agents: {exc}")
    st.stop()


# =========================================================
# TOP BAR
# =========================================================

top_left, top_right = st.columns([5, 1])
with top_left:
    st.markdown(
        '<div class="top-kicker">YOUR ACADEMIC COPILOT</div>',
        unsafe_allow_html=True,
    )
with top_right:
    st.markdown(
        '<div class="top-status"><span></span> SYSTEM READY</div>',
        unsafe_allow_html=True,
    )


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":
    section_heading(
        "WORKSPACE OVERVIEW",
        "Your research, accelerated.",
        "Turn dense academic papers into clear insights, presentations and viva preparation.",
    )

    if st.session_state.paper_hash:
        st.markdown(
            f"""
            <div class="hero-card">
                <div class="hero-content">
                    <div class="eyebrow">CURRENT RESEARCH PAPER</div>
                    <div class="hero-title">{st.session_state.paper_name}</div>
                    <div class="hero-copy">Your paper is indexed and ready for AI-powered analysis.</div>
                </div>
                <div class="hero-chip">● READY TO ANALYZE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Pages extracted", st.session_state.page_count)
        with m2:
            st.metric("Knowledge chunks", st.session_state.chunk_count)
        with m3:
            st.metric("AI tools", "5")
    else:
        st.markdown(
            """
            <div class="hero-card hero-empty">
                <div class="hero-content">
                    <div class="eyebrow">WELCOME TO PAPERPILOT</div>
                    <div class="hero-title">A smarter way to study research.</div>
                    <div class="hero-copy">Upload a PDF from the sidebar to build your personal research workspace.</div>
                </div>
                <div class="hero-orb">✦</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="block-title">YOUR AI TOOLKIT</div>', unsafe_allow_html=True)
    cards = [
        ("01", "Paper Summary", "Get the problem, approach, findings and conclusion.", "📝"),
        ("02", "Key Concepts", "Understand the methods, terminology and contributions.", "✧"),
        ("03", "Presentation Studio", "Transform your paper into a downloadable slide deck.", "▤"),
        ("04", "Viva Preparation", "Practice likely examiner questions with model answers.", "🎓"),
        ("05", "AI Chat", "Ask follow-up questions with paper-aware responses.", "⌘"),
    ]
    cols = st.columns(2)
    for index, (number, title, description, symbol) in enumerate(cards):
        with cols[index % 2]:
            st.markdown(
                f"""
                <div class="tool-card">
                    <div class="tool-card-top"><span>{number} / TOOL</span><span class="tool-symbol">{symbol}</span></div>
                    <div class="tool-card-title">{title}</div>
                    <div class="tool-card-copy">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.info("Choose a tool from the sidebar to get started.")


# =========================================================
# SUMMARY
# =========================================================

elif page == "Paper Summary":
    section_heading(
        "AGENT 01 · SUMMARIZER",
        "Paper summary",
        "A structured overview of the research problem, approach and findings.",
    )

    if require_paper():
        if st.button("✦ Generate summary", type="primary"):
            try:
                with st.spinner("Analyzing your paper..."):
                    st.session_state.summary_result = agents["summarize"].run(
                        query=(
                            "Provide a comprehensive structured summary of this research paper, "
                            "covering its objective, problem statement, methodology, main findings "
                            "and conclusion."
                        ),
                        paper_hash=st.session_state.paper_hash,
                        needs_retrieval=True,
                    )
            except Exception as exc:
                st.error(f"Summary generation failed: {exc}")

        if st.session_state.summary_result:
            result = st.session_state.summary_result
            st.markdown('<div class="result-panel">', unsafe_allow_html=True)
            st.markdown(result.get("response", "No response returned."))
            st.markdown('</div>', unsafe_allow_html=True)
            with st.expander("Sources and verification"):
                render_agent_metadata(result)


# =========================================================
# KEY CONCEPTS
# =========================================================

elif page == "Key Concepts":
    section_heading(
        "AGENT 02 · CONCEPT EXTRACTOR",
        "Key concepts",
        "Explore the ideas, techniques and contributions that matter most.",
    )

    if require_paper():
        if st.button("✦ Extract key concepts", type="primary"):
            try:
                with st.spinner("Extracting concepts from the paper..."):
                    st.session_state.concept_result = agents["concepts"].run(
                        query=(
                            "Identify and explain the most important concepts, technical terms, "
                            "methods and ideas discussed in this research paper."
                        ),
                        paper_hash=st.session_state.paper_hash,
                        needs_retrieval=True,
                    )
            except Exception as exc:
                st.error(f"Concept extraction failed: {exc}")

        if st.session_state.concept_result:
            result = st.session_state.concept_result
            st.markdown('<div class="result-panel">', unsafe_allow_html=True)
            st.markdown(result.get("response", "No response returned."))
            st.markdown('</div>', unsafe_allow_html=True)
            with st.expander("Sources and verification"):
                render_agent_metadata(result)


# =========================================================
# PRESENTATION STUDIO
# =========================================================

elif page == "Presentation Studio":
    section_heading(
        "AGENT 03 · SLIDE GENERATOR",
        "Presentation studio",
        "Create a concise academic presentation from your research paper.",
    )

    if require_paper():
        slide_count = st.slider(
            "Number of content slides",
            min_value=MIN_SLIDE_COUNT,
            max_value=MAX_SLIDE_COUNT,
            value=DEFAULT_SLIDE_COUNT,
        )

        if st.button("✦ Generate presentation", type="primary"):
            try:
                with st.spinner("Building and validating your slide deck..."):
                    result = agents["slides"].run(
                        query=(
                            "Create a professional academic presentation covering the research "
                            "problem, objectives, background, methodology, architecture, results, "
                            "findings and conclusion."
                        ),
                        paper_hash=st.session_state.paper_hash,
                        needs_retrieval=True,
                        slide_count=slide_count,
                    )
                    st.session_state.slide_result = result

                    if result.get("slide_data") is not None:
                        pptx_buffer = build_presentation(result["slide_data"])
                        st.session_state.pptx_data = pptx_buffer.getvalue()
                    else:
                        st.session_state.pptx_data = None
            except Exception as exc:
                st.error(f"Presentation generation failed: {exc}")

        if st.session_state.slide_result:
            result = st.session_state.slide_result
            st.markdown('<div class="result-panel">', unsafe_allow_html=True)
            st.markdown(result.get("response", "No slide preview returned."))
            st.markdown('</div>', unsafe_allow_html=True)

            if st.session_state.pptx_data:
                st.markdown('<div class="download-panel">', unsafe_allow_html=True)
                st.markdown("**Your presentation is ready**")
                st.caption("PowerPoint file · Editable .pptx format")
                safe_name = (
                    (st.session_state.paper_name or "research_paper")
                    .rsplit(".", 1)[0]
                    .replace(" ", "_")
                )
                st.download_button(
                    "⬇ Download PowerPoint",
                    data=st.session_state.pptx_data,
                    file_name=f"{safe_name}_presentation.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    type="primary",
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("Sources and verification"):
                render_agent_metadata(result)


# =========================================================
# VIVA PREPARATION
# =========================================================

elif page == "Viva Preparation":
    section_heading(
        "AGENT 04 · VIVA COACH",
        "Viva preparation",
        "Prepare for your examiner with questions at multiple difficulty levels.",
    )

    if require_paper():
        if st.button("✦ Generate viva questions", type="primary"):
            try:
                with st.spinner("Preparing your viva practice set..."):
                    st.session_state.viva_result = agents["viva"].run(
                        query=(
                            "Generate likely viva examination questions and detailed model answers "
                            "based on this research paper."
                        ),
                        paper_hash=st.session_state.paper_hash,
                        needs_retrieval=True,
                    )
            except Exception as exc:
                st.error(f"Viva generation failed: {exc}")

        if st.session_state.viva_result:
            result = st.session_state.viva_result
            st.markdown('<div class="result-panel">', unsafe_allow_html=True)
            st.markdown(result.get("response", "No response returned."))
            st.markdown('</div>', unsafe_allow_html=True)
            with st.expander("Sources and verification"):
                render_agent_metadata(result)


# =========================================================
# AI CHAT + INTENT ROUTING
# =========================================================

elif page == "AI Chat":
    section_heading(
        "AGENT 05 · ROUTED CONVERSATION",
        "Ask PaperPilot",
        "Chat naturally. PaperPilot routes your request to the appropriate agent.",
    )

    if st.session_state.paper_hash:
        st.caption(f"Research context: {st.session_state.paper_name}")
    else:
        st.info("You can chat generally now. Upload and process a paper to ask paper-specific questions.")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask about your paper, or say hello...")
    if query:
        history = []
        previous = st.session_state.chat_messages[-(2 * MAX_CHAT_HISTORY_TURNS):]
        for message in previous:
            if message["role"] == "user":
                history.append(HumanMessage(content=message["content"]))
            else:
                history.append(AIMessage(content=message["content"]))

        st.session_state.chat_messages.append({"role": "user", "content": query})

        try:
            with st.spinner("PaperPilot is thinking..."):
                result = route_query(
                    query=query,
                    agents=agents,
                    paper_hash=st.session_state.paper_hash,
                    chat_history=history,
                )
            answer = result.get("response", "I couldn't generate a response. Please try again.")
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Chat failed: {exc}")
            # Keep the user's message visible if routing failed.
            st.session_state.chat_messages.pop()


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="app-footer">
        <span>PAPERPILOT AI</span>
        <span>Understand more. Present better. Prepare smarter.</span>
    </div>
    """,
    unsafe_allow_html=True,
)
