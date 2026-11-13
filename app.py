
import streamlit as st

from config.settings import (
    GROQ_API_KEY,
    DEFAULT_SLIDE_COUNT,
    MIN_SLIDE_COUNT,
    MAX_SLIDE_COUNT,
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

from langchain_core.messages import HumanMessage, AIMessage


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="PaperPilot AI",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PaperPilot AI")
st.subheader("Your AI Research Paper Assistant")

st.write(
    "Upload a research paper and let PaperPilot AI "
    "summarize it, explain key concepts, generate presentations, "
    "and prepare you for your viva."
)

st.divider()


# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------

session_defaults = {
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
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------
# AGENT INITIALIZATION
# ---------------------------------------------------------

@st.cache_resource
def load_agents():
    """Initialize all agents once and reuse them."""

    return {
        "summarize": create_summarizer(),
        "concepts": create_concept_extractor(),
        "slides": create_slide_generator(),
        "viva": create_viva_agent(),
        "general": create_general_agent(),
    }


# ---------------------------------------------------------
# PDF UPLOAD AND PROCESSING
# ---------------------------------------------------------

st.header("1. Upload Research Paper")

uploaded_file = st.file_uploader(
    "Choose a research paper",
    type=["pdf"],
    help="Upload a text-based research paper in PDF format.",
)

if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()
    current_hash = get_paper_hash(pdf_bytes)

    st.info(f"Selected file: {uploaded_file.name}")

    if st.session_state.paper_hash != current_hash:

        if st.button(
            "🚀 Process Research Paper",
            type="primary",
        ):

            try:
                with st.spinner(
                    "Extracting text, creating chunks and storing paper..."
                ):

                    documents = extract_text_from_pdf(pdf_bytes)

                    chunks = chunk_documents(
                        documents,
                        paper_hash=current_hash,
                    )

                    if not chunks:
                        st.error(
                            "No text chunks could be generated from this PDF."
                        )
                        st.stop()

                    if collection_exists(current_hash):

                        st.info(
                            "This paper already exists in ChromaDB. "
                            "Reusing its collection."
                        )

                        total_chunks = len(chunks)

                    else:

                        total_chunks = add_documents(
                            current_hash,
                            chunks,
                        )

                    # Save paper information
                    st.session_state.paper_hash = current_hash
                    st.session_state.paper_name = uploaded_file.name
                    st.session_state.page_count = len(documents)
                    st.session_state.chunk_count = total_chunks
                    st.session_state.paper_documents = documents

                    # Clear old generated results
                    st.session_state.summary_result = None
                    st.session_state.concept_result = None
                    st.session_state.slide_result = None
                    st.session_state.pptx_data = None
                    st.session_state.viva_result = None
                    st.session_state.chat_messages = []

                st.success("Research paper processed successfully!")
                st.rerun()

            except ValueError as e:
                st.error(f"PDF processing error: {e}")

            except Exception as e:
                st.error(f"Something went wrong: {e}")

    else:
        st.success("This paper is already loaded in your session.")


# ---------------------------------------------------------
# PAPER INFORMATION
# ---------------------------------------------------------

if st.session_state.paper_hash:

    st.divider()
    st.header("2. Paper Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Pages Extracted",
            st.session_state.page_count,
        )

    with col2:
        st.metric(
            "Text Chunks",
            st.session_state.chunk_count,
        )

    with col3:
        st.metric(
            "Database Status",
            "Ready",
        )

    st.write(
        "**Uploaded Paper:**",
        st.session_state.paper_name,
    )

    with st.expander("📄 Preview Extracted Paper Text"):

        for index, document in enumerate(
            st.session_state.paper_documents
        ):

            st.markdown(
                f"**Page {document.metadata['page']}**"
            )

            st.write(document.page_content)

            if index < len(st.session_state.paper_documents) - 1:
                st.divider()


# ---------------------------------------------------------
# CHECK API KEY BEFORE AI OPERATIONS
# ---------------------------------------------------------

if not GROQ_API_KEY:

    st.divider()

    st.warning(
        "Groq API key not found. Add GROQ_API_KEY to your "
        "project's .env file to enable AI features."
    )

    st.code(
        'GROQ_API_KEY="your_groq_api_key_here"',
        language="text",
    )

    st.stop()


# ---------------------------------------------------------
# INITIALIZE AGENTS
# ---------------------------------------------------------

try:

    agents = load_agents()

except Exception as e:

    st.error(f"Unable to initialize AI agents: {e}")
    st.stop()


# ---------------------------------------------------------
# HELPER FUNCTION FOR DISPLAYING SOURCES
# ---------------------------------------------------------

def display_agent_metadata(result):

    sources = result.get("sources", [])
    chunk_count = result.get("chunk_count", 0)
    verification = result.get("verification", {})

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Retrieved Chunks", chunk_count)

    with col2:
        if sources:
            st.write(
                "**Source pages:**",
                ", ".join(map(str, sources)),
            )
        else:
            st.write("**Source pages:** Not available")

    if result.get("insufficient_retrieval"):
        st.warning(
            "The retrieved paper content may be insufficient "
            "to answer this request reliably."
        )

    if verification:
        st.caption(
            verification.get("status", "Verification unavailable")
        )


# ---------------------------------------------------------
# AI FEATURES
# ---------------------------------------------------------

if st.session_state.paper_hash:

    st.divider()
    st.header("3. AI-Powered Research Tools")

    summary_tab, concepts_tab, slides_tab, viva_tab = st.tabs(
        [
            "📝 Summary",
            "💡 Key Concepts",
            "📊 Presentation",
            "🎓 Viva Preparation",
        ]
    )

    # -----------------------------------------------------
    # SUMMARY AGENT
    # -----------------------------------------------------

    with summary_tab:

        st.subheader("Research Paper Summary")

        st.write(
            "Generate a concise, structured summary "
            "of your uploaded research paper."
        )

        if st.button(
            "Generate Summary",
            key="generate_summary",
            type="primary",
        ):

            try:

                with st.spinner("Summarizing your research paper..."):

                    result = agents["summarize"].run(
                        query=(
                            "Provide a comprehensive structured summary "
                            "of this research paper, covering its "
                            "objective, problem statement, methodology, "
                            "main findings and conclusion."
                        ),
                        paper_hash=st.session_state.paper_hash,
                        needs_retrieval=True,
                    )

                    st.session_state.summary_result = result

            except Exception as e:

                st.error(f"Summary generation failed: {e}")

        if st.session_state.summary_result:

            result = st.session_state.summary_result

            st.markdown(result["response"])

            st.divider()
            st.subheader("Source Information")

            display_agent_metadata(result)


    # -----------------------------------------------------
    # CONCEPT EXTRACTOR AGENT
    # -----------------------------------------------------

    with concepts_tab:

        st.subheader("Important Research Concepts")

        st.write(
            "Extract the important technical concepts, "
            "terms and explanations from your paper."
        )

        if st.button(
            "Extract Key Concepts",
            key="generate_concepts",
            type="primary",
        ):

            try:

                with st.spinner("Extracting important concepts..."):

                    result = agents["concepts"].run(
                        query=(
                            "Identify and explain the most important "
                            "concepts, technical terms, methods and "
                            "ideas discussed in this research paper."
                        ),
                        paper_hash=st.session_state.paper_hash,
                        needs_retrieval=True,
                    )

                    st.session_state.concept_result = result

            except Exception as e:

                st.error(f"Concept extraction failed: {e}")

        if st.session_state.concept_result:

            result = st.session_state.concept_result

            st.markdown(result["response"])

            st.divider()
            st.subheader("Source Information")

            display_agent_metadata(result)


    # -----------------------------------------------------
    # SLIDE GENERATOR AGENT
    # -----------------------------------------------------

    with slides_tab:

        st.subheader("Research Paper to Presentation")

        st.write(
            "Automatically generate presentation slides "
            "from your research paper."
        )

        slide_count = st.slider(
            "Number of content slides",
            min_value=MIN_SLIDE_COUNT,
            max_value=MAX_SLIDE_COUNT,
            value=DEFAULT_SLIDE_COUNT,
        )

        if st.button(
            "Generate Presentation",
            key="generate_slides",
            type="primary",
        ):

            try:

                with st.spinner(
                    "Generating and validating presentation slides..."
                ):

                    result = agents["slides"].run(
                        query=(
                            "Create a professional academic presentation "
                            "covering the research problem, objectives, "
                            "background, methodology, architecture, "
                            "results, findings and conclusion."
                        ),
                        paper_hash=st.session_state.paper_hash,
                        needs_retrieval=True,
                        slide_count=slide_count,
                    )

                    st.session_state.slide_result = result

                    if result.get("slide_data") is not None:

                        pptx_buffer = build_presentation(
                            result["slide_data"]
                        )

                        st.session_state.pptx_data = (
                            pptx_buffer.getvalue()
                        )

                    else:

                        st.session_state.pptx_data = None

            except Exception as e:

                st.error(f"Presentation generation failed: {e}")

        if st.session_state.slide_result:

            result = st.session_state.slide_result

            st.markdown(result["response"])

            st.divider()

            display_agent_metadata(result)

            if st.session_state.pptx_data:

                st.success(
                    "Your PowerPoint presentation is ready!"
                )

                safe_name = (
                    st.session_state.paper_name
                    .rsplit(".", 1)[0]
                    .replace(" ", "_")
                )

                st.download_button(
                    label="⬇️ Download PowerPoint Presentation",
                    data=st.session_state.pptx_data,
                    file_name=f"{safe_name}_presentation.pptx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "presentationml.presentation"
                    ),
                    type="primary",
                )


    # -----------------------------------------------------
    # VIVA AGENT
    # -----------------------------------------------------

    with viva_tab:

        st.subheader("Viva and Examiner Preparation")

        st.write(
            "Generate likely viva questions with model answers "
            "at different difficulty levels."
        )

        if st.button(
            "Generate Viva Questions",
            key="generate_viva",
            type="primary",
        ):

            try:

                with st.spinner(
                    "Preparing likely viva questions and answers..."
                ):

                    result = agents["viva"].run(
                        query=(
                            "Generate likely viva examination questions "
                            "and detailed model answers based on this "
                            "research paper."
                        ),
                        paper_hash=st.session_state.paper_hash,
                        needs_retrieval=True,
                    )

                    st.session_state.viva_result = result

            except Exception as e:

                st.error(f"Viva generation failed: {e}")

        if st.session_state.viva_result:

            result = st.session_state.viva_result

            st.markdown(result["response"])

            st.divider()
            st.subheader("Source Information")

            display_agent_metadata(result)


# ---------------------------------------------------------
# CHAT INTERFACE WITH INTENT ROUTING
# ---------------------------------------------------------

st.divider()
st.header("4. Ask PaperPilot AI")

st.write(
    "Ask questions about your research paper, request explanations, "
    "or have a general conversation with the assistant."
)

if not st.session_state.paper_hash:

    st.info(
        "Upload and process a research paper to enable "
        "paper-specific questions."
    )

# Display previous conversation
for message in st.session_state.chat_messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_query = st.chat_input(
    "Ask something about your research paper..."
)

if user_query:

    if not GROQ_API_KEY:

        st.error("Please configure your Groq API key first.")

    else:

        try:

            # Convert previous chat into LangChain messages
            chat_history = []

            for message in st.session_state.chat_messages[-10:]:

                if message["role"] == "user":

                    chat_history.append(
                        HumanMessage(content=message["content"])
                    )

                else:

                    chat_history.append(
                        AIMessage(content=message["content"])
                    )

            # Route the request to the correct agent
            with st.spinner("PaperPilot AI is thinking..."):

                result = route_query(
                    query=user_query,
                    agents=agents,
                    paper_hash=st.session_state.paper_hash,
                    chat_history=chat_history,
                )

            assistant_response = result.get(
                "response",
                "I couldn't generate a response. Please try again.",
            )

            # Store conversation
            st.session_state.chat_messages.append(
                {
                    "role": "user",
                    "content": user_query,
                }
            )

            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": assistant_response,
                }
            )

            st.rerun()

        except Exception as e:

            st.error(f"Chat error: {e}")


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "PaperPilot AI | AI-Powered Research Paper Analysis "
    "and Viva Preparation"
)
