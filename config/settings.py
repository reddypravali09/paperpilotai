"""
PaperPilot AI — Central Configuration
======================================
All configurable constants are defined here.
Model IDs, chunk sizes, retrieval parameters, and paths.

IMPORTANT: Never hardcode model names or API keys in agent
or core modules. Always import from this file.

To verify Groq model availability:
  curl https://api.groq.com/openai/v1/models \
    -H "Authorization: Bearer $GROQ_API_KEY"
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file (if present) — never exposes keys in source
load_dotenv()

# ── Project Paths ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ── Groq API ──────────────────────────────────────────────
# Key loaded from .env — never hardcoded
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Dual-model strategy:
#   - Classifier: fast, cheap model for intent detection
#   - Generator:  capable model for content generation
# Verified active on Groq API (2026-11-12)

CLASSIFIER_MODEL = "openai/gpt-oss-20b"
GENERATION_MODEL = "openai/gpt-oss-120b"

CLASSIFIER_TEMPERATURE = 0.0   # Deterministic classification
GENERATION_TEMPERATURE = 0.2   # Slight creativity for natural responses
LLM_MAX_RETRIES = 2

# ── Embeddings (local, free — no API calls) ──────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"       # No GPU required

# ── Text Chunking ────────────────────────────────────────
# all-MiniLM-L6-v2 has 256-token context window.
# 500 chars ≈ 100-125 tokens → fits within embedding window.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ── ChromaDB (persistent local storage) ──────────────────
CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma_db")
COLLECTION_PREFIX = "paper_"   # Collections: paper_{sha256_hash}

# ── Retrieval ────────────────────────────────────────────
RETRIEVAL_TOP_K = 7            # Default chunks per query
SUMMARY_TOP_K = 10             # More chunks for summaries
SLIDE_TOP_K = 10               # More chunks for slide generation
MIN_RETRIEVAL_CHUNKS = 2       # Below this → insufficient retrieval guard

# ── Slides ───────────────────────────────────────────────
DEFAULT_SLIDE_COUNT = 10
MIN_SLIDE_COUNT = 8
MAX_SLIDE_COUNT = 12
MAX_SLIDE_RETRIES = 2          # JSON validation retry attempts

# ── Conversation Memory ──────────────────────────────────
MAX_CHAT_HISTORY_TURNS = 5     # Last N exchanges for follow-up context

# ── Logging ──────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
