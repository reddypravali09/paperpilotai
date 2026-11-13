"""
Embedding Generator
====================
Initializes and caches the HuggingFace sentence-transformer
embedding model. Runs locally on CPU — no API calls, completely free.

Model: all-MiniLM-L6-v2 (~90MB, 256-token context, 384-dim output)
"""

from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL, EMBEDDING_DEVICE
from utils.helpers import setup_logger

logger = setup_logger(__name__)

# Module-level singleton — initialized once, reused everywhere
_embeddings_instance = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Get or create the HuggingFace embedding function.

    Uses a module-level singleton to avoid re-loading the model.
    First call downloads ~90MB model (cached at ~/.cache/huggingface/).

    Returns:
        Configured HuggingFaceEmbeddings instance.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info("Embedding model loaded successfully")
    return _embeddings_instance
