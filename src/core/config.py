"""
Configuration settings for the application.
Centralizes file paths, the API provider (OpenRouter) and model definitions,
and exposes the only two factories the rest of the code uses to build models:
`get_chat_llm()` and `get_embeddings()`. Switching provider = editing this file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from the project's .env file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Data Directories
# Where raw PDFs are stored
DATA_DIR = "data"
# Where the vector database and metadata are saved
FAISS_INDEX_PATH = "faiss_index"
CHUNKS_PATH = "faiss_index/chunks.pkl"

# --- API provider: OpenRouter (OpenAI-compatible API) ---
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
APP_TITLE = "Clinical Evidence Assistant"  # shown in the OpenRouter usage dashboard

# Model Configurations (OpenRouter model slugs)
# We use different models for different tasks to balance cost/performance:
# - Retrieval: Used for query rewriting (faster/cheaper)
# - Generation: Used for the final answer + table reconstruction (higher quality/reasoning)
# - Intent: Used for classifying user queries and summarising memory (fast/cheap)
# gpt-3.5-turbo is no longer offered on OpenRouter; gpt-4o-mini replaces it.
OPENAI_MODEL_RETRIEVAL = "openai/gpt-4o-mini"
OPENAI_MODEL_GENERATION = "openai/gpt-4o"
OPENAI_MODEL_INTENT = "openai/gpt-4o-mini"

# Must stay the SAME model the FAISS index was built with, or similarity search breaks.
EMBEDDING_MODEL = "openai/text-embedding-3-large"


def _require_key():
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to the .env file in the project root.")
    return OPENROUTER_API_KEY


def get_chat_llm(model, temperature=None, **kwargs):
    """Chat model routed through OpenRouter."""
    params = dict(
        model=model,
        api_key=_require_key(),
        base_url=OPENROUTER_BASE_URL,
        default_headers={"X-Title": APP_TITLE},
        **kwargs,
    )
    if temperature is not None:
        params["temperature"] = temperature
    return ChatOpenAI(**params)


def get_embeddings():
    """Embedding model routed through OpenRouter."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=_require_key(),
        base_url=OPENROUTER_BASE_URL,
        default_headers={"X-Title": APP_TITLE},
        # LangChain otherwise sends pre-tokenised input (token IDs), which is
        # OpenAI-specific; send plain text instead. Our chunks are well under 8k tokens.
        check_embedding_ctx_length=False,
    )
