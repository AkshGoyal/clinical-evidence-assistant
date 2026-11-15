"""
Streamlit UI Module.

This is the main presentation layer of the application.
It handles:
1. Initializing the application state (loading models, memory, etc.).
2. Rendering the chat interface (native st.chat_message bubbles, sidebar).
3. Capturing user input and displaying the conversation.
4. Orchestrating the calls to the backend logic (Intent -> Retrieval -> Generation).

Rendering note: messages are rendered with st.chat_message + st.markdown (no raw HTML).
The original version wrapped Markdown inside an HTML <div>, which stopped the first
paragraph from being parsed (bold showed as **text**) and rendered user input as raw HTML.
The colour theme is fixed in .streamlit/config.toml so the app looks the same in
light and dark OS modes.
"""
import openai
import streamlit as st
from langchain.memory import ConversationSummaryBufferMemory

# Import modular components
from src.core.config import OPENAI_MODEL_INTENT, OPENROUTER_API_KEY, get_chat_llm
from src.data_ingestion.vector_store import load_resources
from src.retrieval.retrievers import get_smart_retriever
from src.core.agent import (
    get_intent_classifier,
    handle_knowledge_query,
    handle_conversational_query,
    get_query_rewriter,
)

AVATARS = {"assistant": "🩺", "user": "🧑"}

CUSTOM_CSS = """
<style>
    /* Keep the conversation at a readable width on wide screens */
    .block-container { max-width: 960px; padding-top: 2rem; }

    /* Markdown tables inside answers */
    [data-testid="stChatMessage"] table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
    [data-testid="stChatMessage"] th, [data-testid="stChatMessage"] td {
        border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left;
    }
    [data-testid="stChatMessage"] th { background-color: #e8f1f8; font-weight: 600; }
    [data-testid="stChatMessage"] tr:nth-child(even) td { background-color: #f7fafc; }

    /* Card look for each message */
    [data-testid="stChatMessage"] {
        background-color: #ffffff; border: 1px solid #e3e8ef; border-radius: 12px;
        padding: 0.9rem 1rem; box-shadow: 0 1px 3px rgba(16, 42, 67, 0.06);
    }
</style>
"""


def _init_state():
    if "vector_store" not in st.session_state:
        vs, chunks = load_resources()
        st.session_state.vector_store = vs
        st.session_state.chunks = chunks

    if "intent_classifier" not in st.session_state:
        st.session_state.intent_classifier = get_intent_classifier()

    if "query_rewriter" not in st.session_state:
        st.session_state.query_rewriter = get_query_rewriter()

    # SMART MEMORY: keeps recent turns verbatim, summarises older ones
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationSummaryBufferMemory(
            llm=get_chat_llm(OPENAI_MODEL_INTENT),  # cheap model for summarisation
            max_token_limit=5000,  # summarise older turns once history exceeds ~5000 tokens
            memory_key="chat_history",
            return_messages=True,
        )

    # Smart Retriever (hybrid search + reranker)
    if "retriever" not in st.session_state and st.session_state.vector_store:
        st.session_state.retriever = get_smart_retriever(
            st.session_state.vector_store, st.session_state.chunks
        )

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I answer pediatric immunization questions from a library of clinical trials, prescribing information and WHO position papers. What would you like to know?",
        }]


def _render_sidebar():
    with st.sidebar:
        st.header("🧠 Advanced Logic")
        st.markdown(
            "**Pipeline active**\n"
            "1. **Query rewriting**\n"
            "2. **Hybrid retrieval** (FAISS + BM25)\n"
            "3. **Re-ranking** (FlashRank, top 5)\n"
            "4. **Dynamic prompting** (adapts to intent)"
        )
        chunks = st.session_state.get("chunks")
        if chunks:
            n_sources = len({c.metadata.get("source") for c in chunks})
            st.caption(f"Knowledge base: {n_sources} documents · {len(chunks):,} chunks")

        # Persisted in session_state so it survives st.rerun()
        last = st.session_state.get("last_rewrite")
        if last:
            st.divider()
            st.markdown("**🔍 Last query optimisation**")
            st.markdown(f"**Original:** {last['original']}")
            st.markdown(f"**Rewritten:** {last['rewritten']}")
            if last.get("intent"):
                st.caption(f"Intent: {last['intent']}")

        if len(st.session_state.get("messages", [])) > 1:
            st.divider()
            if st.button("🗑️ New conversation", use_container_width=True):
                for key in ("messages", "memory", "last_rewrite"):
                    st.session_state.pop(key, None)
                st.rerun()


def run_ui():
    st.set_page_config(page_title="Clinical Evidence Assistant", page_icon="🩺", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if not OPENROUTER_API_KEY:
        st.error("No API key found. Add `OPENROUTER_API_KEY=sk-or-...` to the `.env` file in the project folder, then restart the app.")
        st.stop()

    _init_state()
    _render_sidebar()

    st.title("🩺 Clinical Evidence Assistant")
    st.caption("Pediatric immunization Q&A · every answer grounded in the vaccine evidence library · AI assistant, not medical advice")

    # Conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about a vaccine, a trial or a schedule...")
    if not prompt:
        return

    if not st.session_state.get("retriever"):
        st.error("The knowledge base is not loaded. Run `python run_ingest.py` first.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            # 1. Classify
            with st.spinner("Thinking..."):
                intent_res = st.session_state.intent_classifier.invoke({"user_input": prompt})
                intent = intent_res["text"].strip().lower()

            # 2. Route
            if "conversational" in intent:
                answer = handle_conversational_query(prompt, st.session_state.memory)
            else:
                answer = handle_knowledge_query(
                    prompt, intent, st.session_state.retriever, st.session_state.memory
                )
                if st.session_state.get("last_rewrite"):
                    st.session_state.last_rewrite["intent"] = intent
        except openai.AuthenticationError:
            st.session_state.messages.pop()  # drop the unanswered user message
            st.error("OpenRouter rejected the API key. Put a valid OPENROUTER_API_KEY in .env and restart the app.")
            return
        except openai.PermissionDeniedError:
            st.session_state.messages.pop()
            st.error("OpenRouter refused the request (403). Check the key's limits and model access on openrouter.ai.")
            return
        except openai.APIStatusError as e:
            st.session_state.messages.pop()
            if e.status_code == 402:
                st.error("OpenRouter account is out of credits. Add credits at openrouter.ai/settings/credits.")
            else:
                st.error(f"OpenRouter request failed ({e.status_code}): {e}")
            return
        except openai.OpenAIError as e:
            st.session_state.messages.pop()
            st.error(f"OpenRouter request failed: {e}")
            return

    # 3. Save to memory & history, then rerun so the sidebar reflects the new state
    st.session_state.memory.save_context({"input": prompt}, {"output": answer})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
