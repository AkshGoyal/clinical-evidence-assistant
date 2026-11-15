"""
Core Agent Logic.

This module acts as the "brain" of the application. It orchestrates the interaction
between the user, the LLMs, and the retrieval system.

Key Functions:
- `get_intent_classifier`: Identifies if the user wants data, advice, or small talk.
- `get_query_rewriter`: Reformulates queries to be "search-engine friendly".
- `handle_knowledge_query`: The main RAG pipeline (Rewrite -> Retrieve -> Rerank -> Generate).
- `handle_conversational_query`: A lighter path for simple chitchat.
"""
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import SystemMessage, HumanMessage

from src.core.prompts import (
    INTENT_CLASSIFICATION_PROMPT, 
    PROMPT_IDENTITY, 
    create_chat_prompt,
    QUERY_REWRITE_PROMPT
)
from src.core.config import OPENAI_MODEL_GENERATION, OPENAI_MODEL_INTENT, OPENAI_MODEL_RETRIEVAL, get_chat_llm

@st.cache_resource
def get_intent_classifier():
    """
    Sets up the Intent Classifier Chain.
    Uses a fast model (gpt-4o-mini via OpenRouter) to categorize the user's input.
    """
    model = get_chat_llm(OPENAI_MODEL_INTENT, temperature=0)
    prompt = PromptTemplate(template=INTENT_CLASSIFICATION_PROMPT, input_variables=["user_input"])
    return LLMChain(llm=model, prompt=prompt)

@st.cache_resource
def get_query_rewriter():
    """
    Sets up an LLMChain for query rewriting.
    """
    model = get_chat_llm(OPENAI_MODEL_RETRIEVAL, temperature=0.1) # Use a cheap, fast model
    prompt = PromptTemplate(
        template=QUERY_REWRITE_PROMPT, 
        input_variables=["chat_history", "original_query"]
    )
    return LLMChain(llm=model, prompt=prompt)

def handle_knowledge_query(query, intent, retriever, memory):
    """
    Manual RAG Flow:
    1. Rewrite Query (if enabled)
    2. Retrieve & Rerank Docs
    3. Load History
    4. Build Dynamic Prompt (Structured)
    5. Generate Response
    """
    # B. Load Summarized History
    history_messages = memory.load_memory_variables({})['chat_history']

    # 1. Query Rewriting (if query_rewriter is available in session state)
    rewritten_query = query
    if st.session_state.get("query_rewriter"):
        # The spinner message will be removed from the main UI
        rewrite_res = st.session_state.query_rewriter.invoke({
            "chat_history": history_messages, # Pass the list
            "original_query": query
        })
        rewritten_query = rewrite_res['text'].strip()
        
        # Stored in session_state (not drawn here) so the sidebar can show it
        # after st.rerun(); drawing it directly was wiped by the rerun.
        st.session_state.last_rewrite = {"original": query, "rewritten": rewritten_query}

    # A. Retrieve & Rerank (The Filter Layer)
    with st.spinner("Searching the evidence library..."):
        # This step now includes the Reranker automatically via the compression_retriever
        docs = retriever.invoke(rewritten_query) # Use rewritten query
    
    # Context Construction
    context_text = ""
    for i, doc in enumerate(docs):
        context_text += f"{doc.page_content}\n\n"

    # C. Build Structured Prompt Template
    prompt_template = create_chat_prompt(intent)
    
    # D. Generate Answer
    llm = get_chat_llm(OPENAI_MODEL_GENERATION, temperature=0.3)
    
    # Modern LangChain Chain: Prompt -> LLM
    chain = prompt_template | llm
    
    response = chain.invoke({
        "context": context_text,
        "chat_history": history_messages, # Pass list of messages
        "user_question": query # Original query for the final response
    })
    
    return response.content

def handle_conversational_query(query, memory):
    # Simple Persona for small talk
    model = get_chat_llm(OPENAI_MODEL_GENERATION, temperature=0.2)
    
    messages = [SystemMessage(content=PROMPT_IDENTITY + "\n\nRespond naturally to small talk.")]
    messages.extend(memory.chat_memory.messages) # Add raw history
    messages.append(HumanMessage(content=query))
    
    response = model.invoke(messages)
    return response.content
