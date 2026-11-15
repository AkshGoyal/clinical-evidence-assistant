"""
Retrieval Logic Module.

This module constructs the advanced retrieval pipeline used to fetch relevant context.
It implements a "Hybrid Search + Reranking" strategy:

1. **Hybrid Search (Ensemble):**
   - **BM25 (Keyword Search):** Good for exact matches, specific names, or rare terms.
   - **FAISS (Semantic Search):** Good for understanding conceptual similarity.
   - These two are weighted and combined (30% BM25, 70% FAISS).

2. **Reranking (Refinement):**
   - The top results (e.g., 20) from the hybrid search are passed to a "Reranker" (FlashRank).
   - The Reranker scores each document based on its relevance to the query.
   - Only the absolute best (top 5) are returned to the LLM.
"""
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import FlashrankRerank

def get_smart_retriever(vector_store, chunks):
    """
    Builds the 'Refinement Layer' using Reranking.
    1. Retrieval: Fetches top 20 docs (Fast/Rough)
    2. Reranking: Selects top 5 docs (Slow/Precise)
    """
    # 1. Base Retrievers (Broad Search)
    # BM25 for keyword matching
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 10
    
    # FAISS for semantic similarity
    faiss = vector_store.as_retriever(search_kwargs={"k": 10})
    
    # Combine them
    ensemble = EnsembleRetriever(
        retrievers=[bm25, faiss],
        weights=[0.3, 0.7] # Give more weight to semantic search
    )
    
    # 2. The Refinement Layer (Re-ranker)
    # Using FlashRank (lightweight, runs locally) to filter noise from the initial retrieval
    compressor = FlashrankRerank(top_n=5)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble
    )
    
    return compression_retriever
