"""
Vector Store Management Module.

This module orchestrates the creation and updating of the FAISS vector database.
It handles:
1. Iterating through PDFs in the data directory.
2. Invoking the extraction logic (text + tables).
3. Splitting text into chunks (preserving table integrity).
4. Sending chunks to the embedding model (via OpenRouter).
5. Saving the index locally.

Includes robust error handling for OpenAI rate limits (exponential backoff).
"""
import os
import time
import pickle
import openai
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.core.config import DATA_DIR, FAISS_INDEX_PATH, CHUNKS_PATH, get_embeddings
from src.data_ingestion.pdf_processor import extract_text_and_tables, _collapse_whitespace

# Hard cap on any single chunk's text (~12k tokens) so one bad chunk can't blow the context window.
MAX_CHUNK_CHARS = 50_000

def upload_batch_with_backoff(vector_store, batch_chunks, embeddings):
    """
    Uploads a batch of documents to the vector store with retry logic.
    Handles RateLimitError (429) by sleeping and retrying.
    Handles BadRequestError (too large) by splitting the batch recursively.
    """
    max_retries = 5
    base_delay = 5 

    for attempt in range(max_retries):
        try:
            if vector_store is None:
                return FAISS.from_documents(batch_chunks, embedding=embeddings)
            else:
                vector_store.add_documents(batch_chunks)
                return vector_store
                
        except openai.RateLimitError:
            wait_time = base_delay * (2 ** attempt) 
            print(f"      [!] Rate limit hit (429). Sleeping for {wait_time} seconds...")
            time.sleep(wait_time)
            
        except openai.BadRequestError as e:
            if 'max_tokens' in str(e) or 'too large' in str(e):
                print(f"      [!] Batch too large. Splitting...")
                mid = len(batch_chunks) // 2
                if mid == 0: raise e 
                vector_store = upload_batch_with_backoff(vector_store, batch_chunks[:mid], embeddings)
                return upload_batch_with_backoff(vector_store, batch_chunks[mid:], embeddings)
            else:
                raise e
        except Exception as e:
            print(f"      [!] Unexpected error: {e}")
            if "429" in str(e): 
                time.sleep(20)
            else:
                raise e
    raise Exception("Max retries exceeded")

def create_or_update_vector_store():
    print("Starting the ingestion process...")
    
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: The '{DATA_DIR}' folder was not found.")
        return

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"ERROR: No PDF files were found inside '{DATA_DIR}'.")
        return

    embeddings = get_embeddings()
    vector_store = None
    all_chunks_accumulator = []
    processed_files = set()

    # Load existing resources if they exist
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        print("Loading existing knowledge base for incremental update...")
        try:
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            with open(CHUNKS_PATH, "rb") as f:
                all_chunks_accumulator = pickle.load(f)
            # Identify which files are already in the index
            processed_files = {doc.metadata.get("source") for doc in all_chunks_accumulator if "source" in doc.metadata}
            print(f"Found {len(processed_files)} already processed documents.")
        except Exception as e:
            print(f"Warning: Could not load existing index: {e}. Starting fresh.")
            vector_store = None
            all_chunks_accumulator = []

    # Filter for only new files
    new_pdf_files = [f for f in pdf_files if f not in processed_files]
    
    if not new_pdf_files:
        print("No new documents found. Knowledge base is up to date.")
        return

    print(f"Detected {len(new_pdf_files)} new document(s) to process.")

    for i, pdf_file in enumerate(new_pdf_files):
        print(f"\n--- Processing new document {i+1}/{len(new_pdf_files)}: {pdf_file} ---")
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        
        # 1. Extract (With Spatial Filtering)
        raw_docs = extract_text_and_tables(pdf_path)
        
        text_docs = [d for d in raw_docs if d.metadata.get('type') == 'text']
        table_docs = [d for d in raw_docs if d.metadata.get('type') == 'table']

        # 2. Split ONLY text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        text_chunks = text_splitter.split_documents(text_docs)
        
        # 3. Merge
        final_chunks = text_chunks + table_docs
        all_chunks_accumulator.extend(final_chunks) 

        print(f"    Generated {len(final_chunks)} chunks (Clean Text: {len(text_chunks)}, Tables: {len(table_docs)})")

        # 4. Upload in Batches
        BATCH_SIZE = 10 
        for j in range(0, len(final_chunks), BATCH_SIZE):
            batch = final_chunks[j : j + BATCH_SIZE]
            print(f"    -> Uploading batch {j//BATCH_SIZE + 1}...")
            vector_store = upload_batch_with_backoff(vector_store, batch, embeddings)
            time.sleep(1) 

    if vector_store:
        vector_store.save_local(FAISS_INDEX_PATH)
        print(f"SUCCESS: FAISS index updated and saved.")

        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(all_chunks_accumulator, f)
        print(f"SUCCESS: Raw chunks updated and saved.")

def load_resources():
    
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        try:
            st.sidebar.error("Resources missing. Please run 'python run_ingest.py' first.")
        except:
            print("Resources missing. Please run 'python run_ingest.py' first.")
        return None, None
    
    try:
        print("Loading knowledge base...")
        embeddings = get_embeddings()  # must match the model the index was built with
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)

        # Repair degenerate chunks in memory (e.g. a table that the LLM padded with
        # ~2 MB of whitespace). Applies to both the BM25 chunk list and the FAISS docstore.
        for doc in list(chunks) + list(vector_store.docstore._dict.values()):
            if len(doc.page_content) > MAX_CHUNK_CHARS:
                doc.page_content = _collapse_whitespace(doc.page_content)[:MAX_CHUNK_CHARS]

        try:
            print("Resources loaded!")
        except:
            print("Resources loaded!")
            
        return vector_store, chunks
    except Exception as e:
        try:
            st.sidebar.error(f"Failed to load: {e}")
        except:
            print(f"Failed to load: {e}")
        return None, None
