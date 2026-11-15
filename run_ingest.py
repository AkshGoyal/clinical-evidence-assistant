"""
Entry point for the Data Ingestion Pipeline.
This script triggers the process of reading PDF documents, extracting text/tables,
chunking them, and updating the vector database (FAISS).
Run this whenever new documents are added to the 'data/' directory.
"""
from src.data_ingestion.vector_store import create_or_update_vector_store

if __name__ == "__main__":
    create_or_update_vector_store()
