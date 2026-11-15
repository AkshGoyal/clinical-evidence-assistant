import os
import sys
from src.data_ingestion.pdf_processor import extract_text_and_tables

# Set path to a sample PDF (grabbing the first one found)
data_dir = "data_temp"
pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]

if not pdf_files:
    print("No PDFs found in data directory.")
    sys.exit()

sample_pdf = os.path.join(data_dir, pdf_files[0]) 
print(f"--- INSPECTING PROCESSING FOR: {sample_pdf} ---")

# Run extraction (this triggers the LLM table refinement)
# Note: This WILL consume a small amount of API credits for the tables in this one PDF.
docs = extract_text_and_tables(sample_pdf)

print(f"\n--- EXTRACTION COMPLETE: Found {len(docs)} chunks ---")

table_count = 0
for doc in docs:
    if doc.metadata.get("type") == "table":
        table_count += 1
        print("="*60)
        print(f"CHUNK TYPE: TABLE (ID: {doc.metadata.get('table_id')})")
        print("="*60)
        print(doc.page_content)
        print("\n" + "="*60 + "\n")

if table_count == 0:
    print("No tables found in this document (or tables were skipped).")
    print("Showing a text chunk example instead:\n")
    for doc in docs:
        if doc.metadata.get("type") == "text":
            print(doc.page_content[:500] + "...\n")
            break
