"""
PDF Processing Module.

This module handles the extraction of text and tables from PDF documents.
It uses `pdfplumber` for extraction and a specialized strategy for tables:
1. Detects tables and their bounding boxes.
2. Uses an LLM (GPT-4o) to "refine" the raw table data into a clean Markdown format,
   using surrounding text as context to generate meaningful headers.
3. Extracts the remaining text while filtering out content that was already inside a table.
"""
import os
import pdfplumber
import pandas as pd
from langchain.schema import Document, SystemMessage, HumanMessage
import re
from src.core.config import OPENAI_MODEL_GENERATION, get_chat_llm

MAX_TABLE_OUTPUT_TOKENS = 4096


def _collapse_whitespace(text):
    """Collapse runs of spaces/tabs (keeps newlines, so Markdown tables survive)."""
    return re.sub(r"[ \t]{3,}", "  ", text).strip()

# --- LLM Table Refiner ---
def refine_table_with_llm(raw_table_data, context_text, filename, page_num):
    """
    Uses GPT-4o to clean the table and generate a descriptive header based on context.
    
    Why: Raw PDF table extraction is often messy (merged cells, missing headers).
    The LLM uses the surrounding text to understand what the table *should* look like.
    """
    # max_tokens caps runaway/degenerate output (one table in the original index
    # came back as ~2 MB of whitespace from a repetition loop).
    llm = get_chat_llm(OPENAI_MODEL_GENERATION, temperature=0, max_tokens=MAX_TABLE_OUTPUT_TOKENS)
    
    # Convert list-of-lists to a raw string representation for the LLM
    raw_table_str = str(raw_table_data)
    
    prompt_content = f"""
    You are a Data Archiving Specialist. Your task is to reconstruct a table from a PDF and explain it.
    
    ### INPUT DATA
    **Source File:** {filename} (Page {page_num})
    **Surrounding Text (Context):** "{context_text[-500:]}" (The text appearing just before the table)
    **Raw Table Data:** {raw_table_str}
    
    ### INSTRUCTIONS
    1. **Analyze the Context:** Read the surrounding text to understand what this table represents (e.g., "Table 3: Efficacy Results").
    2. **Reconstruct the Table:** Create a clean, well-formatted Markdown table from the Raw Table Data. Fix any merged cells or misalignment.
    3. **Output Format:**
       - Start with a **Descriptive Header** in bold (e.g., "**Table 3: Comparison of Rotavirus Vaccine Efficacy**").
       - Follow with a brief 1-sentence summary of what the data shows.
       - Then provide the **Markdown Table**.
       
    ### OUTPUT:
    """
    
    try:
        response = llm.invoke([
            SystemMessage(content="You are an expert at extracting and cleaning data structures from scientific documents."),
            HumanMessage(content=prompt_content)
        ])
        return _collapse_whitespace(response.content)
    except Exception as e:
        print(f"    ! LLM Table Refinement failed: {e}")
        # Fallback to simple Pandas conversion if LLM fails
        try:
            df = pd.DataFrame(raw_table_data[1:], columns=raw_table_data[0])
            return f"**Table (Raw Extraction)**\n\n{df.to_markdown(index=False)}"
        except:
            return str(raw_table_data)

def is_inside_bbox(obj, bboxes):
    """
    Checks if a text object (char/word) is inside any of the table bounding boxes.
    """
    x0, top, x1, bottom = obj["x0"], obj["top"], obj["x1"], obj["bottom"]
    
    for bbox in bboxes:
        # bbox format: (x0, top, x1, bottom)
        bx0, btop, bx1, bbottom = bbox
        
        # Check for overlap (allow for small margin of error)
        if (x0 >= bx0 and x1 <= bx1 and top >= btop and bottom <= bbottom):
            return True
    return False

def extract_text_and_tables(pdf_path):
    """
    Main extraction function.
    Returns a list of LangChain Document objects (some containing text, some containing markdown tables).
    """
    documents = []
    filename = os.path.basename(pdf_path)
    print(f"    -> Opening {filename} with pdfplumber...")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                print("page number ", i+1)
                # --- STEP 1: Identify Tables and Get Bounding Boxes ---
                tables = page.find_tables() # Finds table structures (coordinates)
                # NOTE: do not reuse `i` in an inner loop here — it is the page index
                # and is used below for page-number metadata.
                if tables:
                    print(f"    Found {len(tables)} table(s) on page {i+1}")
                table_bboxes = [t.bbox for t in tables]
                
                # We need the text *before* the table for context.
                # First, extract ALL text to serve as a running context buffer
                full_page_text = page.extract_text() or ""
                
                # --- STEP 2: Extract Clean Tables (The "Good" Data) ---
                if tables:
                    # print(f       Found {len(tables)} table(s) on page {i+1})
                    for table_idx, table in enumerate(tables):
                        # Extract the data from this specific table
                        extracted_data = table.extract()
                        
                        if not extracted_data: continue

                        # Use LLM to refine the table using the page text as context
                        # We pass the full page text as 'preceding context' 
                        # (The LLM is smart enough to find the relevant part near the table)
                        refined_content = refine_table_with_llm(extracted_data, full_page_text, filename, i+1)
                        
                        # Add a strong marker that this is a table for the Embedding Model
                        final_content = (
                            f"*** TABLE DATA ({filename} - Pg {i+1}) ***\n"
                            f"{refined_content}"
                        )
                        
                        table_metadata = {
                            "source": filename, 
                            "page": i + 1, 
                            "type": "table",
                            "table_id": table_idx
                        }
                        documents.append(Document(page_content=final_content, metadata=table_metadata))


                # --- STEP 3: Extract Clean Text (Filtering OUT the Tables) ---
                # We define a filter function that keeps only text NOT inside a table bbox.
                # This prevents duplicating the messy raw table text in the main text chunks.
                def not_inside_tables_filter(obj):
                    return not is_inside_bbox(obj, table_bboxes)

                # Apply the filter and extract text
                try:
                    clean_text = page.filter(not_inside_tables_filter).extract_text() or ""
                except Exception:
                    # Fallback if filter fails
                    clean_text = page.extract_text() or ""

                if clean_text.strip():
                    text_metadata = {"source": filename, "page": i + 1, "type": "text"}
                    documents.append(Document(page_content=clean_text, metadata=text_metadata))

    except Exception as e:
        print(f"    -> ERROR reading PDF {pdf_path}: {e}")
        
    return documents
