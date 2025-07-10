# rag_chatbot_feed_files/rag_bot/ingest.py
import os
import ast
import pickle
from dotenv import load_dotenv
from typing import List, Dict

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- Load Environment Variables ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

# --- Configuration ---
# The directory one level up from this script's location, containing the source files.
SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# The directory where the final index artifacts will be saved.
INDEX_SAVE_DIR = os.path.join(os.path.dirname(__file__), "hybrid_intelligent_index")
os.makedirs(INDEX_SAVE_DIR, exist_ok=True)

# --- Intelligent Parser for Python Files ---
def parse_python_file_ast(file_path: str) -> List[Document]:
    """
    Parses a Python file using its Abstract Syntax Tree (AST) to extract
    semantic chunks for functions and classes. This is superior to simple
    text splitting as it understands the code's structure.
    """
    with open(file_path, "r", encoding="utf-8") as source:
        try:
            tree = ast.parse(source.read())
            source_code = open(file_path, "r", encoding="utf-8").read()
        except Exception as e:
            print(f"  [WARNING] Could not parse AST for {os.path.basename(file_path)}: {e}")
            return []

    documents = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            node_source = ast.get_source_segment(source_code, node)
            if not node_source:
                continue

            node_type = "Function" if isinstance(node, ast.FunctionDef) else "Class"
            node_name = node.name
            docstring = ast.get_docstring(node) or "No docstring provided."

            content = f"""Source File: {os.path.basename(file_path)}
Type: {node_type}
Name: {node_name}
Docstring: {docstring}
---
Code:
{node_source}"""
            
            metadata = {
                "source": file_path,
                "file_type": "python",
                "node_type": node_type.lower(),
                "node_name": node_name,
            }
            documents.append(Document(page_content=content, metadata=metadata))
            
    print(f"  [SUCCESS] Parsed {len(documents)} functions/classes from {os.path.basename(file_path)}")
    return documents

# --- Intelligent Parser for PDF Files ---
def process_pdf_layout_aware(file_path: str) -> List[Document]:
    """
    Processes a PDF using Unstructured to identify layout elements (titles, paragraphs, etc.)
    and implements a "parent document" strategy by storing the full page text in metadata.
    This gives the retriever both precision and broad context.
    """
    print(f"\n[INFO] Parsing PDF with layout-aware engine: {os.path.basename(file_path)}...")
    loader = UnstructuredPDFLoader(file_path, mode="elements")
    elements = loader.load()

    # Group text by page to create the "parent" context for each chunk
    pages_text: Dict[int, str] = {}
    for el in elements:
        page_num = el.metadata.get('page_number', 0)
        pages_text[page_num] = pages_text.get(page_num, "") + "\n\n" + el.page_content

    final_docs = []
    for el in elements:
        page_num = el.metadata.get('page_number', 0)
        chunk_metadata = el.metadata.copy()
        # Add the full page text as parent context
        chunk_metadata["parent_context"] = pages_text.get(page_num, "")
        chunk_metadata["file_type"] = "pdf"
        
        final_docs.append(Document(page_content=el.page_content, metadata=chunk_metadata))
        
    print(f"[SUCCESS] Created {len(final_docs)} layout-aware chunks from PDF.")
    return final_docs

# --- Main Ingestion Execution ---
if __name__ == "__main__":
    print("--- Starting Intelligent Hybrid Ingestion ---")
    print(f"Reading source files from: {SOURCE_DIR}")
    print(f"Saving index artifacts to: {INDEX_SAVE_DIR}")

    pdf_documents = []
    code_documents = []

    # Find and process all supported files in the source directory
    for filename in os.listdir(SOURCE_DIR):
        file_path = os.path.join(SOURCE_DIR, filename)
        if filename.endswith(".pdf"):
            pdf_docs = process_pdf_layout_aware(file_path)
            pdf_documents.extend(pdf_docs)
        elif filename.endswith(".py"):
            print(f"\n[INFO] Parsing Python file: {filename}...")
            code_docs = parse_python_file_ast(file_path)
            code_documents.extend(code_docs)

    if not pdf_documents and not code_documents:
        raise ValueError("FATAL ERROR: No processable PDF or Python documents were found in the source directory.")

    # --- Store Artifacts for Hybrid Retrieval ---

    # 1. Create and save a FAISS vector store for the PDF documents (for semantic search)
    if pdf_documents:
        print("\n[INFO] Creating FAISS vector store for PDF documents...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
        pdf_db = FAISS.from_documents(pdf_documents, embeddings)
        
        faiss_save_path = os.path.join(INDEX_SAVE_DIR, "faiss_paper_index")
        pdf_db.save_local(folder_path=faiss_save_path)
        print(f"[SUCCESS] Paper FAISS index saved to: {faiss_save_path}")
    else:
        print("[WARNING] No PDF documents found. Skipping FAISS index creation.")

    # 2. Save the raw code documents to a pickle file (for BM25 keyword search)
    if code_documents:
        print("\n[INFO] Saving raw code document chunks for BM25 retriever...")
        code_pickle_path = os.path.join(INDEX_SAVE_DIR, "bm25_code_docs.pkl")
        with open(code_pickle_path, "wb") as f:
            pickle.dump(code_documents, f)
        print(f"[SUCCESS] Code document chunks saved to: {code_pickle_path}")
    else:
        print("[WARNING] No code documents found. Skipping BM25 file creation.")

    print("\n--- Intelligent Ingestion Complete! ---")