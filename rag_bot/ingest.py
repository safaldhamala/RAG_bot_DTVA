import os
import pickle
from dotenv import load_dotenv

# LangChain components for document loading, splitting, and storage
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, PythonCodeTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
# This script should be located in a subdirectory (e.g., 'rag_bot')
# This path navigates up one level to the main project directory.
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAG_BOT_DIR_NAME = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
INDEX_SAVE_DIR = os.path.join(PROJECT_ROOT_DIR, RAG_BOT_DIR_NAME, "rag_index")


# Define paths for the separate indexes
PROSE_FAISS_PATH = os.path.join(INDEX_SAVE_DIR, "prose_faiss_index") # For PDF and TXT semantic index
PROSE_CHUNKS_PICKLE_PATH = os.path.join(INDEX_SAVE_DIR, "prose_chunks.pkl") # Path for the prose pickle file
CODE_CHUNKS_PICKLE_PATH = os.path.join(INDEX_SAVE_DIR, "code_chunks.pkl") # For Python code

# --- Main Ingestion Logic ---
def main():
    """
    Main function to discover, load, chunk, and index all relevant project files.
    """
    print("--- Starting Fresh RAG Ingestion Process ---")
    
    # 1. Load Environment Variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file. Please ensure it is set.")
    print("✅ Environment variables loaded.")

    # 2. Discover and Load Documents
    all_docs = []
    
    # Find all relevant files in the root project directory
    files_to_process = [f for f in os.listdir(PROJECT_ROOT_DIR) if f.endswith((".pdf", ".txt", ".py"))]
    
    if not files_to_process:
        print(f"[WARNING] No .pdf, .txt, or .py files found in the root directory: {PROJECT_ROOT_DIR}. Exiting.")
        return
        
    print(f"\n[INFO] Found {len(files_to_process)} files to process in {PROJECT_ROOT_DIR}...")

    for file_name in files_to_process:
        file_path = os.path.join(PROJECT_ROOT_DIR, file_name)
        try:
            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                # Tag documents to identify them later for chunking
                for doc in docs:
                    doc.metadata['file_type'] = 'pdf'
                all_docs.extend(docs)
                print(f"  [SUCCESS] Loaded PDF: {file_name}")

            elif file_name.endswith(".txt"):
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                for doc in docs:
                    doc.metadata['file_type'] = 'txt'
                all_docs.extend(docs)
                print(f"  [SUCCESS] Loaded Text File: {file_name}")

            elif file_name.endswith(".py"):
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                for doc in docs:
                    doc.metadata['file_type'] = 'python'
                all_docs.extend(docs)
                print(f"  [SUCCESS] Loaded Python File: {file_name}")

        except Exception as e:
            print(f"  [ERROR] Failed to load {file_name}: {e}")

    # 3. Separate documents by type for specialized chunking
    prose_docs = [doc for doc in all_docs if doc.metadata['file_type'] in ['pdf', 'txt']]
    code_docs = [doc for doc in all_docs if doc.metadata['file_type'] == 'python']

    # 4. Chunk and Index Prose Documents (PDF, TXT)
    if prose_docs:
        print("\n[INFO] Chunking prose documents (PDF, TXT)...")
        prose_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        prose_chunks = prose_splitter.split_documents(prose_docs)
        print(f"[SUCCESS] Created {len(prose_chunks)} prose chunks.")

        print("[INFO] Creating FAISS index for prose documents...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
        prose_db = FAISS.from_documents(prose_chunks, embeddings)
        
        os.makedirs(PROSE_FAISS_PATH, exist_ok=True)
        prose_db.save_local(PROSE_FAISS_PATH)
        print(f"[SUCCESS] Prose FAISS index saved to: {PROSE_FAISS_PATH}")

        # Save the prose chunks to a pickle file for verification
        print("[INFO] Saving prose chunks for verification...")
        with open(PROSE_CHUNKS_PICKLE_PATH, "wb") as f:
            pickle.dump(prose_chunks, f)
        print(f"[SUCCESS] Prose chunks saved to: {PROSE_CHUNKS_PICKLE_PATH}")

    else:
        print("\n[INFO] No prose documents to index.")

    # 5. Chunk and Save Code Documents (Python)
    if code_docs:
        print("\n[INFO] Chunking code documents (Python)...")
        code_splitter = PythonCodeTextSplitter(chunk_size=1000, chunk_overlap=100)
        code_chunks = code_splitter.split_documents(code_docs)
        print(f"[SUCCESS] Created {len(code_chunks)} code chunks.")

        print("[INFO] Saving code chunks for keyword search...")
        os.makedirs(os.path.dirname(CODE_CHUNKS_PICKLE_PATH), exist_ok=True)
        with open(CODE_CHUNKS_PICKLE_PATH, "wb") as f:
            pickle.dump(code_chunks, f)
        print(f"[SUCCESS] Code chunks saved to: {CODE_CHUNKS_PICKLE_PATH}")
    else:
        print("\n[INFO] No code documents to index.")
        
    print("\n--- Ingestion Process Complete! ---")


if __name__ == "__main__":
    main()
