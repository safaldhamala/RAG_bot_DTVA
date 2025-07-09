import os
import pickle
import random
from dotenv import load_dotenv

# You will need langchain components to load the FAISS index
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def check_ingestion():
    """
    Loads and inspects the data from the RAG ingestion process to verify its integrity.
    """
    print("--- RAG Ingestion Verification Script ---")

    # 1. Load Environment Variables to get the API key for embeddings
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("\n[FATAL ERROR] OPENAI_API_KEY not found in .env file.")
        print("This key is required to load the FAISS vector store.")
        return
    
    # 2. Define Paths based on the structure from ingest.py
    project_root_dir = os.path.dirname(os.path.abspath(__file__))
    index_save_dir = os.path.join(project_root_dir, "rag_index")
    # We will now assume a pickle file exists for prose chunks, similar to code chunks.
    prose_chunks_pickle_path = os.path.join(index_save_dir, "prose_chunks.pkl") 
    code_chunks_pickle_path = os.path.join(index_save_dir, "code_chunks.pkl")

    # --- Verification for Prose Documents (PDF, TXT) ---
    print("\n" + "="*50)
    print("Verifying Prose Chunks (from PDF, TXT)...")
    print("="*50)

    if not os.path.exists(prose_chunks_pickle_path):
        print(f"[ERROR] Prose chunks pickle file not found at the expected location: {prose_chunks_pickle_path}")
        print("Please ensure your ingest.py saves the prose chunks to this file.")
    else:
        try:
            with open(prose_chunks_pickle_path, "rb") as f:
                prose_chunks = pickle.load(f)
            
            print(f"[SUCCESS] Prose chunks file loaded successfully.")
            num_prose_chunks = len(prose_chunks)
            print(f"[INFO] Total number of prose (PDF/TXT) chunks found: {num_prose_chunks}")

            if num_prose_chunks > 0:
                print("\n--- Displaying up to 3 random full prose chunks for verification ---")
                random_samples = random.sample(prose_chunks, min(3, num_prose_chunks))
                for i, chunk in enumerate(random_samples):
                    source_file = chunk.metadata.get('source', 'Unknown source')
                    page_number = chunk.metadata.get('page', 'N/A')
                    
                    print(f"\n--- Sample #{i+1} ---")
                    print(f"  Source File: {os.path.basename(source_file)}")
                    print(f"  Page Number: {page_number}")
                    print(f"  Full Content:\n\n{chunk.page_content}\n")
                    print("-" * 20)
            else:
                print("[WARNING] The prose chunks file is empty. No PDF or TXT files may have been processed.")

        except Exception as e:
            print(f"[FATAL ERROR] An error occurred while loading the prose chunks pickle file: {e}")


    # --- Verification for Code Documents (Python) ---
    print("\n" + "="*50)
    print("Verifying Code Chunks (Keyword Search - Pickle file)...")
    print("="*50)

    if not os.path.exists(code_chunks_pickle_path):
        print(f"[ERROR] Code chunks pickle file not found at the expected location: {code_chunks_pickle_path}")
        print("Please run ingest.py to create the file.")
    else:
        try:
            with open(code_chunks_pickle_path, "rb") as f:
                code_chunks = pickle.load(f)
            
            print(f"[SUCCESS] Code chunks file loaded successfully.")
            num_code_chunks = len(code_chunks)
            print(f"[INFO] Total number of code (.py) chunks found: {num_code_chunks}")

            if num_code_chunks > 0:
                print("\n--- Displaying up to 3 random code chunks for verification ---")
                random_samples = random.sample(code_chunks, min(3, num_code_chunks))
                for i, chunk in enumerate(random_samples):
                    source_file = chunk.metadata.get('source', 'Unknown source')
                    content_snippet = chunk.page_content.strip().replace('\n', ' ')[:150] + "..."
                    print(f"\nSample #{i+1}:")
                    print(f"  Source File: {os.path.basename(source_file)}")
                    print(f"  Content Snippet: \"{content_snippet}\"")
            else:
                print("[WARNING] The code chunks file is empty. No .py files may have been processed.")
        
        except Exception as e:
            print(f"[FATAL ERROR] An error occurred while loading the code chunks pickle file: {e}")

    print("\n--- Verification Complete ---")


if __name__ == "__main__":
    check_ingestion()
