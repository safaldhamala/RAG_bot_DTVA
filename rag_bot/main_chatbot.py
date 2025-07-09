# rag_bot/main_chatbot.py
# don't need to run this, can run stream_lit.py, dk why its still here, it is the
# old CLI chatbot implementation

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Load Environment Variables ---
# This assumes your .env file is in the rag_bot/ directory alongside this script
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file or environment variables. Please ensure it's set.")

# --- Configuration ---
# current_dir is rag_chatbot/rag_bot/
current_dir = os.path.dirname(os.path.abspath(__file__))

# Name of the FAISS index directory created by ingest.py
vectorstore_filename = "faiss_index_pdf_only"
vectorstore_path = os.path.join(current_dir, vectorstore_filename)

# Embedding model - MUST MATCH the one used in ingest.py
embeddings_model_name = "text-embedding-3-small"

# LLM for generation
llm_model_name = "gpt-4o"

# --- Initialize Embeddings ---
embeddings = OpenAIEmbeddings(model=embeddings_model_name, openai_api_key=openai_api_key)

# --- Load Vector Store ---
if not os.path.exists(vectorstore_path):
    raise FileNotFoundError(
        f"FAISS index not found at {vectorstore_path}. "
        f"Please run the ingest.py script first to create the '{vectorstore_filename}' index."
    )

print(f"Loading vector store from: {vectorstore_path}")
db = FAISS.load_local(
    vectorstore_path,
    embeddings,
    allow_dangerous_deserialization=True  # Required for FAISS with Langchain
)
retriever = db.as_retriever(search_kwargs={'k': 3}) # Retrieve top 3 relevant chunks
print("Vector store loaded successfully.")

# --- Set up LLM ---
llm = ChatOpenAI(model_name=llm_model_name, temperature=0.2, openai_api_key=openai_api_key)
# temperature=0.2 aims for more factual, less creative responses.

# --- Define Prompt Template ---
# Adjusted to reflect that the knowledge base is primarily the research paper.
prompt_template_str = """You are an AI assistant.
Your knowledge base primarily consists of a research paper.
Use the following pieces of context from the paper to answer the question at the end.
If you don't know the answer from the provided context, clearly state that you don't know based on the information available. Do not try to make up an answer.
Aim for concise and helpful responses.

Context:
{context}

Question: {question}

Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template_str)

# --- Create RetrievalQA Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True # Set to True if you want to see which chunks were retrieved
)
print("Chatbot chain initialized.")

# --- Chat Loop ---
if __name__ == "__main__":
    print(f"\nRAG Chatbot is ready to answer questions about '{vectorstore_filename}'.")
    print("Type 'exit' or 'quit' to end the chat.")
    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break

        if not user_query.strip():
            print("Please enter a question.")
            continue

        try:
            print("Processing your question...")
            result = qa_chain.invoke({"query": user_query})

            print("\nAnswer:")
            print(result["result"])

            # Uncomment to see source documents for debugging or insight
            # print("\n--- Source Documents Retrieved ---")
            # for i, doc in enumerate(result["source_documents"]):
            #     print(f"\nSource {i+1}:")
            #     print(f"  Page: {doc.metadata.get('page', 'N/A')}") # PyPDFLoader often adds 'page' metadata
            #     print(f"  Source File: {doc.metadata.get('source', 'Unknown')}")
            #     # print(f"  Content Snippet: {doc.page_content[:250]}...") # Display a snippet of the chunk
            # print("--- End Source Documents ---")

        except Exception as e:
            print(f"An error occurred while processing your question: {e}")
            # You might want to add more specific error handling or logging here