# streamlit_app.py
import streamlit as st
import os
import pickle
import json

# --- Core LangChain and AI components ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# --- Page Configuration ---
st.set_page_config(page_title="Intelligent RAG Agent", page_icon="", layout="wide")

# --- Load Secrets ---
# Best practice for Streamlit deployment
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except FileNotFoundError:
    st.error("Secrets file not found. Please ensure your secrets are configured for deployment.")
    st.stop()
except KeyError:
    st.error("OPENAI_API_KEY not found in secrets. Please add it to your Streamlit secrets.")
    st.stop()

# --- Paths & Config ---
# This structure assumes the script is in a directory and the index is in the same directory.
# This works well for Streamlit Cloud deployment where the repo root is the main directory.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_SAVE_DIR = os.path.join(APP_DIR, "hybrid_intelligent_index")
PAPER_FAISS_PATH = os.path.join(INDEX_SAVE_DIR, "faiss_paper_index")
CODE_BM25_PATH = os.path.join(INDEX_SAVE_DIR, "bm25_code_docs.pkl")
CONVERSATION_HISTORY_TURNS = 3 # Number of past Q&A turns to include in history

# --- Component Loading (Cached for performance) ---
@st.cache_resource
def load_components():
    """Loads all necessary AI components, models, and retrievers."""
    components = {}
    try:
        # LLMs for different tasks
        components["llm"] = ChatOpenAI(model="gpt-4o", temperature=0.1, openai_api_key=openai_api_key)
        components["router_llm"] = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
        components["critique_llm"] = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key, model_kwargs={"response_format": {"type": "json_object"}})

        # Retrievers
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
        db = FAISS.load_local(PAPER_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # Tool 1: Precise Retriever (with Reranker for high accuracy on specific questions)
        precise_retriever = db.as_retriever(search_kwargs={'k': 10})
        cross_encoder = HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=4)
        components["precise_paper_retriever"] = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=precise_retriever)

        # Tool 2: Broad Retriever (for summaries and overviews)
        components["broad_paper_retriever"] = db.as_retriever(search_kwargs={'k': 15})

        # Tool 3: Code Retriever (Keyword-based)
        with open(CODE_BM25_PATH, "rb") as f:
            code_docs = pickle.load(f)
        components["code_retriever"] = BM25Retriever.from_documents(code_docs)
        components["code_retriever"].k = 5
        
        return components
    except Exception as e:
        st.error(f"Fatal Error: Could not load RAG components. Please check index paths and configurations. Details: {e}")
        st.stop()

# --- Agent Logic ---
def route_query(user_prompt, llm):
    """Classifies the user's query to select the appropriate tool."""
    prompt = f"""You are a query routing expert. Classify the user's question as either 'specific' or 'broad'.
- 'specific' questions ask for a number, a definition, a specific detail, or a code implementation.
- 'broad' questions ask for a summary, an overview, the general idea, or a high-level concept.

User Question: "{user_prompt}"
Classification:"""
    response = llm.invoke(prompt)
    classification = response.content.strip().lower()
    # Default to 'specific' if classification is ambiguous
    return "broad" if "broad" in classification else "specific"

# --- Initialization & UI ---
components = load_components()
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am an intelligent agent for your documents. How can I help?"}]

st.title("RAG Assistant for the DTVA Project")
st.caption("An AI assistant with query routing and self-critique capabilities.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Check if the content is the dictionary structure with confidence score
        if isinstance(message["content"], dict) and "answer" in message["content"]:
            st.metric(label="Confidence", value=f"{message['content']['confidence_score']}%", delta=message['content']['justification'], delta_color="off")
            st.markdown(message["content"]["answer"])
        else:
            st.markdown(str(message["content"])) # Ensure content is always a string

if user_prompt := st.chat_input("Ask a question about the documents..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        # Use a single spinner for the whole process for a cleaner look
        with st.spinner("Thinking..."):
            # --- 1. ROUTE THE QUERY ---
            route = route_query(user_prompt, components["router_llm"])

            # --- 2. EXECUTE THE CORRECT TOOL ---
            if route == "specific":
                paper_docs = components["precise_paper_retriever"].get_relevant_documents(user_prompt)
                code_docs = components["code_retriever"].get_relevant_documents(user_prompt)
                answer_prompt_template = "Based *only* on the provided context, provide a precise answer to the user's question."
            else: # route == "broad"
                paper_docs = components["broad_paper_retriever"].get_relevant_documents(user_prompt)
                code_docs = [] # Don't include code in broad summaries
                answer_prompt_template = "Based *only* on the provided document excerpts, provide a high-level summary that answers the user's broad question."

            all_docs = paper_docs + code_docs
            rag_context = "\n\n---\n\n".join([doc.page_content for doc in all_docs])
            
            # --- 3. GENERATE INITIAL ANSWER ---
            # Create a concise history string for context
            history_window = st.session_state.messages[-(CONVERSATION_HISTORY_TURNS * 2):-1]
            chat_history_text = "\n".join([f"{msg['role']}: {str(msg['content'].get('answer', msg['content']))}" for msg in history_window])
            
            final_answer_prompt = f"""You are a world-class AI assistant. Review the chat history and the verified context to answer the user's last question.
{answer_prompt_template}

CHAT HISTORY:
{chat_history_text}
---
VERIFIED CONTEXT FROM DOCUMENTS:
{rag_context}
---
User's last question: "{user_prompt}"
"""
            initial_answer = components["llm"].invoke(final_answer_prompt).content

            # --- 4. SELF-CRITIQUE FOR CONFIDENCE ---
            critique_prompt = f"""You are a strict fact-checker. Evaluate an AI's answer based *only* on the provided context.
- **Task Type:** The AI was answering a '{route}' question.
- **Score (1-100):** How well is the answer supported by the context?
- **Justification:** Briefly explain your reasoning.
Your output MUST be a valid JSON object with "score" and "justification" keys.

CONTEXT PROVIDED TO THE AI:
{rag_context}
---
AI'S GENERATED ANSWER:
"{initial_answer}"
---
JSON:
"""
            critique_response = components["critique_llm"].invoke(critique_prompt)
            try:
                critique_data = json.loads(critique_response.content.strip())
                confidence_score = int(critique_data.get("score", 0))
                justification = str(critique_data.get("justification", "Critique failed."))
            except (json.JSONDecodeError, TypeError, ValueError):
                confidence_score = 0
                justification = "Failed to parse critique."

        # --- 5. DISPLAY FINAL RESULTS ---
        final_answer_data = {"answer": initial_answer, "confidence_score": confidence_score, "justification": justification}
        
        # Display the confidence metric first
        st.metric(label="Confidence", value=f"{confidence_score}%", delta=justification, delta_color="off")
        # Then display the answer
        st.markdown(initial_answer)

    # Store the structured data in session state for consistent display on rerun
    st.session_state.messages.append({"role": "assistant", "content": final_answer_data})

