# streamlit_app.py (Enhanced with Streaming and UX Improvements)
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
st.set_page_config(page_title="Intelligent RAG Agent", page_icon="🧠", layout="wide")

# --- Load Secrets ---
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("OPENAI_API_KEY not found in secrets. Please add it to your Streamlit secrets for deployment.")
    st.stop()

# --- Paths & Config ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_SAVE_DIR = os.path.join(APP_DIR, "hybrid_intelligent_index")
PAPER_FAISS_PATH = os.path.join(INDEX_SAVE_DIR, "faiss_paper_index")
CODE_BM25_PATH = os.path.join(INDEX_SAVE_DIR, "bm25_code_docs.pkl")
CONVERSATION_HISTORY_TURNS = 3

# --- Component Loading (Cached for performance) ---
@st.cache_resource
def load_components():
    """Loads all necessary AI components, models, and retrievers."""
    components = {}
    # ... (This function remains exactly the same as your version) ...
    try:
        components["llm"] = ChatOpenAI(model="gpt-4o", temperature=0.1, openai_api_key=openai_api_key, streaming=True) # Added streaming=True
        components["router_llm"] = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
        components["critique_llm"] = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key, model_kwargs={"response_format": {"type": "json_object"}})
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
        db = FAISS.load_local(PAPER_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        precise_retriever = db.as_retriever(search_kwargs={'k': 10})
        cross_encoder = HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=4)
        components["precise_paper_retriever"] = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=precise_retriever)
        components["broad_paper_retriever"] = db.as_retriever(search_kwargs={'k': 15})
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
    # ... (This function remains exactly the same) ...
    prompt = f"""You are a query routing expert. Classify the user's question as either 'specific' or 'broad'.
- 'specific' questions ask for a number, a definition, a specific detail, or a code implementation.
- 'broad' questions ask for a summary, an overview, the general idea, or a high-level concept.
User Question: "{user_prompt}"
Classification:"""
    response = llm.invoke(prompt)
    classification = response.content.strip().lower()
    return "broad" if "broad" in classification else "specific"

# --- UI & Main App Logic ---
components = load_components()
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am an intelligent agent for your documents. How can I help?"}]

st.title("Intelligent RAG Agent")
st.caption("An AI assistant with query routing, self-critique, and streaming responses.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict) and "answer" in message["content"]:
            st.metric(label="Confidence", value=f"{message['content']['confidence_score']}%", delta=message['content']['justification'], delta_color="off")
            st.markdown(message["content"]["answer"])
        else:
            st.markdown(str(message["content"]))

if user_prompt := st.chat_input("Ask a question about the documents..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        # --- 1. ROUTE THE QUERY ---
        with st.spinner("Analyzing question type..."):
            route = route_query(user_prompt, components["router_llm"])
        
        # --- 2. EXECUTE THE CORRECT TOOL ---
        st.info(f"Agent decided this is a **'{route}'** question. Searching documents...")
        if route == "specific":
            paper_docs = components["precise_paper_retriever"].get_relevant_documents(user_prompt)
            code_docs = components["code_retriever"].get_relevant_documents(user_prompt)
            answer_prompt_template = "Based *only* on the provided context, provide a precise answer to the user's question."
        else: # route == "broad"
            paper_docs = components["broad_paper_retriever"].get_relevant_documents(user_prompt)
            code_docs = []
            answer_prompt_template = "Based *only* on the provided document excerpts, provide a high-level summary that answers the user's broad question."

        all_docs = paper_docs + code_docs
        
        # --- Enhancement: Handle No Documents Found ---
        if not all_docs:
            st.warning("I could not find any relevant information in the documents to answer this question.")
            st.stop()

        rag_context = "\n\n---\n\n".join([doc.page_content for doc in all_docs])
        
        # --- 3. GENERATE STREAMING ANSWER ---
        history_window = st.session_state.messages[-(CONVERSATION_HISTORY_TURNS * 2):-1]
        chat_history_text = "\n".join([f"{msg['role']}: {msg['content'].get('answer', msg['content']) if isinstance(msg['content'], dict) else msg['content']}" for msg in history_window])
        
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
        
        # --- Enhancement: Streaming Response ---
        st.info("Generating response...")
        answer_placeholder = st.empty()
        full_response = ""
        for chunk in components["llm"].stream(final_answer_prompt):
            full_response += chunk.content
            answer_placeholder.markdown(full_response + "▌")
        answer_placeholder.markdown(full_response)
        initial_answer = full_response

        # --- 4. SELF-CRITIQUE FOR CONFIDENCE (after the answer is fully generated) ---
        st.info("Evaluating confidence...")
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
JSON:"""
        critique_response = components["critique_llm"].invoke(critique_prompt)
        try:
            critique_data = json.loads(critique_response.content.strip())
            confidence_score = int(critique_data.get("score", 0))
            justification = str(critique_data.get("justification", "Critique failed."))
        except (json.JSONDecodeError, TypeError, ValueError):
            confidence_score = 0
            justification = "Failed to parse critique."

        # --- 5. FINALIZE AND DISPLAY ---
        final_answer_data = {"answer": initial_answer, "confidence_score": confidence_score, "justification": justification}
        
        # We need to clear the placeholder and re-render the final message with the metric
        answer_placeholder.empty()
        st.metric(label="Confidence", value=f"{confidence_score}%", delta=justification, delta_color="off")
        st.markdown(initial_answer)
        
        with st.expander("Show Retrieval Debugger"):
            st.info(f"Query was routed as: **{route}**")
            st.text_area("Context Provided to LLM", rag_context, height=300)

    # Store the structured data in session state for consistent display on rerun
    st.session_state.messages.append({"role": "assistant", "content": final_answer_data})