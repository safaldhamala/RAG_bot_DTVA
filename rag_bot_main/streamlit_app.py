# streamlit_app.py (Final Router-Based Agent)
import streamlit as st
import os
import pickle
import json


# LangChain and other core imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# --- Page Configuration ---
st.set_page_config(page_title="Intelligent RAG Agent", page_icon="🧠", layout="wide")


openai_api_key = st.secrets["OPENAI_API_KEY"]
if not openai_api_key:
    st.error("OPENAI_API_KEY not found.")
    st.stop()

# --- Paths & Config ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_SAVE_DIR = os.path.join(APP_DIR, "hybrid_intelligent_index")
PAPER_FAISS_PATH = os.path.join(INDEX_SAVE_DIR, "faiss_paper_index")
CODE_BM25_PATH = os.path.join(INDEX_SAVE_DIR, "bm25_code_docs.pkl")
CONVERSATION_HISTORY_TURNS = 3

# --- Component Loading (Cached) ---
@st.cache_resource
def load_components():
    components = {}
    st.write("Loading RAG components...")
    try:
        # LLMs for different tasks
        components["llm"] = ChatOpenAI(model="gpt-4o", temperature=0.1, openai_api_key=openai_api_key)
        components["router_llm"] = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
        components["critique_llm"] = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key, model_kwargs={"response_format": {"type": "json_object"}})
        st.write("✅ LLMs (Main, Router, Critique) loaded.")

        # Retrievers
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
        db = FAISS.load_local(PAPER_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # Tool 1: Precise Retriever (with Reranker)
        precise_retriever = db.as_retriever(search_kwargs={'k': 10})
        cross_encoder = HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=4)
        components["precise_paper_retriever"] = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=precise_retriever)
        st.write("✅ Precise Paper Retriever loaded.")

        # Tool 2: Broad Retriever (for Summarization)
        components["broad_paper_retriever"] = db.as_retriever(search_kwargs={'k': 15})
        st.write("✅ Broad Paper Retriever loaded.")

        # Tool 3: Code Retriever
        with open(CODE_BM25_PATH, "rb") as f:
            code_docs = pickle.load(f)
        components["code_retriever"] = BM25Retriever.from_documents(code_docs)
        components["code_retriever"].k = 5
        st.write("✅ Code Retriever loaded.")
        
        return components
    except Exception as e:
        st.error(f"Error loading components: {e}")
        st.stop()

# --- Agent Logic ---
def route_query(user_prompt, llm):
    """Classifies the user's query as 'specific' or 'broad'."""
    prompt = f"""You are a query routing expert. Classify the user's question as either 'specific' or 'broad'.
- 'specific' questions ask for a number, a definition, a specific detail, or a code implementation.
- 'broad' questions ask for a summary, an overview, the general idea, or a high-level concept.

User Question: "{user_prompt}"
Classification:"""
    response = llm.invoke(prompt)
    classification = response.content.strip().lower()
    return "specific" if "specific" in classification else "broad"

# --- Initialization & UI ---
components = load_components()
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am an intelligent agent for your documents. How can I help?"}]

st.title("🧠 Intelligent RAG Agent")
st.caption("With Query Routing and Self-Critique Confidence.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict):
            st.metric(label="Confidence", value=f"{message['content']['confidence_score']}%", delta=message['content']['justification'], delta_color="off")
            st.markdown(message["content"]["answer"])
        else:
            st.markdown(message["content"])

if user_prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # --- 1. ROUTE THE QUERY ---
        with st.spinner("Analyzing question type..."):
            route = route_query(user_prompt, components["router_llm"])
            st.info(f"Agent decided this is a **'{route}'** question. Using appropriate tool...")

        # --- 2. EXECUTE THE CORRECT TOOL ---
        if route == "specific":
            with st.spinner("Performing high-precision search..."):
                paper_docs = components["precise_paper_retriever"].get_relevant_documents(user_prompt)
                code_docs = components["code_retriever"].get_relevant_documents(user_prompt)
                answer_prompt_template = "Based *only* on the provided context, provide a precise answer to the user's question."
        else: # route == "broad"
            with st.spinner("Gathering broad context for summary..."):
                paper_docs = components["broad_paper_retriever"].get_relevant_documents(user_prompt)
                code_docs = [] # Typically don't summarize code
                answer_prompt_template = "Based *only* on the provided document excerpts, provide a high-level summary that answers the user's broad question."

        all_docs = paper_docs + code_docs
        rag_context = "\n\n---\n\n".join([f"Source: {doc.metadata.get('source', 'N/A')} | Page: {doc.metadata.get('page_number', 'N/A')}\n{doc.page_content}" for doc in all_docs])
        
        # --- 3. GENERATE INITIAL ANSWER ---
        with st.spinner("GPT-4o is generating the initial answer..."):
            history_window = st.session_state.messages[-(CONVERSATION_HISTORY_TURNS * 2):-1]
            chat_history_text = "\n".join([f"{msg['role']}: {str(msg['content'])}" for msg in history_window])
            
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
        with st.spinner("Evaluating confidence..."):
            critique_prompt = f"""You are a strict fact-checker. Evaluate an AI's answer based *only* on the provided context.
- **Task Type:** The AI was answering a '{route}' question.
- **Score (1-100):** How well is the answer supported by the context? For 'specific' questions, score high only if it's a direct fact. For 'broad' questions, score based on how well it summarizes the main points in the context.
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

        # --- 5. DISPLAY RESULTS ---
        final_answer_data = {"answer": initial_answer, "confidence_score": confidence_score, "justification": justification}
        st.metric(label="Confidence", value=f"{confidence_score}%", delta=justification, delta_color="off")
        message_placeholder.markdown(initial_answer)
        with st.expander("Show Retrieval Debugger"):
            st.info(f"Query was routed as: **{route}**")
            st.text_area("Context Provided to LLM", rag_context, height=300)

    st.session_state.messages.append({"role": "assistant", "content": final_answer_data})
