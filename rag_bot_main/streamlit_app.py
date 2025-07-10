# streamlit_app.py (Final Tool-Using Agent)
import streamlit as st
import os
import pickle
import json

# --- Core LangChain and AI components ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# --- Page Configuration ---
st.set_page_config(page_title="Tool-Using RAG Agent", page_icon="🛠️", layout="wide")

# --- Load Secrets ---
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("OPENAI_API_KEY not found in secrets.")
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
    components = {}
    st.write("Loading RAG components...")
    try:
        # LLMs
        components["llm"] = ChatOpenAI(model="gpt-4o", temperature=0.1, openai_api_key=openai_api_key)
        components["tool_chooser_llm"] = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key, model_kwargs={"response_format": {"type": "json_object"}})
        
        # Retrievers
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
        db = FAISS.load_local(PAPER_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        components["paper_retriever"] = db.as_retriever(search_kwargs={'k': 7})

        with open(CODE_BM25_PATH, "rb") as f:
            code_docs = pickle.load(f)
        components["code_retriever"] = BM25Retriever.from_documents(code_docs)
        components["code_retriever"].k = 5
        
        return components
    except Exception as e:
        st.error(f"Fatal Error: Could not load RAG components. Details: {e}")
        st.stop()

# --- Agent Logic ---
def choose_tool(user_prompt, chat_history, llm):
    """Uses an LLM to decide which tool to use."""
    prompt = f"""You are an expert routing agent. Your job is to choose the best tool to answer the user's question based on the CHAT HISTORY and the LATEST QUESTION.

You have three tools available:
1. `paper_search`: Use this for questions about concepts, results, definitions, and high-level ideas described in the research paper.
2. `code_search`: Use this for questions about specific Python functions, classes, implementation details, or to see code snippets.
3. `general_conversation`: Use this for greetings, follow-ups that don't require new information, or if the question is ambiguous.

Based on the conversation, which tool is most appropriate?

---
CHAT HISTORY:
{chat_history}
---
LATEST QUESTION: "{user_prompt}"
---

Respond with a JSON object containing two keys: "tool" (the name of the tool) and "query" (a rephrased, self-contained query for the chosen tool).
Example: {{"tool": "code_search", "query": "find the Python function for R_syn reward calculation"}}
"""
    response = llm.invoke(prompt)
    try:
        decision = json.loads(response.content.strip())
        return decision.get("tool", "general_conversation"), decision.get("query", user_prompt)
    except (json.JSONDecodeError, TypeError):
        return "general_conversation", user_prompt

# --- UI & Main App Logic ---
components = load_components()
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am a tool-using agent for your documents. How can I help?"}]

st.title("🛠️ Tool-Using RAG Agent")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(str(message["content"]))

if user_prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # --- 1. CHOOSE THE TOOL ---
            history_window = st.session_state.messages[-(CONVERSATION_HISTORY_TURNS * 2):-1]
            chat_history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history_window])
            
            tool_name, query_for_tool = choose_tool(user_prompt, chat_history_text, components["tool_chooser_llm"])
            st.info(f"Agent decided to use tool: **`{tool_name}`**")

            # --- 2. EXECUTE THE CHOSEN TOOL ---
            rag_context = "No context was retrieved."
            if tool_name == "paper_search":
                retrieved_docs = components["paper_retriever"].get_relevant_documents(query_for_tool)
                rag_context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            elif tool_name == "code_search":
                retrieved_docs = components["code_retriever"].get_relevant_documents(query_for_tool)
                rag_context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            elif tool_name == "general_conversation":
                rag_context = "No retrieval needed for this conversational turn."

            # --- 3. GENERATE FINAL ANSWER ---
            final_answer_prompt = f"""You are a world-class AI assistant. Your goal is to answer the user's question.
- If a specific context is provided, base your answer *exclusively* on that context.
- If no context is provided, answer the question based on the conversation history.

CHAT HISTORY:
{chat_history_text}
---
CONTEXT FROM THE CHOSEN TOOL ('{tool_name}'):
{rag_context}
---
User's last question: "{user_prompt}"
"""
            final_answer = components["llm"].invoke(final_answer_prompt).content
            
        # --- 4. DISPLAY RESULTS ---
        st.markdown(final_answer)
        with st.expander("Show Agent's Reasoning"):
            st.info(f"**Tool Selected:** `{tool_name}`")
            st.info(f"**Query Sent to Tool:** `{query_for_tool}`")
            st.text_area("Context Provided to LLM", rag_context, height=300)

    st.session_state.messages.append({"role": "assistant", "content": final_answer})