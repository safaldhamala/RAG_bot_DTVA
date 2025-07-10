# streamlit_app.py (Final Version with Code Quoting AND Confidence Score)
import streamlit as st
import os
import pickle
import json

# --- Core LangChain and AI components ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# --- Page Configuration ---
st.set_page_config(page_title="Code-Aware RAG Agent", page_icon="👩‍💻", layout="wide")

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
        # Add the critique LLM back
        components["critique_llm"] = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key, model_kwargs={"response_format": {"type": "json_object"}})
        
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
Example: {{"tool": "code_search", "query": "find Python code for SA-AMPPO agent instantiation"}}
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
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am a code-aware agent for your documents. How can I help?"}]

st.title("👩‍💻 Code-Aware RAG Agent with Confidence Scoring")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict) and "answer" in message["content"]:
            st.metric(label="Confidence", value=f"{message['content']['confidence_score']}%", delta=message['content']['justification'], delta_color="off")
            st.markdown(message["content"]["answer"], unsafe_allow_html=True)
        else:
            st.markdown(str(message["content"]), unsafe_allow_html=True)

if user_prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # --- 1. CHOOSE THE TOOL ---
            history_window = st.session_state.messages[-(CONVERSATION_HISTORY_TURNS * 2):-1]
            chat_history_text = "\n".join([f"{msg['role']}: {msg['content'].get('answer', msg['content']) if isinstance(msg['content'], dict) else msg['content']}" for msg in history_window])
            
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

            # --- 3. GENERATE INITIAL ANSWER ---
            answer_instruction = ""
            if tool_name == 'code_search':
                answer_instruction = """First, provide a conceptual explanation based on the context.
Then, find the single most relevant code snippet from the context that demonstrates this concept and present it in a formatted markdown block under a 'Relevant Code Snippet:' heading."""
            else:
                answer_instruction = "Provide a comprehensive answer based on the provided context and conversation history."

            final_answer_prompt = f"""You are a world-class AI assistant and expert programmer. Your goal is to answer the user's question based on the provided information.
**INSTRUCTIONS:**
{answer_instruction}

---
CHAT HISTORY:
{chat_history_text}
---
CONTEXT FROM THE CHOSEN TOOL ('{tool_name}'):
{rag_context}
---
User's last question: "{user_prompt}"
**Final Answer:**
"""
            initial_answer = components["llm"].invoke(final_answer_prompt).content
            
            # --- 4. SELF-CRITIQUE FOR CONFIDENCE ---
            critique_prompt = f"""You are a strict fact-checker. Evaluate an AI's answer based *only* on the provided context.
- **Task Type:** The AI was answering a '{tool_name}' question.
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

        # --- 5. DISPLAY RESULTS ---
        final_answer_data = {"answer": initial_answer, "confidence_score": confidence_score, "justification": justification}
        
        st.metric(label="Confidence", value=f"{confidence_score}%", delta=justification, delta_color="off")
        st.markdown(initial_answer, unsafe_allow_html=True)
        
        with st.expander("Show Agent's Reasoning"):
            st.info(f"**Tool Selected:** `{tool_name}`")
            st.info(f"**Query Sent to Tool:** `{query_for_tool}`")
            st.text_area("Context Provided to LLM", rag_context, height=300)

    # Store the structured data in session state for consistent display
    st.session_state.messages.append({"role": "assistant", "content": final_answer_data})