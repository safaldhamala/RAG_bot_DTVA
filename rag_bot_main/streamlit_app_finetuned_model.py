# streamlit_app_v3.py (Final "Generative Agent" Architecture)
import streamlit as st
import os
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# --- Page Configuration ---
st.set_page_config(page_title="DTVA Generative Agent", page_icon="🤖")

# --- Load Environment & Constants ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY for embeddings not found for RAG grounding.")
    st.stop()

# --- IMPORTANT: Point to your best fine-tuned model ---
FINETUNED_MODEL_PATH = "../finetuning/dtva-expert-model" # Assuming you've run the V2 training
APP_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_SAVE_DIR = os.path.join(APP_DIR, "../rag_bot/final_index") # Correct path to the RAG index
PAPER_FAISS_PATH = os.path.join(INDEX_SAVE_DIR, "paper_faiss_index")

# --- Component Loading (Cached) ---
@st.cache_resource
def load_components():
    """Loads the fine-tuned model and the RAG retriever."""
    components = {}
    st.write(f"Loading fine-tuned model from: {FINETUNED_MODEL_PATH}")
    try:
        # Load the fine-tuned model pipeline
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL_PATH, quantization_config=bnb_config, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
        tokenizer.pad_token = tokenizer.eos_token
        components["llm_pipeline"] = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, temperature=0.1)
        components["tokenizer"] = tokenizer
        st.write("✅ Fine-tuned model pipeline loaded.")

        # Load the RAG paper retriever for grounding
        if os.path.exists(PAPER_FAISS_PATH):
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
            db = FAISS.load_local(PAPER_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
            components["paper_retriever"] = db.as_retriever(search_kwargs={'k': 3})
            st.write("✅ RAG paper retriever loaded for grounding.")
        else:
            st.warning(f"Paper index not found at {PAPER_FAISS_PATH}. Grounding will be disabled.")
            components["paper_retriever"] = None

        return components
    except Exception as e:
        st.error(f"Error loading components: {e}")
        st.exception(e)
        st.stop()

# --- Initialize Components and Memory ---
components = load_components()
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am a fine-tuned expert on the DTVA paper. Ask me anything about it."}]

# --- Main UI and Logic ---
st.title("🤖 DTVA Generative Agent")
st.caption("This agent uses its fine-tuned knowledge first, then grounds its answers with RAG.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Ask a question about the paper..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            # --- STEP 1: GET THE EXPERT ANSWER (NO RAG) ---
            # Format the prompt for the fine-tuned model
            tokenizer = components["tokenizer"]
            expert_prompt = tokenizer.apply_chat_template([{"role": "user", "content": user_prompt}], tokenize=False, add_generation_prompt=True)
            
            # Get the initial answer from the model's internal knowledge
            expert_output = components["llm_pipeline"](expert_prompt)[0]['generated_text']
            expert_answer = expert_output.split("[/INST]")[-1].strip()

        with st.spinner("Finding sources to ground the answer..."):
            # --- STEP 2: GROUND THE ANSWER WITH RAG ---
            final_answer = expert_answer # Default to the expert answer
            retrieved_docs = []
            if components.get("paper_retriever"):
                # A more powerful query for the retriever: the question + the expert's answer
                retrieval_query = f"{user_prompt}\n{expert_answer}"
                retrieved_docs = components["paper_retriever"].get_relevant_documents(retrieval_query)
                rag_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

                # --- STEP 3: SYNTHESIZE A FINAL, GROUNDED ANSWER ---
                grounding_prompt_template = f"""You are a helpful AI assistant. You have provided an initial answer to a user's question. Now, refine your answer using the provided context from the source document to add more detail, specifics, and citations. Ensure your final answer is comprehensive and directly addresses the user's original question.

Original Question: {user_prompt}

Your Initial Answer: {expert_answer}

Supporting Context from Document:
---
{rag_context}
---

Final, Refined Answer:"""
                
                final_prompt = tokenizer.apply_chat_template([{"role": "user", "content": grounding_prompt_template}], tokenize=False, add_generation_prompt=True)
                final_output = components["llm_pipeline"](final_prompt)[0]['generated_text']
                final_answer = final_output.split("[/INST]")[-1].strip()

        # --- Display logic ---
        full_response_for_ui = final_answer
        with st.expander("Show Retrieval Debugger", expanded=False):
            st.markdown("##### Initial Expert Answer (No RAG):")
            st.info(expert_answer)
            st.markdown("##### Supporting Context Retrieved for Grounding:")
            if retrieved_docs:
                st.json([{"source": doc.metadata.get('source'), "page": doc.metadata.get('page'), "content_snippet": doc.page_content[:250]+"..."} for doc in retrieved_docs], expanded=False)
            else:
                st.warning("No context was retrieved.")

        message_placeholder.markdown(full_response_for_ui)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response_for_ui})