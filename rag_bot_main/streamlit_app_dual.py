# dual_llm_agent.py
import streamlit as st
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# --- Page Configuration ---
st.set_page_config(page_title="Dual-LLM Agent", page_icon="🧠->🤖")

# --- Load Environment & Constants ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY not found.")
    st.stop()

FINETUNED_MODEL_PATH = "../finetuning/dtva-expert-model"

# --- Component Loading (Cached) ---
@st.cache_resource
def load_components():
    """Loads the fine-tuned expert (context generator) and the OpenAI LLM (answer synthesizer)."""
    components = {}
    st.write("Loading Dual-LLM components...")
    try:
        # Load our fine-tuned model for context generation
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        expert_model = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL_PATH, quantization_config=bnb_config, device_map="auto")
        expert_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
        expert_tokenizer.pad_token = expert_tokenizer.eos_token
        
        components["context_generator_pipeline"] = pipeline("text-generation", model=expert_model, tokenizer=expert_tokenizer, max_new_tokens=512, temperature=0.2)
        st.write("✅ Fine-tuned Context Generator loaded.")

        # Load the powerful general-purpose model for final answer synthesis
        components["answer_synthesizer_llm"] = ChatOpenAI(model="gpt-4-turbo", temperature=0, openai_api_key=openai_api_key)
        st.write("✅ GPT-4 Answer Synthesizer loaded.")
        
        return components
    except Exception as e:
        st.error(f"Error loading components: {e}")
        st.exception(e)
        st.stop()

# --- Initialization ---
components = load_components()
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am a dual-LLM agent for the DTVA paper. How can I help?"}]

# --- Main UI and Logic ---
st.title("🧠->🤖 Dual-LLM Conversational Agent")
st.caption("Using a fine-tuned expert for context generation and GPT-4 for final reasoning.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Ask a question about the paper..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        context_generator = components["context_generator_pipeline"]
        answer_synthesizer = components["answer_synthesizer_llm"]
        
        # --- PHASE 1: GENERATE EXPERT CONTEXT ---
        with st.spinner("Consulting fine-tuned expert for context..."):
            meta_prompt = f"""You are a specialized expert on a research paper about a 'slice-aware digital-twin virtualization architecture (DTVA)'. A senior AI is about to answer a user's question. Your task is to provide a comprehensive, detailed 'briefing document' containing all the relevant facts, definitions, and relationships from your internal knowledge that the senior AI will need.

User's Question: "{user_prompt}"

BRIEFING DOCUMENT:"""
            
            expert_prompt = context_generator.tokenizer.apply_chat_template([{"role": "user", "content": meta_prompt}], tokenize=False, add_generation_prompt=True)
            expert_output = context_generator(expert_prompt)[0]['generated_text']
            generated_context = expert_output.split("[/INST]")[-1].strip()

        # --- PHASE 2: SYNTHESIZE FINAL ANSWER ---
        with st.spinner("GPT-4 is synthesizing the final answer..."):
            final_prompt = f"""You are a world-class AI assistant. Your task is to answer the user's question in a clear, conversational way.
- Base your answer *exclusively* on the information found in the provided 'EXPERT-GENERATED BRIEFING DOCUMENT'.
- Do not use any other knowledge.
- If the briefing document is empty or doesn't contain the answer, state "My internal expert could not provide the necessary context to answer this question."

EXPERT-GENERATED BRIEFING DOCUMENT:
---
{generated_context}
---

Based on this document, answer the user's question: "{user_prompt}"
"""
            response = answer_synthesizer.invoke(final_prompt)
            final_answer = response.content

        # --- Display the final result and the reasoning process ---
        message_placeholder.markdown(final_answer)
        with st.expander("Show Dual-LLM Reasoning"):
            st.markdown("#### Phase 1: Expert-Generated Briefing Document")
            st.info(generated_context)
            
    st.session_state.messages.append({"role": "assistant", "content": final_answer})