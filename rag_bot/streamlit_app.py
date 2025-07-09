import streamlit as st
import os
from dotenv import load_dotenv
import pickle
from typing import List, Dict, Any
import json

# Fix for collections.MutableSet compatibility issue
import collections
import collections.abc
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping

# Now import the packages that might have compatibility issues
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.retrievers import BM25Retriever
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# Load environment variables
load_dotenv()

class RAGBot:
    def __init__(self, model_type="openai", model_name="gpt-4o"):
        """
        Initialize the RAG bot with configurable model types
        
        Args:
            model_type: "openai" or "huggingface"
            model_name: Model name (gpt-4o for OpenAI, HF model path for local)
        """
        self.model_type = model_type
        self.model_name = model_name
        self.embeddings = None
        self.llm = None
        self.prose_retriever = None
        self.code_retriever = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self._initialize_components()
        self._load_indexes()
    
    def _initialize_components(self):
        """Initialize embeddings and LLM based on model type"""
        if self.model_type == "openai":
            self.embeddings = OpenAIEmbeddings()
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=0.7,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        elif self.model_type == "huggingface":
            # Placeholder for future HuggingFace implementation
            from transformers import AutoTokenizer, AutoModelForCausalLM
            # This will be implemented when switching to local model
            raise NotImplementedError("HuggingFace model support coming soon!")
    

    def _load_indexes(self):
        """
        Load the FAISS index for prose (semantic search) and the code chunks 
        for the BM25 retriever (keyword search).
        """
        try:
            # --- Define Paths (CORRECTED) ---
            # Get the directory where the script is running (e.g., 'rag_bot')
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # The 'rag_index' directory is in the same folder as the script.
            index_dir = os.path.join(script_dir, "rag_index")

            prose_index_path = os.path.join(index_dir, "prose_faiss_index")
            code_pkl_path = os.path.join(index_dir, "code_chunks.pkl")

            # --- Load Prose Index (FAISS for Semantic Search) ---
            if os.path.exists(prose_index_path):
                # Assumes self.embeddings has been initialized, e.g.:
                # self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
                prose_vectorstore = FAISS.load_local(
                    prose_index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.prose_retriever = prose_vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                )
                st.success("✅ Prose (semantic) index loaded successfully.")
            else:
                st.error(f"❌ Prose index not found at: {prose_index_path}")
                self.prose_retriever = None

            # --- Load Code Chunks and Initialize BM25 Retriever (Keyword Search) ---
            if os.path.exists(code_pkl_path):
                with open(code_pkl_path, "rb") as f:
                    code_chunks = pickle.load(f)
                
                self.code_retriever = BM25Retriever.from_documents(code_chunks)
                self.code_retriever.k = 3
                st.success("✅ Code (keyword) index loaded successfully.")
            else:
                st.error(f"❌ Code chunks file not found at: {code_pkl_path}")
                self.code_retriever = None

        except Exception as e:
            st.error(f"An error occurred while loading indexes: {str(e)}")
            self.prose_retriever = None
            self.code_retriever = None




    
    def retrieve_relevant_chunks(self, query: str) -> Dict[str, List[Document]]:
        """Retrieve relevant chunks from both indexes"""
        results = {"prose": [], "code": []}
        
        try:
            if self.prose_retriever:
                prose_docs = self.prose_retriever.get_relevant_documents(query)
                results["prose"] = prose_docs
            
            if self.code_retriever:
                code_docs = self.code_retriever.get_relevant_documents(query)
                results["code"] = code_docs
                
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
        
        return results
    
    def combine_context(self, prose_docs: List[Document], code_docs: List[Document]) -> str:
        """Combine retrieved documents into context"""
        context_parts = []
        
        if prose_docs:
            context_parts.append("=== PROSE CONTEXT ===")
            for i, doc in enumerate(prose_docs):
                context_parts.append(f"Prose Document {i+1}:")
                context_parts.append(doc.page_content)
                context_parts.append("")
        
        if code_docs:
            context_parts.append("=== CODE CONTEXT ===")
            for i, doc in enumerate(code_docs):
                context_parts.append(f"Code Document {i+1}:")
                context_parts.append(doc.page_content)
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response using the LLM"""
        if self.model_type == "openai":
            prompt = f"""Based on the following context, please answer the user's question. 
            If the context doesn't contain relevant information, say so clearly.
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:"""
            
            try:
                response = self.llm.invoke(prompt)
                return response.content
            except Exception as e:
                return f"Error generating response: {str(e)}"
        
        elif self.model_type == "huggingface":
            # Placeholder for HuggingFace implementation
            return "HuggingFace model support coming soon!"
    
    def chat(self, query: str) -> Dict[str, Any]:
        """Main chat function that combines retrieval and generation"""
        # Retrieve relevant chunks
        retrieved_docs = self.retrieve_relevant_chunks(query)
        
        # Combine context
        context = self.combine_context(
            retrieved_docs["prose"], 
            retrieved_docs["code"]
        )
        
        # Generate response
        response = self.generate_response(query, context)
        
        return {
            "response": response,
            "retrieved_docs": retrieved_docs,
            "context": context
        }

def main():
    st.set_page_config(
        page_title="RAG Bot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG Bot - Conversational AI")
    st.markdown("Ask questions about your documents and code!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection (future-proofing)
        model_type = st.selectbox(
            "Model Type",
            ["openai", "huggingface"],
            index=0,
            help="Choose between OpenAI API or local HuggingFace model"
        )
        
        if model_type == "openai":
            model_name = st.selectbox(
                "OpenAI Model",
                ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
                index=0
            )
        else:
            model_name = st.text_input(
                "HuggingFace Model",
                value="NextGLab/ORANSight_Mistral_7B_Instruct",
                help="Enter HuggingFace model path"
            )
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.rag_bot = None
            st.rerun()
    
    # Initialize RAG bot
    if "rag_bot" not in st.session_state or st.session_state.get("current_model") != (model_type, model_name):
        try:
            with st.spinner("Initializing RAG Bot..."):
                st.session_state.rag_bot = RAGBot(model_type=model_type, model_name=model_name)
                st.session_state.current_model = (model_type, model_name)
        except Exception as e:
            st.error(f"Failed to initialize RAG Bot: {str(e)}")
            return
    
    # Initialize chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show debug info if available
            if "debug_info" in message:
                with st.expander("🔍 Debug Info - Retrieved Chunks"):
                    debug_info = message["debug_info"]
                    
                    # Prose chunks
                    if debug_info["retrieved_docs"]["prose"]:
                        st.subheader("📄 Prose Chunks")
                        for i, doc in enumerate(debug_info["retrieved_docs"]["prose"]):
                            st.text_area(
                                f"Prose Chunk {i+1}",
                                doc.page_content,
                                height=100,
                                key=f"prose_{message.get('timestamp', 0)}_{i}"
                            )
                    
                    # Code chunks
                    if debug_info["retrieved_docs"]["code"]:
                        st.subheader("💻 Code Chunks")
                        for i, doc in enumerate(debug_info["retrieved_docs"]["code"]):
                            st.code(
                                doc.page_content,
                                language="python",
                                line_numbers=True
                            )
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.rag_bot.chat(prompt)
                    response = result["response"]
                    
                    st.markdown(response)
                    
                    # Add assistant message to chat history with debug info
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "debug_info": result,
                        "timestamp": len(st.session_state.messages)
                    })
                    
                    # Show debug info
                    with st.expander("🔍 Debug Info - Retrieved Chunks"):
                        # Prose chunks
                        if result["retrieved_docs"]["prose"]:
                            st.subheader("📄 Prose Chunks")
                            for i, doc in enumerate(result["retrieved_docs"]["prose"]):
                                st.text_area(
                                    f"Prose Chunk {i+1}",
                                    doc.page_content,
                                    height=100,
                                    key=f"current_prose_{i}"
                                )
                        
                        # Code chunks
                        if result["retrieved_docs"]["code"]:
                            st.subheader("💻 Code Chunks")
                            for i, doc in enumerate(result["retrieved_docs"]["code"]):
                                st.code(
                                    doc.page_content,
                                    language="python",
                                    line_numbers=True
                                )
                
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()