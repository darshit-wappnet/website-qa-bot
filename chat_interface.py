import os
import streamlit as st
import logging
import torch
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from datetime import datetime
from db_utils import init_db, save_chat, get_chat_history, clear_chat_history

from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings('ignore')

# Initialize database
init_db()

# Logging configuration
logging.basicConfig(
    filename="qa_sources.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Website Q&A Bot"
)

def log_qa_interaction(question, answer, sources):
    """Log Q&A interaction using Python's logging module"""
    # Log the question and answer
    logger.info(f"Question: {question}")
    logger.info(f"Answer: {answer}")
    
    # Log the sources
    logger.info("Sources:")
    for doc in sources:
        source = doc.metadata.get('source', 'Unknown source')
        logger.info(f"- {source}")
    
    # Add a separator
    logger.info("")

def initialize_vector_store():
    """Initialize and return the vector store"""
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url or not qdrant_api_key:
        st.error("QDRANT_URL and QDRANT_API_KEY must be set in .env file")
        st.stop()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )
    
    try:
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        
        return Qdrant(
            client=client,
            collection_name="website_qa",
            embeddings=embeddings
        )
    except Exception as e:
        st.error(f"Error connecting to Qdrant: {str(e)}")
        st.stop()

def setup_rag_chain(vector_store):
    """Set up the RAG chain with the vector store"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables. Please check your .env file.")
        st.stop()
    
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.7
    )
    
    prompt_template = """You are a helpful AI assistant for our website. Your knowledge comes from the website's content that has been provided to you. Your role is to help users find information quickly without them having to search through the website manually.

    Please answer the question based on the website content provided in the context below. If the website content doesn't contain enough information to answer the question, respond with: "I apologize, but the website doesn't contain enough information to answer this question."

    Remember:
    - Only use information from the website content provided
    - Be friendly and professional in your responses
    - Never make up or infer information not present in the website content
    - If information is incomplete or unclear, say so directly

    Website Content Context:
    {context}

    User Question: {question}

    Assistant: """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={
            "prompt": PROMPT,
            "document_variable_name": "context"
        },
        return_source_documents=True
    )

def main():
    st.title("Website Q&A Bot 🤖")
    
    if 'vector_store' not in st.session_state:
        with st.spinner("Loading knowledge base..."):
            st.session_state.vector_store = initialize_vector_store()
            st.session_state.qa_chain = setup_rag_chain(st.session_state.vector_store)
    
    # Input section
    user_question = st.text_input("Ask a question about the website:", key="user_input")
    col1, col2 = st.columns([6, 1])
    with col2:
        submit_button = st.button("Submit")
    
    st.markdown("---")
    
    # Add a clear history button
    if st.sidebar.button("Clear Chat History"):
        clear_chat_history()
        st.experimental_rerun()
    
    if submit_button:
        if not user_question:
            st.warning("Please enter a question.")
            return
        
        with st.spinner("Searching for answer..."):
            try:
                response = st.session_state.qa_chain({"query": user_question})
                answer = response["result"]
                
                # Save to database
                save_chat(user_question, answer)
                
                # Log the Q&A interaction
                log_qa_interaction(user_question, answer, response["source_documents"])
                
                st.experimental_rerun()
            
            except Exception as e:
                logger.error(f"An error occurred: {str(e)}")
                st.error(f"An error occurred: {str(e)}")
    
    # Display conversation history from database
    chat_history = get_chat_history()
    for question, answer, timestamp in chat_history:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"**You:** {question}")
            st.caption(f"Asked on: {timestamp}")
        with col2:
            st.markdown(f"**Assistant:** {answer}")
        st.markdown("---")

if __name__ == "__main__":
    main() 
    