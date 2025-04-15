import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Cohere
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import warnings
warnings.filterwarnings('ignore')

def load_vector_store():
    """Load the existing vector store from ChromaDB."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    return vector_store

def setup_rag_chain(vector_store):
    """Set up the RAG chain for Q&A."""
    llm = Cohere(
        cohere_api_key="UUMhoFnjwYN9Bo31Skj9lgAAxfJCRkhQEp9QovFd",
        model="command",
        temperature=0.6
    )
    
    prompt_template = """You must answer the question strictly based on the provided context from the database. Do not use any external knowledge, your own understanding, or information outside the context. If the context does not contain enough information to answer the question, state clearly that the database lacks the relevant data and do not provide any additional answer.

Context: {context}

Question: {question}

Answer: """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain

def main():
    st.title("Website Q&A")
    
    # Initialize session state for vector store and chain
    if 'vector_store' not in st.session_state:
        st.write("Loading vector store...")
        try:
            st.session_state.vector_store = load_vector_store()
            st.write("Vector store loaded successfully!")
            st.session_state.rag_chain = setup_rag_chain(st.session_state.vector_store)
        except Exception as e:
            st.error(f"Error loading vector store: {e}")
            st.info("Please make sure you have run rag_system.py first to create the vector store.")
            return
    
    # Question input
    question = st.text_input("Ask a question:")
    
    # Submit button
    if st.button("Submit"):
        if question:
            try:
                answer = st.session_state.rag_chain.run(question)
                st.write("Answer:", answer)
            except Exception as e:
                st.error(f"Error processing question: {e}")
        else:
            st.warning("Please enter a question first.")

if __name__ == '__main__':
    main()