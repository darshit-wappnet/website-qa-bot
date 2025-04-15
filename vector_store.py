import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain_cohere import Cohere
from langchain.llms import Cohere
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import glob
import time

import warnings
warnings.filterwarnings('ignore')

def load_documents(data_dir):
    """Load documents from text files."""
    documents = []
    for file_path in glob.glob(os.path.join(data_dir, '*.txt')):
        loader = TextLoader(file_path, encoding='utf-8')
        documents.extend(loader.load())
    return documents

def create_vector_store(documents, force_recreate=False):
    """Create and populate Chroma vector store."""
    persist_directory = "./chroma_db"
    
    # Check if vector store exists and load it unless force_recreate is True
    if os.path.exists(persist_directory) and not force_recreate:
        print(f"Loading existing vector store from {persist_directory}...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        print("Vector store loaded successfully!")
        return vector_store
    
    # Create new vector store
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    print("Creating vector store...")
    batch_size = 5000
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} ({len(batch)} chunks)...")
        if i == 0:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=persist_directory
            )
        else:
            vector_store.add_documents(
                documents=batch
            )
        print(f"Batch {i//batch_size + 1} completed")
    
    print("Vector store created successfully!")
    return vector_store

def main():
    data_dir = 'scraped_data'
    
    print("Loading documents...")
    documents = load_documents(data_dir)
    if not documents:
        print("No documents found. Please run scrape_website.py first.")
        return
    
    print(f"Loaded {len(documents)} documents")
    print("Creating vector store...")
    vector_store = create_vector_store(documents, force_recreate=False)
    print("Vector store created successfully!")

if __name__ == '__main__':
    main()