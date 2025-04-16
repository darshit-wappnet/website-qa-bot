import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import glob
import time
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

def load_documents(data_dir):
    """Load documents from text files."""
    documents = []
    for file_path in glob.glob(os.path.join(data_dir, '*.txt')):
        loader = TextLoader(file_path, encoding='utf-8')
        documents.extend(loader.load())
    return documents

def create_vector_store(documents, force_recreate=False):
    """Create and populate Qdrant vector store."""
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url or not qdrant_api_key:
        print("Error: QDRANT_URL and QDRANT_API_KEY must be set in .env file")
        return None
    
    collection_name = "website_qa"
    
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
    try:
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=300  
        )
        
        batch_size = 500  
        total_batches = len(chunks) // batch_size + (1 if len(chunks) % batch_size != 0 else 0)
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    if i == 0:
                        vector_store = Qdrant.from_documents(
                            documents=batch,
                            embedding=embeddings,
                            url=qdrant_url,
                            api_key=qdrant_api_key,
                            collection_name=collection_name,
                            force_recreate=force_recreate if i == 0 else False,
                            timeout=300  
                        )
                    else:
                        vector_store.add_documents(batch)
                    
                    print(f"Batch {batch_num}/{total_batches} completed successfully")
                    break
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        print(f"Failed to process batch {batch_num} after {max_retries} attempts: {str(e)}")
                        return None
                    print(f"Retry {retry_count}/{max_retries} for batch {batch_num} after error: {str(e)}")
                    time.sleep(5)  
        
        print("Vector store created successfully!")
        return vector_store
        
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        return None

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
    if vector_store:
        print("Vector store created successfully!")

if __name__ == '__main__':
    main()