# Website Q&A Bot 🤖

A Retrieval-Augmented Generation (RAG) based Q&A system that allows users to ask questions about website content. The system uses web scraping to gather content, processes it into a vector store, and provides accurate answers based on the stored knowledge.


## 🛠️ Technology Stack

- **LangChain**: For RAG implementation and chain management
- **Cohere**: LLM for generating responses
- **ChromaDB**: Vector store for document embeddings
- **HuggingFace**: Sentence transformers for embeddings
- **Streamlit**: Web interface
- **Beautiful Soup**: Web scraping
- **Python-dotenv**: Environment variable management

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/website-qa-bot.git
cd website-qa-bot
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory:
```bash
COHERE_API_KEY=your_api_key_here
```

## 💻 Usage

1. First, run the web scraper to gather content:
```bash
python scrape_website.py
```

2. Process the scraped content into the vector store:
```bash
python vector_store.py
```

3. Launch the Q&A interface:
```bash
streamlit run chat_interface.py
```
