# UK Policy News RAG System

A Retrieval Augmented Generation (RAG) system for UK public policy news using Gemma 3, NVIDIA NIM and Pinecone.

## Features

- Fetches articles from BBC, Guardian, and gov.uk RSS feeds
- Processes and stores article content in Pinecone vector database
- Uses Gemma 3 via NVIDIA NIM for natural language generation
- Provides a simple interface for querying the system about UK public policy

## Setup

### Install dependencies
```bash
pip install -r requirements.txt
```
### Deployment

Create a `.streamlit/secrets.toml` file with the following structure:
   ```toml
   [pinecone]
   api_key = "your_pinecone_api_key"
   environment = "your_env_name"
   index_name = "your_index_name"
   
   [nim]
   api_key = "your_nvidia_nim_api_key"
   ```

## Usage

### Streamlit Web Interface

To run the system with the Streamlit web interface:

```
streamlit run app.py
```

This will start a local web server with a user-friendly interface where you can:
1. Ask questions about UK policy
2. View latest topics
3. Fetch and manage news articles


### Command Line Interface

For development and testing, you can run the system from the command line:

```python
python rag_system.py
```

Specific commands:
- `python rag_system.py fetch` - Fetch and store news articles
- `python rag_system.py topics` - List available topics
- `python rag_system.py query <query>` - Answer a specific query

## System Components

- `news_fetcher.py`: Fetches and processes news from RSS feeds
- `vector_store.py`: Handles interactions with Pinecone vector database
- `llm_model.py`: Manages interactions with Gemma 3 via NVIDIA NIM
- `rag_system.py`: Main system that integrates all components
- `config.py`: Configuration variables loaded from environment or Streamlit secrets
- `app.py`: Streamlit web interface
