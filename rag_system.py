import os
import sys
from typing import List, Optional, Dict, Any, Set
import traceback

from news_fetcher import get_all_news_articles, NewsArticle
from vector_store import VectorStore, documents_from_articles
from llm_model import Gemma3LLM
from config import PINECONE_API_KEY, NVIDIA_NIM_API_KEY

class UKPolicyRAG:
    def __init__(self):
        """Initialize the UK Policy RAG system."""
        print("Checking API keys...")
        # Check for required API keys
        if not PINECONE_API_KEY:
            raise ValueError("Pinecone API key is required. Set it in .env file or environment variables.")
        if not NVIDIA_NIM_API_KEY:
            raise ValueError("NVIDIA NIM API key is required. Set it in .env file or environment variables.")
        
        print(f"Pinecone API key: {PINECONE_API_KEY[:5]}...")
        print(f"NVIDIA NIM API key: {NVIDIA_NIM_API_KEY[:5]}...")
        
        print("Initializing vector store...")
        try:
            # Initialize components
            self.vector_store = VectorStore()
            print("Vector store initialized successfully.")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            traceback.print_exc()
            raise
        
        print("Initializing LLM...")
        try:
            self.llm = Gemma3LLM()
            print("LLM initialized successfully.")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            traceback.print_exc()
            raise
        
        print("UK Policy RAG system initialized successfully.")
    
    def fetch_and_store_articles(self) -> int:
        """
        Step 1: Fetch articles from RSS feeds and store them in the vector store.
        This function deduplicates based on article URLs.
        
        Returns:
            Number of new articles added to the database
        """
        print("Fetching news articles...")
        
        # Get existing article URLs to avoid duplicates
        existing_urls = self.vector_store.get_all_document_urls()
        
        # Fetch new articles, skipping ones we already have
        articles = get_all_news_articles(existing_urls)
        
        if not articles:
            print("No new articles found.")
            return 0
        
        print(f"Converting {len(articles)} new articles to documents...")
        documents = documents_from_articles(articles)
        
        print("Adding documents to vector store...")
        self.vector_store.add_documents(documents)
        
        return len(articles)
    
    def get_topics(self) -> List[str]:
        """
        Step 2: Extract topics from the stored articles.
        
        Returns:
            List of topic strings
        """
        print("Retrieving latest articles for topic extraction...")
        
        # Get latest articles from BBC and Guardian
        bbc_docs = self.vector_store.get_documents_by_source("bbc", limit=7)
        guardian_docs = self.vector_store.get_documents_by_source("guardian", limit=7)
        
        # Combine documents
        sample_docs = bbc_docs + guardian_docs
        
        if not sample_docs:
            return ["No documents available for topic extraction"]
        
        print(f"Extracting topics from {len(sample_docs)} latest documents ({len(bbc_docs)} BBC, {len(guardian_docs)} Guardian)")
        topics = self.llm.extract_topics(sample_docs, max_topics=10)
        
        return topics
    
    def query(self, query_text: str, num_results: int = 3) -> Dict[str, Any]:
        """
        Step 3: Process a user query using the RAG system.
        
        Args:
            query_text: The query to process
            num_results: Number of documents to retrieve
            
        Returns:
            Dict containing the query, sources, and generated response
        """
        print(f"Processing query: {query_text}")
        
        # Retrieve relevant documents
        docs = self.vector_store.similar_search(query_text, k=num_results)
        
        if not docs:
            return {
                "query": query_text,
                "sources": [],
                "response": "No relevant information found. Please try a different query."
            }
        
        # Generate response
        response = self.llm.generate_response(query_text, docs)
        
        # Format source information
        sources = []
        for doc in docs:
            source_info = {
                "title": doc.metadata.get("title", "Unknown Title"),
                "url": doc.metadata.get("url", ""),
                "date": doc.metadata.get("date", ""),
                "source": doc.metadata.get("source", "Unknown")
            }
            sources.append(source_info)
        
        # Return the complete result
        return {
            "query": query_text,
            "sources": sources,
            "response": response
        }
    
    def clear_database(self) -> None:
        """Clear all documents from the vector store."""
        self.vector_store.clear_vector_store()
        print("Vector store cleared successfully.")

def fetch_news():
    """Step 1: Standalone function to fetch and store news articles."""
    try:
        rag = UKPolicyRAG()
        num_articles = rag.fetch_and_store_articles()
        print(f"Added {num_articles} new articles to the vector store.")
        return True
    except Exception as e:
        print(f"Error fetching news: {e}")
        traceback.print_exc()
        return False

def get_available_topics():
    """Step 2: Standalone function to extract and display topics."""
    try:
        rag = UKPolicyRAG()
        topics = rag.get_topics()
        
        print("\n===== AVAILABLE TOPICS =====")
        for i, topic in enumerate(topics):
            print(f"{i+1}. {topic}")
        print("===========================\n")
        
        return topics
    except Exception as e:
        print(f"Error getting topics: {e}")
        traceback.print_exc()
        return []

def answer_query(query_text: str):
    """Step 3: Standalone function to answer a user query."""
    try:
        rag = UKPolicyRAG()
        result = rag.query(query_text)
        
        print("\n" + "=" * 50)
        print(f"Query: {query_text}")
        print("-" * 50)
        print(f"Response: {result['response']}")
        print("-" * 50)
        print("Sources:")
        for i, source in enumerate(result['sources']):
            print(f"{i+1}. {source['title']} ({source['source']}, {source['date']})")
            print(f"   URL: {source['url']}")
        print("=" * 50)
        
        return result
    except Exception as e:
        print(f"Error answering query: {e}")
        traceback.print_exc()
        return None

def main():
    """Main function demonstrating the interactive workflow."""
    print(f"Starting RAG app with Python {sys.version}")
    
    # Parse command line arguments for simple command-line interface
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "fetch":
            # Step 1: Fetch news
            fetch_news()
        
        elif command == "topics":
            # Step 2: Get topics
            get_available_topics()
        
        elif command == "query" and len(sys.argv) > 2:
            # Step 3: Answer a query
            query = " ".join(sys.argv[2:])
            answer_query(query)
        
        else:
            print("Usage:")
            print("  python rag_system.py fetch            # Fetch and store news articles")
            print("  python rag_system.py topics           # List available topics")
            print("  python rag_system.py query <query>    # Answer a specific query")
    
    else:
        print("Usage:")
        print("  python rag_system.py fetch            # Fetch and store news articles")
        print("  python rag_system.py topics           # List available topics")
        print("  python rag_system.py query <query>    # Answer a specific query")

if __name__ == "__main__":
    main() 