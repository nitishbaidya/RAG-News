from typing import List, Dict, Any, Optional, Set
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.document import Document
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME
from datetime import datetime
import os

class VectorStore:
    def __init__(self):
        """Initialize the vector store connection."""
        self.pinecone_api_key = PINECONE_API_KEY
        self.pinecone_environment = PINECONE_ENVIRONMENT
        self.index_name = PINECONE_INDEX_NAME
        
        # Initialize Pinecone with the new client format
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Check if the index exists by listing all indexes
        try:
            index_list = self.pc.list_indexes()
            available_indexes = [idx["name"] for idx in index_list]
            
            if self.index_name not in available_indexes:
                print(f"Warning: Index '{self.index_name}' not found in Pinecone. Please ensure it's created.")
        except Exception as e:
            print(f"Error checking indexes: {e}")
            
        # Initialize embedding model (using MPNET as a default good model)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-mpnet-base-v2"
        )
        
        # Get the index
        self.index = self.pc.Index(self.index_name)
        
        # Initialize vector store with the updated PineconeVectorStore
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embedding_model,
            text_key="text",
            pinecone_api_key=self.pinecone_api_key
        )
    
    def add_documents(self, docs: List[Document]) -> None:
        """Add documents to the vector store."""
        self.vector_store.add_documents(docs)
    
    def similar_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents."""
        return self.vector_store.similarity_search(query, k=k)
    
    def get_documents_by_source(self, source: str, limit: int = 10) -> List[Document]:
        """
        Retrieve documents by source, sorted by date (newest first).
        
        Args:
            source: Source name to filter by (e.g., 'bbc', 'guardian')
            limit: Maximum number of documents to return
            
        Returns:
            List of documents from the specified source
        """
        try:
            # First, use a similar_search with a general query to get documents
            all_docs = self.similar_search(f"{source} news", k=100)
            
            # Filter by source
            source_docs = []
            for doc in all_docs:
                doc_source = doc.metadata.get('source', '')
                if doc_source.lower() == source.lower():
                    source_docs.append(doc)
            
            # Sort by date (newest first)
            try:
                source_docs.sort(key=lambda x: datetime.fromisoformat(x.metadata.get('date', '2000-01-01')), reverse=True)
            except Exception as e:
                print(f"Error sorting documents by date: {e}")
            
            # Limit the number of results
            return source_docs[:limit]
            
        except Exception as e:
            print(f"Error retrieving documents by source: {e}")
            return []
    
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents from the vector store by ID."""
        self.vector_store.delete(ids)
    
    def clear_vector_store(self) -> None:
        """Delete all documents from the vector store."""
        self.index.delete(delete_all=True)
    
    def get_all_document_urls(self) -> Set[str]:
        """Retrieve all URLs from documents in the vector store."""
        try:
            # Fetch all vectors with their metadata
            fetch_response = self.index.query(
                vector=[0.0] * 768,  # Dummy vector, we only care about metadata
                top_k=10000,         # Set high to get all documents
                include_metadata=True,
                include_values=False
            )
            
            # Extract URLs from metadata
            urls = set()
            if hasattr(fetch_response, 'matches'):
                for match in fetch_response.matches:
                    if hasattr(match, 'metadata') and match.metadata:
                        url = match.metadata.get('url', '')
                        if url:
                            urls.add(url)
            
            return urls
            
        except Exception as e:
            print(f"Error retrieving document URLs: {e}")
            return set()
    
    def db_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            print(f"Error getting vector store stats: {e}")
            return {}

def document_from_article(article: Any) -> Document:
    """Convert a news article to a Document for vector storage."""
    metadata = {
        "title": article.title,
        "url": article.url,
        "date": article.date.isoformat() if hasattr(article.date, 'isoformat') else article.date,
        "source": article.source
    }
    
    # Create document with article content and metadata
    return Document(page_content=article.content, metadata=metadata)

def documents_from_articles(articles: List[Any]) -> List[Document]:
    """Convert multiple news articles to Documents."""
    return [document_from_article(article) for article in articles] 