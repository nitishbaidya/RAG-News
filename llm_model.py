from typing import List, Dict, Any
import json
import traceback
import sys
from openai import OpenAI
from langchain.schema.document import Document
from config import NVIDIA_NIM_API_KEY, GEMMA_MODEL

# Maximum context size for Gemma 3 model
MAX_CONTEXT_SIZE = 3500  # Keeping some buffer below the 4096 limit

class Gemma3LLM:
    def __init__(self):
        """Initialize the Gemma 3 model with Nvidia NIM."""
        self.api_key = NVIDIA_NIM_API_KEY
        self.model_name = GEMMA_MODEL
        
        print(f"Configuring NIM client with model: {self.model_name}")
        print(f"API key first 5 chars: {self.api_key[:5]}...")
        
        # Initialize the OpenAI client configured for NIM
        try:
            # Using the correct base URL for NVIDIA NIM
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://integrate.api.nvidia.com/v1"
            )
            print("NIM client initialized successfully")
        except Exception as e:
            print(f"Error initializing NIM client: {e}")
            traceback.print_exc(file=sys.stdout)
            raise
    
    def generate_response(self, query: str, docs: List[Document]) -> str:
        """Generate a response to a query based on retrieved documents."""
        try:
            # Create a context from documents (with length limit)
            context = self._format_documents_with_limit(docs, MAX_CONTEXT_SIZE)
            
            # Create a prompt for the RAG query
            prompt = f"""Answer the following question about UK public policy based on the provided context. 
            If the question cannot be answered based on the context, simply state that you don't have enough information.
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:"""
            
            print(f"\nSending query to NIM API: {query}")
            print(f"Using model: {self.model_name}")
            print(f"Context length: {len(context)} characters")
            
            # Generate response using the OpenAI client
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1024
                )
                
                print("NIM API response received successfully")
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error in NIM API call: {e}")
                print(f"Headers: {getattr(e, 'headers', 'No headers')}")
                print(f"Body: {getattr(e, 'body', 'No body')}")
                traceback.print_exc(file=sys.stdout)
                return f"Sorry, I encountered an error with the NIM API: {str(e)}"
                
        except Exception as e:
            print(f"Error generating response: {e}")
            traceback.print_exc(file=sys.stdout)
            return "Sorry, I encountered an error while generating a response."
    
    def extract_topics(self, docs: List[Document], max_topics: int = 10) -> List[str]:
        """Extract main topics from a collection of documents."""
        try:
            # Sample documents to keep within token limits
            sample_docs = docs[:min(len(docs), 15)]  # Take at most 15 documents
            
            # Create a context from sampled documents
            context = self._format_documents_with_limit(sample_docs, MAX_CONTEXT_SIZE)
            
            # Create a prompt for topic extraction
            prompt = f"""Below are snippets from various news articles about UK public policy and politics.
            
            Articles:
            {context}
            
            Based only on these articles, identify the main topics discussed. 
            List exactly {max_topics} topics as a bulleted list in the format:
            • Topic 1
            • Topic 2
            
            Keep each topic name very short (1-3 words). Don't use complete sentences. For example, use "NHS Funding" not "The funding challenges facing the NHS".
            """
            
            print(f"\nExtracting topics from {len(sample_docs)} documents")
            
            # Generate topics using the OpenAI client
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # Slightly higher temperature for diversity
                max_tokens=256
            )
            
            # Process the response to extract topics
            topics_text = response.choices[0].message.content
            
            # Parse bulleted list (lines starting with • or - or *)
            topics = []
            for line in topics_text.split('\n'):
                line = line.strip()
                if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                    # Remove the bullet point and trim
                    topic = line[1:].strip()
                    topics.append(topic)
            
            # Take only the requested number of topics
            return topics[:max_topics]
            
        except Exception as e:
            print(f"Error extracting topics: {e}")
            traceback.print_exc(file=sys.stdout)
            return ["Error extracting topics"]
    
    def summarize_text(self, text: str) -> str:
        """Summarize the given text."""
        try:
            # Truncate text if it's too long
            if len(text) > MAX_CONTEXT_SIZE:
                text = text[:MAX_CONTEXT_SIZE] + "..."
                
            prompt = f"""Summarize the following text in a concise manner:
            
            {text}
            
            Summary:"""
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error summarizing text: {e}")
            traceback.print_exc(file=sys.stdout)
            return "Failed to generate summary."
    
    def _format_documents(self, docs: List[Document]) -> str:
        """Format documents into a string context."""
        formatted_docs = []
        
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            source = metadata.get("source", "Unknown")
            title = metadata.get("title", "Untitled")
            url = metadata.get("url", "")
            date = metadata.get("date", "")
            
            formatted_doc = f"Document {i+1}:\nTitle: {title}\nSource: {source}\nDate: {date}\nURL: {url}\n\nContent: {doc.page_content}\n"
            formatted_docs.append(formatted_doc)
        
        return "\n".join(formatted_docs)
        
    def _format_documents_with_limit(self, docs: List[Document], max_chars: int) -> str:
        """Format documents into a string context with a maximum character limit."""
        if not docs:
            return "No documents available."
            
        # Determine how many chars we can use per document (approximately)
        # Reserve some space for metadata per document
        metadata_overhead = 150  # Approximate chars for metadata per doc
        available_chars_per_doc = (max_chars // len(docs)) - metadata_overhead
        
        # If we have very few documents, we don't need to truncate as much
        if available_chars_per_doc > 1000:
            available_chars_per_doc = 1000  # Cap at 1000 chars per document to allow for more documents
            
        formatted_docs = []
        total_length = 0
        
        for i, doc in enumerate(docs):
            if total_length >= max_chars:
                break
                
            metadata = doc.metadata
            source = metadata.get("source", "Unknown")
            title = metadata.get("title", "Untitled")
            url = metadata.get("url", "")
            date = metadata.get("date", "")
            
            # Calculate metadata length
            metadata_text = f"Document {i+1}:\nTitle: {title}\nSource: {source}\nDate: {date}\nURL: {url}\n\nContent: "
            metadata_length = len(metadata_text)
            
            # Calculate available space for content
            remaining_chars = max_chars - total_length - metadata_length
            
            # If not enough space for meaningful content, skip this document
            if remaining_chars < 100:
                break
                
            # Get content and truncate if necessary
            content = doc.page_content
            if len(content) > available_chars_per_doc:
                content = content[:available_chars_per_doc] + "..."
                
            # Add formatted document
            formatted_doc = metadata_text + content + "\n"
            formatted_docs.append(formatted_doc)
            
            # Update total length
            total_length += len(formatted_doc)
            
        return "\n".join(formatted_docs) 