import os
import streamlit as st

# Set environment variables from Streamlit secrets
os.environ["PINECONE_API_KEY"] = st.secrets["pinecone"]["api_key"]
os.environ["PINECONE_ENVIRONMENT"] = st.secrets["pinecone"]["environment"]

# Also set the regular variables for our own code
PINECONE_API_KEY = st.secrets["pinecone"]["api_key"]
PINECONE_ENVIRONMENT = st.secrets["pinecone"]["environment"]
PINECONE_INDEX_NAME = st.secrets["pinecone"]["index_name"]

# Nvidia NIM configuration
os.environ["NVIDIA_NIM_API_KEY"] = st.secrets["nim"]["api_key"]
NVIDIA_NIM_API_KEY = st.secrets["nim"]["api_key"]

# RSS Feed URLs
RSS_FEEDS = {
    "bbc": "https://feeds.bbci.co.uk/news/uk/rss.xml",
    "guardian": "https://www.theguardian.com/uk-news/rss",
    "gov_uk": "https://www.gov.uk/government/publications.atom"
}

# Gemma 3 model configuration
GEMMA_MODEL = "google/gemma-3-27b-it" 
