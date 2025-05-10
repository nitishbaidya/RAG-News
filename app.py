import streamlit as st
import time
from rag_system import UKPolicyRAG, fetch_news, get_available_topics, answer_query

# Set page config
st.set_page_config(
    page_title="UK News RAG",
    page_icon="🇬🇧",
    layout="wide"
)

# Disable some animations for better performance
st.markdown("""
<style>
    div.element-container {
        transition: none !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_rag_system():
    """Initialize the RAG system once and cache it."""
    return UKPolicyRAG()

def main():
    # Display headers
    st.title("Retrieval-Augmented Generation on UK News")
    st.subheader("Powered by Gemma 3, NVIDIA NIM, and Pinecone")
    
    # Initialize session state variables
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        
    if 'topics' not in st.session_state:
        st.session_state.topics = []
    
    if 'refresh_status' not in st.session_state:
        st.session_state.refresh_status = None
    
    if 'refresh_time' not in st.session_state:
        st.session_state.refresh_time = 0
    
    if 'num_articles' not in st.session_state:
        st.session_state.num_articles = 0
        
    if 'query_result' not in st.session_state:
        st.session_state.query_result = None
    
    # Create a two-column layout
    left_col, right_col = st.columns([1, 1.5])
    
    # Left column - Topics and Fetch
    with left_col:
        # Initialize RAG system and load topics (if not already done)
        if not st.session_state.initialized:
            with st.spinner("Loading RAG system and topics..."):
                rag = get_rag_system()
                st.session_state.topics = rag.get_topics()
                st.session_state.initialized = True
        else:
            rag = get_rag_system()
        
        # Display topics
        if st.session_state.topics:
            st.subheader("Latest Topics")
            topic_cols = st.columns(2)
            for i, topic in enumerate(st.session_state.topics):
                col = topic_cols[i % 2]
                col.markdown(f"- {topic}")
        else:
            st.info("Loading topics...")
        
        # Add refresh button
        st.markdown("---")
        refresh_pressed = st.button("Refresh News Articles", key="refresh_news", use_container_width=True)
        st.info("Fetches articles from BBC, Guardian, and Gov.uk")
        
        # Process the refresh
        if refresh_pressed:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Initializing...")
            progress_bar.progress(10)
            
            status_text.text("Checking for existing articles...")
            progress_bar.progress(20)
            
            with st.spinner("Fetching and processing articles...(it may take a while!)"):
                start_time = time.time()
                num_articles = rag.fetch_and_store_articles()
                end_time = time.time()
                
            progress_bar.progress(100)
            
            if num_articles > 0:
                st.success(f"Successfully added {num_articles} new articles in {end_time - start_time:.2f} seconds!")
                # Reload topics only if new articles were added
                with st.spinner("Updating topics..."):
                    st.session_state.topics = rag.get_topics()
            else:
                st.info("No new articles found.")
    
    # Right column - Query
    with right_col:
        st.header("Ask about UK News Topics")
        
        # Only show query interface if system is initialized
        if st.session_state.initialized:
            # Use a form to prevent page reloads
            with st.form(key="query_form"):
                st.text_area("Your question:", height=100, 
                          placeholder="What are the recent changes to NHS funding?",
                          key="query_input")
                
                # Store current input to detect changes
                if 'current_query' not in st.session_state:
                    st.session_state.current_query = ""
                
                submitted = st.form_submit_button("Submit Question", use_container_width=True)
            
            # Create a placeholder for the loading indicator AFTER the form
            loading_placeholder = st.empty()
            
            # Process the query outside of the callback for better UI control
            if submitted and st.session_state.query_input:
                query = st.session_state.query_input
                
                # Only process if the query has changed
                if query != st.session_state.current_query:
                    st.session_state.current_query = query
                    
                    # Show loading message in place
                    loading_placeholder.info("Searching for information...")
                    
                    # Process the query
                    st.session_state.query_result = rag.query(query)
                    
                    # Clear the loading message
                    loading_placeholder.empty()
            
            # Display results if available
            if st.session_state.query_result:
                result = st.session_state.query_result
                
                # Display the answer
                st.markdown("### Answer")
                st.markdown(f"""<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
                    {result["response"]}
                """, unsafe_allow_html=True)
                
                # Display sources
                st.markdown("### Sources")
                for i, source in enumerate(result["sources"]):
                    with st.expander(f"{i+1}. {source['title']} ({source['source']}, {source['date']})"):
                        st.markdown(f"**URL**: [{source['url']}]({source['url']})")
        else:
            st.info("Loading system... Query functionality will be available shortly.")

if __name__ == "__main__":
    main() 