import streamlit as st
import os
from news_summarizer import NewsArticleSummarizer
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="News Article Summarizer",
    page_icon="📰",
    layout="wide"
)

# Title and description
st.title("📰 News Article Summarizer")
st.markdown("Enter a news article URL to generate a detailed summary using AI")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    model_type = st.selectbox(
        "Model Type",
        ["aimlapi"],
        help="Select the AI model to use for summarization"
    )
    
    # Model name selection
    model_name = st.selectbox(
        "Model Name",
        ["gpt-3.5-turbo-instruct", "gpt-4o-mini", "gpt-4"],
        help="Select the specific model"
    )
    
    # Summary type
    summary_type = st.selectbox(
        "Summary Type",
        ["detailed", "concise"],
        help="Choose between detailed or concise summary"
    )
    
    # API Key input
    api_key = st.text_input(
        "AIMLAPI API Key",
        type="password",
        value=os.getenv("AIMLAPI_API_KEY", ""),
        help="Enter your AIMLAPI API key or set it in .env file"
    )
    
    #st.markdown("---")
    #st.markdown("### 📊 Settings")
    #st.info("The article will be split into chunks for processing. Each chunk will be summarized separately, then combined into a final summary.")

# Main content area
url = st.text_input(
    "Enter News Article URL",
    placeholder="https://example.com/news-article",
    help="Paste the URL of the news article you want to summarize"
)

# Initialize session state
if "summarizer" not in st.session_state:
    st.session_state.summarizer = None
if "summary_result" not in st.session_state:
    st.session_state.summary_result = None

# Button to generate summary
if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
    if not url:
        st.error("⚠️ Please enter a valid URL")
    elif not api_key:
        st.error("⚠️ Please enter your AIMLAPI API key")
    else:
        try:
            with st.spinner("🔄 Processing article..."):
                # Initialize summarizer
                summarizer = NewsArticleSummarizer(
                    api_key=api_key,
                    model_type=model_type,
                    model_name=model_name
                )
                st.session_state.summarizer = summarizer
                
                # Fetch article first to show info
                article = summarizer.fetch_article(url)
                if not article:
                    st.error("❌ Failed to fetch article. Please check the URL and try again.")
                else:
                    # Calculate document count
                    article_length = len(article.text)
                    overlap = summarizer.text_splitter._chunk_overlap
                    chunk_size = summarizer.text_splitter._chunk_size
                    num_documents = (article_length - overlap) / (chunk_size - overlap)
                    
                    # Show article info
                    st.info(f"📄 Article loaded: {article_length:,} characters | Will be split into ~{num_documents:.1f} documents")
                    
                    # Generate summary
                    summary_result = summarizer.summarize(url, summary_type=summary_type)
                    st.session_state.summary_result = summary_result
                    
                    st.success("✅ Summary generated successfully!")
                    
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Display results
if st.session_state.summary_result:
    result = st.session_state.summary_result
    
    if "error" in result:
        st.error(f"❌ {result['error']}")
    else:
        # Article Information
        st.header("📋 Article Information")

        st.markdown(f"**Title:** {result['title']}")

        col2, col3 = st.columns(2)
        
        #with col1:
        #    st.metric("Title", result['title'] or "N/A")
        with col2:
            authors = ', '.join(result['authors']) if result['authors'] else "N/A"
            st.metric("Authors", authors)
        with col3:
            publish_date = result['publish_date'].strftime("%Y-%m-%d") if result['publish_date'] else "N/A"
            st.metric("Published", publish_date)
        
        st.markdown(f"**URL:** {result['url']}")
        st.markdown(f"**Model:** {result['model_info']['type']} - {result['model_info']['name']}")
        
        st.markdown("---")
        
        # Summary
        st.header("📝 Summary")
        summary_text = result['summary']
        
        # Check if summary is a dict (from map_reduce chain)
        if isinstance(summary_text, dict):
            # Display the final combined summary
            if 'output_text' in summary_text:
                st.markdown(summary_text['output_text'])
            elif 'text' in summary_text:
                st.markdown(summary_text['text'])
            else:
                st.markdown(str(summary_text))
            
            # Show all document contents
            #st.markdown("---")
            #st.header("📚 Document Contents")
            
            #if 'input_documents' in summary_text:
            #    input_docs = summary_text['input_documents']
            #    st.info(f"📊 Total documents processed: {len(input_docs)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Powered by AIMLAPI & LangChain</p>
    </div>
    """,
    unsafe_allow_html=True
)

