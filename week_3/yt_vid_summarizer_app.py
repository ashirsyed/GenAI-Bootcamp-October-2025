import streamlit as st
import os
from yt_vid_summarizer import YoutubeVideoSummarizer
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="YouTube Video Summarizer & Chat",
    page_icon="🎥",
    layout="wide"
)

# Title and description
st.title("🎥 YouTube Video Summarizer & Chat")
st.markdown("Enter a YouTube URL to generate a summary and chat about the video content")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # LLM Model selection
    st.subheader("LLM Model")
    llm_choice = st.selectbox(
        "Choose LLM Model",
        ["AWS Bedrock", "AIMLAPI GPT-4o-mini"],
        index=0,
        help="Select the AI model for summarization and Q&A"
    )
    
    # Embedding selection
    st.subheader("Embeddings")
    embedding_choice = st.selectbox(
        "Choose Embeddings",
        ["AWS Bedrock (Titan)", "AIMLAPI", "Chroma Default"],
        index=0,
        help="Select the embedding model for vector search"
    )
    
    # Model name for Bedrock
    if llm_choice == "AWS Bedrock":
        bedrock_model = st.selectbox(
            "Bedrock Model",
            ["us.amazon.nova-pro-v1:0", "anthropic.claude-3-5-sonnet-20241022-v2:0", "anthropic.claude-3-haiku-20240307-v1:0"],
            index=0
        )
    else:
        bedrock_model = None
    
    st.markdown("---")
    st.markdown("### 📊 Info")
    st.info("The video will be downloaded, transcribed, and processed for summarization and Q&A.")
    
    # Check API keys
    if llm_choice == "AWS Bedrock" or embedding_choice == "AWS Bedrock (Titan)":
        aws_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK") or os.getenv("AWS_ACCESS_KEY_ID")
        if not aws_key:
            st.warning("⚠️ AWS credentials not found in .env file")
    
    if llm_choice == "AIMLAPI" or embedding_choice == "AIMLAPI":
        aimlapi_key = os.getenv("AIMLAPI_API_KEY")
        if not aimlapi_key:
            st.warning("⚠️ AIMLAPI_API_KEY not found in .env file")

# Initialize session state
if "summarizer" not in st.session_state:
    st.session_state.summarizer = None
if "video_result" not in st.session_state:
    st.session_state.video_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Main content area
url = st.text_input(
    "Enter YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=...",
    help="Paste the URL of the YouTube video you want to summarize"
)

# Button to process video
if st.button("🚀 Process Video", type="primary", use_container_width=True):
    if not url:
        st.error("⚠️ Please enter a valid YouTube URL")
    else:
        try:
            # Configure model settings based on user selection
            if llm_choice == "AWS Bedrock":
                llm_type = "bedrock"
                llm_model_name = bedrock_model or "us.amazon.nova-pro-v1:0"
            else:
                llm_type = "aimlapi"
                llm_model_name = "gpt-4o-mini"
            
            if embedding_choice == "AWS Bedrock (Titan)":
                embedding_type = "bedrock"
            elif embedding_choice == "AIMLAPI":
                embedding_type = "aimlapi"
            else:
                embedding_type = "chroma"
            
            with st.spinner("🔄 Processing video... This may take a few minutes"):
                # Show progress steps
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialize summarizer
                status_text.text("Initializing models...")
                progress_bar.progress(10)
                summarizer = YoutubeVideoSummarizer(
                    llm_type=llm_type,
                    llm_model_name=llm_model_name,
                    embedding_type=embedding_type,
                )
                st.session_state.summarizer = summarizer
                
                # Process video
                status_text.text("Downloading video...")
                progress_bar.progress(20)
                
                status_text.text("Transcribing audio...")
                progress_bar.progress(40)
                
                status_text.text("Generating summary...")
                progress_bar.progress(60)
                
                status_text.text("Creating vector store...")
                progress_bar.progress(80)
                
                result = summarizer.process_video(url)
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                
                if result:
                    st.session_state.video_result = result
                    st.session_state.chat_history = []  # Reset chat history
                    st.success("✅ Video processed successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to process video. Please check the URL and try again.")
                    
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# Display results if video is processed
if st.session_state.video_result:
    result = st.session_state.video_result
    
    # Video Information
    st.header("📹 Video Information")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### {result['title']}")
        st.markdown(f"**URL:** {result.get('url', url)}")
    
    with col2:
        model_info = st.session_state.summarizer.get_model_info()
        st.markdown("**Model Configuration:**")
        st.caption(f"LLM: {model_info['llm_type']}")
        st.caption(f"Model: {model_info['llm_model']}")
        st.caption(f"Embeddings: {model_info['embedding_type']}")
    
    st.markdown("---")
    
    # Summary Section
    st.header("📝 Summary")
    summary_text = result['summary']
    
    # Handle different summary formats
    if isinstance(summary_text, dict):
        if 'output_text' in summary_text:
            st.markdown(summary_text['output_text'])
        elif 'text' in summary_text:
            st.markdown(summary_text['text'])
        else:
            st.markdown(str(summary_text))
    else:
        st.markdown(summary_text)
    
    st.markdown("---")
    
    # Chat Section
    st.header("💬 Chat About the Video")
    st.markdown("Ask questions about the video content. The AI will answer based on the video transcript.")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Chat History")
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)
    
    # Chat input
    user_question = st.chat_input("Ask a question about the video...")
    
    if user_question:
        # Add user question to chat
        with st.chat_message("user"):
            st.write(user_question)
        
        # Get answer from QA chain
        try:
            with st.spinner("Thinking..."):
                qa_response = result["qa_chain"].invoke({"question": user_question})
                answer = qa_response.get("answer", "Sorry, I couldn't generate an answer.")
            
            # Add to chat history
            st.session_state.chat_history.append((user_question, answer))
            
            # Display answer
            with st.chat_message("assistant"):
                st.write(answer)
            
            # Rerun to update chat history display
            st.rerun()
            
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
    
    # Full Transcript Section (Collapsible)
    st.markdown("---")
    with st.expander("📄 View Full Transcript"):
        transcript = result.get('full_transcript', 'Transcript not available')
        st.text_area(
            "Full Transcript",
            value=transcript,
            height=400,
            label_visibility="collapsed"
        )
        
        # Download transcript
        st.download_button(
            label="📥 Download Transcript",
            data=transcript,
            file_name=f"transcript_{result['title'][:50].replace(' ', '_')}.txt",
            mime="text/plain"
        )
    
    # Clear button
    st.markdown("---")
    if st.button("🗑️ Clear & Process New Video", type="secondary"):
        st.session_state.video_result = None
        st.session_state.chat_history = []
        st.session_state.summarizer = None
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Powered by Whisper, LangChain & AWS Bedrock/AIMLAPI</p>
    </div>
    """,
    unsafe_allow_html=True
)

