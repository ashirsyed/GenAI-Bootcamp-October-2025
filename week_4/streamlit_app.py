import streamlit as st
from app import AgenticBlogWriter
import os

# Page configuration
st.set_page_config(
    page_title="Agentic Blog Writer",
    page_icon="📝",
    layout="wide"
)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = ""
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "llama-3.1-8b-instant"
if 'blog_writer' not in st.session_state:
    st.session_state.blog_writer = None

# Available Groq models
GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Groq API Key input (optional)
    groq_api_key = st.text_input(
        "Groq API Key (Optional)",
        type="password",
        value=st.session_state.groq_api_key,
        help="Optional: Enter your Groq API key. If not provided, will use GROQ_API_KEY from environment variables."
    )
    
    # Model selection
    selected_model = st.selectbox(
        "Select Model",
        options=GROQ_MODELS,
        index=GROQ_MODELS.index(st.session_state.selected_model) if st.session_state.selected_model in GROQ_MODELS else 0,
        help="Choose the Groq model to use for blog generation"
    )
    
    # Update session state
    st.session_state.groq_api_key = groq_api_key
    st.session_state.selected_model = selected_model
    
    try:
        # Reinitialize if model changed or writer doesn't exist
        if (st.session_state.blog_writer is None or 
            st.session_state.get('current_model') != selected_model):
            st.session_state.blog_writer = AgenticBlogWriter(
                groq_api_key=st.session_state.groq_api_key if st.session_state.groq_api_key else None,
                model=selected_model
            )
            st.session_state.current_model = selected_model
    except Exception as e:
        st.error(f"Error initializing blog writer: {e}")
        st.session_state.blog_writer = None
    
    st.divider()
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.conversation_history = []
        st.rerun()

# Main content area
st.title("📝 Agentic Blog Writer")
st.markdown("Generate complete blog posts with title, content, audio, and images!")

if st.session_state.conversation_history:
    for idx, entry in enumerate(st.session_state.conversation_history):
        if entry.get('status') == 'generating':
            st.info(f"🔄 Generating blog for: {entry.get('topic', 'Unknown Topic')}...")
            continue
        
        if entry.get('status') == 'error':
            st.error(f"❌ Error generating blog for: {entry.get('topic', 'Unknown Topic')}")
            if entry.get('error'):
                st.error(f"Error: {entry.get('error')}")
            continue
        
        st.divider()
        
        # Title
        if entry.get('title'):
            st.markdown(f"## {entry['title']}")
        
        # Layout: Image and Audio on left, Content on right
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display image
            if entry.get('photo_url') and os.path.exists(entry['photo_url']):
                st.image(entry['photo_url'], use_container_width=True)
            
            # Display audio
            if entry.get('audio_url') and os.path.exists(entry['audio_url']):
                st.markdown("**🎵 Audio**")
                with open(entry['audio_url'], 'rb') as audio_file:
                    st.audio(audio_file.read(), format='audio/mp3')
        
        with col2:
            # Display content
            if entry.get('content'):
                st.markdown("**📄 Content**")
                st.markdown(entry['content'])

# Topic input
topic = st.chat_input("Enter a topic for your blog post...")

if topic:
    if st.session_state.blog_writer is None:
        st.error("❌ Blog writer not initialized. Please check your configuration.")
    else:
        # Add user message to conversation
        st.session_state.conversation_history.append({
            'topic': topic,
            'status': 'generating'
        })
        
        # Show progress
        with st.spinner(f"🔄 Generating blog for '{topic}'... This may take a minute."):
            try:
                # Generate blog
                result = st.session_state.blog_writer.write_blog(topic)
                
                # Extract blog data
                blog_data = result.get('blog', {})
                
                # Update conversation history with results
                st.session_state.conversation_history[-1] = {
                    'topic': topic,
                    'title': blog_data.get('title', ''),
                    'content': blog_data.get('content', ''),
                    'audio_url': blog_data.get('audio_url', ''),
                    'photo_url': blog_data.get('photo_url', ''),
                    'status': 'completed'
                }
                
                st.success("✅ Blog generated successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating blog: {e}")
                st.session_state.conversation_history[-1]['status'] = 'error'
                st.session_state.conversation_history[-1]['error'] = str(e)
                st.rerun()
