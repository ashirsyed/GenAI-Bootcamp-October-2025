"""
Streamlit app to interact with the LLMApp class
"""

from sre_parse import State
import streamlit as st
from main import LLMApp

# page configuration
st.set_page_config(
    page_title="Mini LLM Chat Application",
    page_icon=":robot_face:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# initialize session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm_app" not in st.session_state:
    st.session_state.llm_app = None

# Title and description
st.title(":robot_face: Mini LLM Chat Application")
st.markdown("Chat with powerful AI models powered by Groq and OpenAI APIs.")

# Implement sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    #API Key Inputs
    api_key = st.sidebar.text_input("Groq API Key", type="password", help="Enter your Groq API key")
    if not api_key:
        api_key = LLMApp().groq_api_key

    # Model Selection
    model = st.selectbox(
        "Model",
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-120b",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
        ],
        help="Select the model to use for the chat"
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Select a value to control response randomness. Higher values make output more random."
    )

    max_tokens = st.slider(
        "Max Tokens",
        min_value=256,
        max_value=8192,
        value=1024,
        step=256,
        help="Select the maximum number of tokens for the chat"
    )

    system_prompt = st.text_area(
        "System Prompt (Optional)",
        placeholder="Enter a system prompt for the chatbot",
        help="Enter a system prompt for the chatbot. This will be used to guide the chatbot's behavior."
    )

    # Clear chat button
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.llm_app:
            st.session_state.llm_app.clear_conversation_history()
        st.success("Chat history cleared!")
        st.rerun()

if st.session_state.llm_app is None:
    try:
        st.session_state.llm_app = LLMApp(groq_api_key=api_key, model=model, temperature=temperature, max_tokens=max_tokens, system_prompt=system_prompt)
        st.success("Chatbot initialized successfully!")
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}")
        st.stop()

# display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# user input
if prompt := st.chat_input("Enter a message:"):
    if not api_key:
        st.warning("Please enter your API key for Groq.")

    else:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": f"{prompt}"
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        # get response from chatbot
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.llm_app.chat(
                        user_message=prompt,
                        system_prompt=system_prompt if system_prompt else None,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        model=model,
                    )

                    st.markdown(response)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"{response}"
                        }
                    )
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                        }
                    )

