# Mini LLM Chat Application

A powerful chatbot application that seamlessly integrates with both **Groq** and **OpenAI** APIs, allowing you to chat with multiple AI models through a user-friendly Streamlit interface.

## Features

### 🤖 Multi-Provider Support
- **Automatic Provider Detection**: Automatically switches between Groq and OpenAI based on the selected model
- **Groq Models**: Fast inference with Llama, Mixtral, and Gemma models
- **OpenAI Models**: Access to GPT-4, GPT-3.5-turbo, and future GPT-5 models
- **Dynamic Model Switching**: Change models mid-conversation without losing chat history

### 💬 Conversation Management
- **Persistent History**: Maintains conversation context across messages
- **Chatbot Identity**: Customizable chatbot personality and name
- **System Prompts**: Configure custom system prompts to guide AI behavior
- **Clear History**: One-click button to clear conversation history

### 🎛️ Flexible Configuration
- **Temperature Control**: Adjust response creativity (0.0 - 2.0)
- **Token Limits**: Configure maximum output length (256 - 8192 tokens)
- **Custom System Prompts**: Define the AI's behavior and personality
- **Environment Variables**: Secure API key management via `.env` file

### 🎨 User Interface
- Clean, modern Streamlit interface
- Real-time chat experience
- Sidebar configuration panel
- Error handling and user feedback

## Installation

### Prerequisites
- Python 3.8 or higher
- Groq API key (get one at [console.groq.com](https://console.groq.com))
- OpenAI API key (optional, for GPT models)

### Setup

1. **Clone the repository**
   ```bash
   cd week_1
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file**
   ```bash
   cp .env.example .env  # Or create manually
   ```

5. **Add your API keys to `.env`**
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Python Module

```python
from main import LLMApp

# Initialize with default Groq model
app = LLMApp(model="llama-3.3-70b-versatile")

# Chat with custom parameters
response = app.chat(
    user_message="Hello!",
    model="gpt-4",  # Switch to OpenAI
    temperature=0.7,
    max_tokens=512
)

print(response)

# Access conversation history
history = app.get_conversation_history()
print(history)
```

## Supported Models

### Groq Models (Fast & Free)
- `llama-3.1-8b-instant` - Fast, efficient Llama model
- `llama-3.3-70b-versatile` - Advanced Llama model for complex tasks
- `mixtral-8x7b-32768` - Large context window Mixtral model
- `gemma-7b-it` - Google's Gemma model (instruction-tuned)

### OpenAI Models (Requires API Key)
- `gpt-4` - OpenAI's most capable model
- `gpt-4-turbo` - Enhanced GPT-4 with better performance
- `gpt-4o` - Latest GPT-4 Optimized version
- `gpt-3.5-turbo` - Fast and cost-effective
- `gpt-5`, `gpt-5-mini`, `gpt-5-nano` - Future OpenAI models

## Configuration

### Environment Variables

Create a `.env` file in the `week_1` directory:

```env
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```

### LLMApp Parameters

```python
LLMApp(
    groq_api_key="optional",      # Override .env if needed
    model="llama-3.3-70b-versatile",  # Default model
    temperature=0.5,                # Creativity (0-2)
    max_tokens=1024,               # Response length
    system_prompt="Custom prompt", # AI personality
    chatbot_name="Ally"            # Bot name
)
```

## Project Structure

```
week_1/
├── main.py              # LLMApp class - core chatbot logic
├── streamlit_app.py     # Streamlit web interface
├── appconfig.py         # Configuration management
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## API

### LLMApp Class

#### Methods

- `chat(user_message, model=None, system_prompt=None, temperature=None, max_tokens=None)`
  - Send a message and get AI response
  - All parameters are optional and override instance defaults

- `set_model(model)`
  - Dynamically switch AI models
  - Automatically selects provider (Groq/OpenAI)

- `clear_conversation_history()`
  - Reset conversation history

- `get_conversation_history()`
  - Returns full conversation history as a list

- `get_provider_info()`
  - Get current provider, model, and chatbot info

## How It Works

1. **Provider Detection**: Automatically detects whether the model uses Groq or OpenAI
2. **Client Switching**: Dynamically switches API clients when changing models
3. **History Management**: Maintains conversation context across all messages
4. **Parameter Updates**: Temperature and token settings update in real-time

### Example Flow

```python
# Start with Groq
app = LLMApp(model="llama-3.3-70b-versatile")
app.chat("Hello!")

# Switch to OpenAI automatically
app.chat("Tell me a joke", model="gpt-4")

# History is preserved
print(app.get_conversation_history())
# Includes both Groq and OpenAI messages
```

## Troubleshooting

### "API key is required" Error
- Ensure your `.env` file exists in `week_1/` directory
- Check that API keys are correctly set in `.env`
- Verify no extra spaces in the keys

### Import Errors
```bash
# Install missing dependencies
pip install -r requirements.txt
```

### Model Not Found
- Verify the model name spelling
- Check if the model is available in your API tier
- For OpenAI models, ensure you have API credits

## Contributing

This is a week 1 project for the GenAI Bootcamp (October 2025).

## License

Educational project for bootcamp purposes.

