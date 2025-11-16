# Agentic Blog Writer

An intelligent blog generation system that creates complete blog posts with titles, content, audio narration, and cover images using LangGraph, Groq, AWS Bedrock, and ElevenLabs.

## 🚀 Features

- **AI-Powered Blog Generation**: Generate complete blog posts from a simple topic
- **Multi-Modal Output**: Creates title, content, audio narration, and cover images
- **LangGraph Workflow**: Uses stateful graph-based workflow for sequential blog generation
- **Streamlit Interface**: User-friendly web interface for blog generation
- **Multiple LLM Models**: Support for various Groq models (Llama, Mixtral, Gemma)
- **Audio Generation**: Text-to-speech using ElevenLabs API
- **Image Generation**: Cover images using AWS Bedrock Nova Canvas

## 📋 Prerequisites

- Python 3.11 or higher
- Groq API key (get one from [Groq Console](https://console.groq.com/))
- AWS credentials configured (for image generation via Bedrock)
- ElevenLabs API key (optional, for better quality audio)

## 🛠️ Installation

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd week_4
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file** in the project root:
   ```bash
   touch .env
   ```

5. **Add your API keys to `.env`**:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
   ELEVENLABS_API_URL=https://api.elevenlabs.io/v1/text-to-speech/YOUR_VOICE_ID
   AWS_REGION=us-east-1
   ```

   **Note**: For AWS Bedrock, you can either:
   - Use AWS credentials from `~/.aws/credentials` (recommended)
   - Or set `BEDROCK_API_KEY` and `AWS_BEARER_TOKEN` in `.env` if using a custom endpoint

## 🎯 Usage

### Streamlit Web App (Recommended)

Run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

The app will open in your browser. You can:
- Enter your Groq API key (optional if set in `.env`)
- Select a model
- Enter a topic in the chat input
- View generated blogs with title, content, audio, and images

### Command Line

Run the Python script directly:

```bash
python app.py
```

Or modify the `__main__` section in `app.py` to use your own topic.

## 📁 Project Structure

```
week_4/
├── app.py                 # Main AgenticBlogWriter class
├── appconfig.py           # Configuration management
├── blog_node.py           # Blog generation nodes (title, content, audio, photo)
├── blogstate.py           # State schema definitions
├── graph_builder.py       # LangGraph workflow builder
├── streamlit_app.py       # Streamlit web interface
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── data/                  # Generated files (audio, images)
│   ├── blog_audio_*.mp3
│   └── blog_image_*.png
└── README.md             # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Your Groq API key |
| `ELEVENLABS_API_KEY` | Optional | ElevenLabs API key for audio generation |
| `ELEVENLABS_API_URL` | Optional | ElevenLabs API endpoint URL |
| `AWS_REGION` | Optional | AWS region (default: us-east-1) |
| `BEDROCK_API_KEY` | Optional | Bedrock API key (if using custom endpoint) |
| `AWS_BEARER_TOKEN` | Optional | AWS bearer token (if using custom endpoint) |

### AWS Credentials

For AWS Bedrock image generation, configure AWS credentials using one of these methods:

1. **AWS Credentials File** (Recommended):
   ```bash
   ~/.aws/credentials
   ```
   ```ini
   [default]
   aws_access_key_id = YOUR_ACCESS_KEY
   aws_secret_access_key = YOUR_SECRET_KEY
   ```

2. **Environment Variables**:
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_REGION=us-east-1
   ```

## 🏗️ Architecture

The application uses **LangGraph** to create a stateful workflow:

1. **Title Creation**: Generates SEO-friendly blog title
2. **Content Generation**: Creates detailed blog content
3. **Audio Generation**: Converts content to audio using ElevenLabs
4. **Photo Generation**: Creates cover image using AWS Bedrock Nova Canvas

```
START → Title Creation → Content Generation → Audio Generation → Photo Generation → END
```

## 📦 Dependencies

- `groq` - Groq API client
- `python-dotenv` - Environment variable management
- `streamlit` - Web interface framework
- `boto3` - AWS SDK for Bedrock
- `langchain` - LLM framework
- `langgraph` - Stateful graph workflows
- `langchain_core` - Core LangChain components
- `langchain_groq` - Groq integration for LangChain
- `requests` - HTTP library for API calls

## 🎨 Supported Models

The application supports the following Groq models:

- `llama-3.1-8b-instant` (default) - Fast and efficient
- `llama-3.1-70b-versatile` - More capable, slower
- `mixtral-8x7b-32768` - Good for longer contexts
- `gemma2-9b-it` - Alternative option

## 📝 Generated Files

All generated files are saved in the `data/` directory:

- **Audio**: `blog_audio_YYYYMMDD_HHMMSS.mp3`
- **Images**: `blog_image_YYYYMMDD_HHMMSS.png`

## 🔍 Troubleshooting

### Common Issues

1. **"Blog writer not initialized"**
   - Ensure your Groq API key is set in `.env` or entered in the Streamlit sidebar

2. **Image generation fails**
   - Verify AWS credentials are configured correctly
   - Check that Nova Canvas model is enabled in your AWS Bedrock console
   - Ensure you're using the correct AWS region

3. **Audio generation fails**
   - Check ElevenLabs API key and URL in `.env`
   - Verify the voice ID in the API URL is correct

4. **Import errors**
   - Make sure virtual environment is activated
   - Run `pip install -r requirements.txt` again

## 📄 License

This project is part of the Andela GenAI Bootcamp.

## 🤝 Contributing

This is a learning project. Feel free to experiment and modify as needed!

## 📚 Resources

- [Groq Documentation](https://console.groq.com/docs)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [ElevenLabs Documentation](https://elevenlabs.io/docs)

---

**Happy Blog Writing! 📝✨**

