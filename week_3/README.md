# Week 3: AI-Powered Content Processing & Voice Assistant Projects

This directory contains three comprehensive AI-powered applications built with LangChain, Streamlit, and various AI models for content summarization, video processing, and voice-assisted RAG (Retrieval-Augmented Generation).

## 📋 Table of Contents

- [Projects Overview](#projects-overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Projects](#projects)
  - [News Article Summarizer](#1-news-article-summarizer)
  - [YouTube Video Summarizer](#2-youtube-video-summarizer)
  - [Voice RAG Assistant](#3-voice-rag-assistant)
- [Running Multiple Apps](#running-multiple-apps)
- [Troubleshooting](#troubleshooting)

---

## Projects Overview

### 1. News Article Summarizer 📰
A web application that fetches news articles from URLs and generates AI-powered summaries using map-reduce techniques.

### 2. YouTube Video Summarizer 🎥
Downloads YouTube videos, transcribes them using Whisper, generates summaries, and enables interactive Q&A about video content.

### 3. Voice RAG Assistant 🎤
A voice-enabled RAG system that processes documents, allows voice queries, and responds with AI-generated voice answers.

---

## Installation

### Prerequisites

- Python 3.11+
- FFmpeg (for video/audio processing)
- Virtual environment (recommended)

### Step 1: Create Virtual Environment

```bash
cd week_3
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**Windows:**
Download from [FFmpeg website](https://www.hostinger.com/tutorials/how-to-install-ffmpeg)

---

## Configuration

### Environment Variables

Create a `.env` file in the `week_3` directory with the following variables:

```bash
# AIMLAPI Configuration
AIMLAPI_API_KEY=your_aimlapi_api_key_here

# AWS Bedrock Configuration (Choose one method)
# Method 1: API Key (Recommended)
AWS_BEARER_TOKEN_BEDROCK=your_bedrock_api_key_here
AWS_REGION=us-east-1

# Method 2: Traditional AWS Credentials
# AWS_ACCESS_KEY_ID=your_access_key
# AWS_SECRET_ACCESS_KEY=your_secret_key
# AWS_REGION=us-east-1

# ElevenLabs Configuration (for Voice Assistant)
ELEVEN_LABS_API_KEY=your_elevenlabs_api_key_here

# SSL Configuration (optional, for corporate networks)
SSL_VERIFY=false  # Set to true to enable SSL verification
```

### Getting API Keys

1. **AIMLAPI**: Sign up at [aimlapi.com](https://aimlapi.com) and get your API key
2. **AWS Bedrock**: 
   - Go to AWS Console → Amazon Bedrock
   - Generate API key or configure IAM credentials
   - Request access to desired models
3. **ElevenLabs**: Sign up at [elevenlabs.io](https://elevenlabs.io) and get your API key

---

## Projects

## 1. News Article Summarizer 📰

### Description
Fetches news articles from URLs, splits them into chunks, and generates comprehensive summaries using AI models. Supports both detailed and concise summary types.

### Features
- ✅ URL-based article fetching using `newspaper3k`
- ✅ Intelligent text chunking with overlap
- ✅ Map-reduce summarization for long articles
- ✅ Support for multiple AI models (AIMLAPI, AWS Bedrock)
- ✅ Document count calculation and preview
- ✅ Streamlit web interface

### Files
- `news_summarizer.py` - Core summarizer class
- `news_summarizer_app.py` - Streamlit web application

### Usage

#### Command Line:
```bash
python news_summarizer.py
```

#### Streamlit App:
```bash
streamlit run news_summarizer_app.py
```

### Example
```python
from news_summarizer import NewsArticleSummarizer

summarizer = NewsArticleSummarizer(
    api_key="your_api_key",
    model_type="aimlapi",
    model_name="gpt-3.5-turbo-instruct"
)

result = summarizer.summarize(
    url="https://example.com/news-article",
    summary_type="detailed"
)
```

### Supported Models
- **AIMLAPI**: `gpt-3.5-turbo-instruct`, `gpt-4o-mini`, `gpt-4`
- **AWS Bedrock**: `us.amazon.nova-pro-v1:0`, `anthropic.claude-3-5-sonnet-20241022-v2:0`

---

## 2. YouTube Video Summarizer 🎥

### Description
Downloads YouTube videos, extracts audio, transcribes using Whisper AI, generates summaries, and provides an interactive Q&A interface based on video content.

### Features
- ✅ YouTube video download and audio extraction
- ✅ Audio transcription using OpenAI Whisper
- ✅ AI-powered video summarization
- ✅ Vector store for semantic search
- ✅ Interactive Q&A chat interface
- ✅ Full transcript access
- ✅ Support for multiple LLM and embedding models
- ✅ Streamlit web interface

### Files
- `yt_vid_summarizer.py` - Core video processing class
- `yt_vid_summarizer_app.py` - Streamlit web application

### Usage

#### Command Line:
```bash
python yt_vid_summarizer.py
```

#### Streamlit App:
```bash
streamlit run yt_vid_summarizer_app.py
```

### Example
```python
from yt_vid_summarizer import YoutubeVideoSummarizer

summarizer = YoutubeVideoSummarizer(
    llm_type="bedrock",
    llm_model_name="us.amazon.nova-pro-v1:0",
    embedding_type="bedrock"
)

result = summarizer.process_video("https://www.youtube.com/watch?v=...")

# Get summary
print(result["summary"])

# Ask questions
response = result["qa_chain"].invoke({"question": "What is the main topic?"})
print(response["answer"])
```

### Supported Models

**LLM Models:**
- AWS Bedrock: `us.amazon.nova-pro-v1:0`, `anthropic.claude-3-5-sonnet-20241022-v2:0`, `anthropic.claude-3-haiku-20240307-v1:0`
- AIMLAPI: `gpt-4o-mini`, `gpt-4`

**Embedding Models:**
- AWS Bedrock Titan: `amazon.titan-embed-text-v1`
- AIMLAPI: `text-embedding-ada-002`
- Chroma Default (HuggingFace)

### Processing Steps
1. **Download**: Extracts audio from YouTube video
2. **Transcribe**: Converts audio to text using Whisper
3. **Chunk**: Splits transcript into manageable chunks
4. **Summarize**: Generates summary using map-reduce chain
5. **Vectorize**: Creates embeddings and vector store
6. **Q&A**: Sets up conversational retrieval chain

---

## 3. Voice RAG Assistant 🎤

### Description
A voice-enabled RAG (Retrieval-Augmented Generation) system that processes documents, allows voice queries via microphone, and responds with AI-generated voice answers using ElevenLabs.

### Features
- ✅ Document processing (PDF, TXT, MD)
- ✅ Vector store creation with persistent storage
- ✅ Voice recording from microphone
- ✅ Speech-to-text using Whisper
- ✅ RAG-based question answering
- ✅ Text-to-speech using ElevenLabs
- ✅ Multiple voice options
- ✅ Streamlit web interface

### Files
- `voice_assis.py` - Complete voice assistant application

### Usage

#### Streamlit App:
```bash
streamlit run voice_assis.py
```

### Workflow

1. **Setup Knowledge Base**:
   - Upload documents (PDF, TXT, MD)
   - Documents are processed and chunked
   - Vector store is created and persisted

2. **Voice Assistant**:
   - Record audio query (1-10 seconds)
   - Audio is transcribed to text
   - Query is processed through RAG system
   - Response is generated using AI
   - Response is converted to speech

### Supported Document Types
- PDF files (`.pdf`)
- Text files (`.txt`)
- Markdown files (`.md`)

### Voice Options
- Rachel (default)
- Domi, Bella, Antoni, Elli, Josh, Arnold, Adam, Sam
- Or any voice from your ElevenLabs account

### Example Workflow
```
1. Upload documents → Process → Vector store created
2. Record question: "What is the main topic?"
3. System transcribes → Searches documents → Generates answer
4. Answer is spoken back using selected voice
```

---

## Running Multiple Apps

To run multiple Streamlit apps simultaneously, use different ports:

```bash
# Terminal 1 - News Summarizer (port 8501)
streamlit run news_summarizer_app.py --server.port 8501

# Terminal 2 - YouTube Video Summarizer (port 8502)
streamlit run yt_vid_summarizer_app.py --server.port 8502

# Terminal 3 - Voice Assistant (port 8503)
streamlit run voice_assis.py --server.port 8503
```

Access them at:
- `http://localhost:8501` - News Summarizer
- `http://localhost:8502` - YouTube Video Summarizer
- `http://localhost:8503` - Voice Assistant

---

## Troubleshooting

### Common Issues

#### 1. SSL Certificate Errors
If you encounter SSL certificate errors, add to your `.env`:
```bash
SSL_VERIFY=false
```

#### 2. AWS Bedrock Authentication Errors
- **Error**: `UnrecognizedClientException` or `InvalidClientTokenId`
- **Solution**: 
  - Verify `AWS_BEARER_TOKEN_BEDROCK` or `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` in `.env`
  - Ensure credentials are valid and not expired
  - Check IAM permissions for Bedrock access

#### 3. AIMLAPI Quota Exceeded
- **Error**: `403 - exhausted the available resource limit`
- **Solution**: 
  - Update payment method at [aimlapi.com/app/billing](https://aimlapi.com/app/billing)
  - Or switch to AWS Bedrock as alternative

#### 4. FFmpeg Not Found
- **Error**: `ffmpeg: command not found`
- **Solution**: Install FFmpeg (see Installation section)

#### 5. Module Not Found Errors
- **Error**: `ModuleNotFoundError: No module named 'langchain.xxx'`
- **Solution**: 
  - Ensure virtual environment is activated
  - Run `pip install -r requirements.txt`
  - Check that you're using LangChain 1.0+ compatible imports

#### 6. ElevenLabs API Errors
- **Error**: `'ElevenLabs' object has no attribute 'generate'`
- **Solution**: The code uses `text_to_speech.convert()` method (already fixed)

#### 7. Whisper Model Download
- First run will download Whisper model (~150MB)
- Ensure stable internet connection
- Model is cached for future use

---

## Project Structure

```
week_3/
├── news_summarizer.py          # News summarizer core class
├── news_summarizer_app.py      # News summarizer Streamlit app
├── yt_vid_summarizer.py        # YouTube video summarizer core class
├── yt_vid_summarizer_app.py    # YouTube video summarizer Streamlit app
├── voice_assis.py              # Voice RAG assistant (complete app)
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
├── README.md                   # This file
└── downloads/                  # Created automatically for video downloads
```

---

## Dependencies

Key packages used:
- `langchain` & `langchain-classic` - LLM framework
- `langchain-aws` - AWS Bedrock integration
- `langchain-aimlapi` - AIMLAPI integration
- `streamlit` - Web interface
- `whisper` - Speech-to-text
- `yt-dlp` - YouTube video download
- `newspaper3k` - News article extraction
- `elevenlabs` - Text-to-speech
- `chromadb` - Vector database
- `boto3` - AWS SDK

See `requirements.txt` for complete list.

---

## License

This project is part of the Andela GenAI Bootcamp (October 2025).

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Verify your `.env` file configuration
3. Ensure all dependencies are installed
4. Check API key validity and quotas

---

## Notes

- All apps use session state to maintain context
- Vector stores are persisted for faster subsequent loads
- Audio/video files are cleaned up after processing
- SSL verification can be disabled for corporate networks (development only)

