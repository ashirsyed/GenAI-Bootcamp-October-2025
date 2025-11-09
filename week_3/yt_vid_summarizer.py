# pip install yt_dlp
# pip3 install -q git+https://github.com/openai/whisper.git

# Required to install also:
# MacOS (requires https://brew.sh/):
# brew install ffmpeg

# ubuntu
# sudo apt install ffmpeg

# windows
# https://www.hostinger.com/tutorials/how-to-install-ffmpeg

import yt_dlp
import whisper
import os
import ssl
import urllib.request
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

#from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_aimlapi import ChatAimlapi, AimlapiLLM, AimlapiEmbeddings
from langchain_aws import ChatBedrock, BedrockEmbeddings
#from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.documents import Document


from dotenv import load_dotenv

load_dotenv()

# Configure SSL verification to handle certificate issues
# This is useful for corporate networks, proxies, or development environments
# WARNING: Disabling SSL verification reduces security. Use only in trusted environments.
# To enable SSL verification, set SSL_VERIFY=true in your .env file
if os.getenv("SSL_VERIFY", "false").lower() != "true":
    ssl._create_default_https_context = ssl._create_unverified_context
    # Also disable SSL verification warnings for urllib3 (used by requests/httpx)
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except ImportError:
        pass  # urllib3 not available, skip


class EmbeddingModel:
    """Handles different embedding models"""

    def __init__(self, model_type="bedrock"):
        self.model_type = model_type
        if model_type == "aimlapi":
            self.embedding_fn = AimlapiEmbeddings(
                model="text-embedding-ada-002",
                aimlapi_api_key=os.getenv("AIMLAPI_API_KEY"),
            )
        elif model_type == "bedrock":
            # AWS Bedrock embeddings
            # Available models: amazon.titan-embed-text-v1, amazon.titan-embed-text-v2:0
            # Supports both API key (AWS_BEARER_TOKEN_BEDROCK) and traditional credentials
            bedrock_api_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK") or os.getenv("BEDROCK_API_KEY")
            if bedrock_api_key:
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = bedrock_api_key
            self.embedding_fn = BedrockEmbeddings(
                model_id="amazon.titan-embed-text-v1",
                region_name=os.getenv("AWS_REGION", "us-east-1"),
            )
        elif model_type == "chroma":
            from langchain.embeddings import HuggingFaceEmbeddings

            self.embedding_fn = HuggingFaceEmbeddings()
        #elif model_type == "nomic":
            #from langchain.embeddings import OllamaEmbeddings

            #self.embedding_fn = OllamaEmbeddings(
            #    model="nomic-embed-text", base_url="http://localhost:11434"
            #)
        else:
            raise ValueError(f"Unsupported embedding type: {model_type}")


class LLMModel:
    """Handles different LLM models"""

    def __init__(self, model_type="bedrock", model_name="us.amazon.nova-pro-v1:0"):
        self.model_type = model_type
        self.model_name = model_name

        if model_type == "aimlapi":
            if not os.getenv("AIMLAPI_API_KEY"):
                raise ValueError("AIMLAPI API key is required for AIMLAPI models")
            self.llm = ChatAimlapi(model_name=model_name, temperature=0)
        
        elif model_type == "bedrock":
            model_name = "us.amazon.nova-pro-v1:0"
            aws_region = os.getenv("AWS_REGION", "us-east-1")
            
            # Check for API key first (simpler authentication method)
            bedrock_api_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK") or os.getenv("BEDROCK_API_KEY")
            
            if bedrock_api_key:
                # Use API key authentication (newer, simpler method)
                print("✓ Using AWS Bedrock API key authentication")
                # Set the bearer token for boto3/bedrock SDK
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = bedrock_api_key
                self.llm = ChatBedrock(
                    model_id=model_name,
                    region_name=aws_region,
                    model_kwargs={"temperature": 0},
                )
            else:
                # Fall back to traditional AWS credentials
                aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
                aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
                
                if not aws_access_key or not aws_secret_key:
                    raise ValueError("AWS credentials are required for Bedrock models")
                
                # Credentials are automatically used by boto3 from environment variables
                self.llm = ChatBedrock(
                    model_id=model_name,
                    region_name=aws_region,
                    model_kwargs={"temperature": 0},
                )
        
        elif model_type == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OpenAI API key is required for OpenAI models")
            self.llm = ChatAimlapi(model_name=model_name, temperature=0)
        #elif model_type == "ollama":
            #self.llm = ChatOllama(
                #model=model_name,
                #temperature=0,
                #format="json",
                #timeout=120,
            #)
        else:
            raise ValueError(f"Unsupported LLM type: {model_type}")


class YoutubeVideoSummarizer:
    def __init__(
        self, llm_type="aimlapi", llm_model_name="gpt-4o-mini", embedding_type="aimlapi"
    ):
        """Initialize with different LLM and embedding options"""
        # Initialize Models
        self.embedding_model = EmbeddingModel(embedding_type)
        self.llm_model = LLMModel(llm_type, llm_model_name)

        # Initialize Whisper
        self.whisper_model = whisper.load_model("base")

    def get_model_info(self) -> Dict:
        """Return current model configuration"""
        return {
            "llm_type": self.llm_model.model_type,
            "llm_model": self.llm_model.model_name,
            "embedding_type": self.embedding_model.model_type,
        }

    def download_video(self, url: str) -> tuple[str, str]:
        """Download video and extract audio"""
        print("Downloading video...")
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": "downloads/%(title)s.%(ext)s",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_path = ydl.prepare_filename(info).replace(".webm", ".mp3")
            video_title = info.get("title", "Unknown Title")
            return audio_path, video_title

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper"""
        print("Transcribing audio...")
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]

    def create_documents(self, text: str, video_title: str) -> List[Document]:
        """Split text into chunks and create Document objects"""
        print("Creating documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""]
        )
        texts = text_splitter.split_text(text)
        return [
            Document(page_content=chunk, metadata={"source": video_title})
            for chunk in texts
        ]

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create vector store from documents"""
        print(
            f"Creating vector store using {self.embedding_model.model_type} embeddings..."
        )

        # Create vector store using LangChain's interface
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model.embedding_fn,
            collection_name=f"youtube_summary_{self.embedding_model.model_type}",
        )

    def generate_summary(self, documents: List[Document]) -> str:
        """Generate summary using LangChain's summarize chain"""
        print("Generating summary...")
        map_prompt = ChatPromptTemplate.from_template(
            """Write a concise summary of the following transcript section:
            "{text}"
            CONCISE SUMMARY:"""
        )

        combine_prompt = ChatPromptTemplate.from_template(
            """Write a detailed summary of the following video transcript sections:
            "{text}"
            
            Include:
            - Main topics and key points
            - Important details and examples
            - Any conclusions or call to action
            
            DETAILED SUMMARY:"""
        )

        print(f"\n\n\nModel: {self.llm_model.llm}\n\n\n")
        #print("\n\n\nWORKING HERE 1\n\n\n")

        summary_chain = load_summarize_chain(
            llm=self.llm_model.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True,
        )
        #print("\n\n\nWORKING HERE 2\n\n\n")
        return summary_chain.invoke(documents)

    def setup_qa_chain(self, vector_store: Chroma):
        """Set up question-answering chain"""
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm_model.llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            verbose=True,
        )

    def process_video(self, url: str) -> Dict:
        """Process video and return summary and QA chain"""
        try:
            # Create downloads directory if it doesn't exist
            os.makedirs("downloads", exist_ok=True)

            # Download and process
            audio_path, video_title = self.download_video(url)
            transcript = self.transcribe_audio(audio_path)
            documents = self.create_documents(transcript, video_title)
            summary = self.generate_summary(documents)
            print("\n\n\nWORKING HERE 3\n\n\n")
            vector_store = self.create_vector_store(documents)
            qa_chain = self.setup_qa_chain(vector_store)

            # Clean up
            os.remove(audio_path)

            return {
                "summary": summary,
                "qa_chain": qa_chain,
                "title": video_title,
                "full_transcript": transcript,
            }
        except Exception as e:
            error_msg = str(e)
            print(f"Error processing video: {error_msg}")
            return None

def main():
    # use these urls for testing
    urls = [
        "https://www.youtube.com/watch?v=v48gJFQvE1Y&ab_channel=BrockMesarich%7CAIforNonTechies",
        "https://www.youtube.com/watch?v=XwZkNaTYBQI&ab_channel=TheGadgetGameShow%3AWhatTheHeckIsThat%3F%21",
    ]
    # Get model preferences
    print("\nAvailable LLM Models:")
    print("1. AIMLAPI GPT-4o-mini")
    print("2. AWS Bedrock")
    #print("3. Ollama Llama3.2")
    #llm_choice = input("Choose LLM model (1/2/3): ").strip()
    llm_choice = "2"

    print("\nAvailable Embeddings:")
    print("1. AIMLAPI")
    print("2. AWS Bedrock (Titan)")
    print("3. Chroma Default")
    #print("4. Nomic (via Ollama)")
    #embedding_choice = input("Choose embeddings (1/2/3/4): ").strip()
    embedding_choice = "2"

    # Configure model settings
    if llm_choice == "1":
        llm_type = "aimlapi"
        llm_model_name = "gpt-4o-mini"
    elif llm_choice == "2":
        llm_type = "bedrock"
        llm_model_name = "us.amazon.nova-pro-v1:0"
    else:
        llm_type = "aimlapi"
        llm_model_name = "gpt-4o-mini"

    if embedding_choice == "1":
        embedding_type = "aimlapi"
    elif embedding_choice == "2":
        embedding_type = "bedrock"
    elif embedding_choice == "3":
        embedding_type = "chroma"
    else:
        embedding_type = "aimlapi"  # default to aimlapi

    try:
        # Verify AWS credentials are loaded if using Bedrock
        if llm_type == "bedrock" or embedding_type == "bedrock":
            aws_region = os.getenv("AWS_REGION", "us-east-1")
            bedrock_api_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK") or os.getenv("BEDROCK_API_KEY")
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            
            if bedrock_api_key:
                print(f"\n✓ AWS Bedrock API key loaded from .env file")
                print(f"  Region: {aws_region}")
                print(f"  API Key: {bedrock_api_key[:10]}...{bedrock_api_key[-4:]}")
            elif aws_access_key and aws_secret_key:
                print(f"\n✓ AWS credentials loaded from .env file")
                print(f"  Region: {aws_region}")
                print(f"  Access Key ID: {aws_access_key[:10]}...{aws_access_key[-4:]}")
            else:
                print("\n⚠ Warning: AWS Bedrock authentication not found in .env file")
        
        # Initialize summarizer
        summarizer = YoutubeVideoSummarizer(
            llm_type=llm_type,
            llm_model_name=llm_model_name,
            embedding_type=embedding_type,
        )

        # Display configuration
        model_info = summarizer.get_model_info()
        print("\nCurrent Configuration:")
        print(f"LLM: {model_info['llm_type']} ({model_info['llm_model']})")
        print(f"Embeddings: {model_info['embedding_type']}")

        # Process video
        #url = input("\nEnter YouTube URL: ")
        url = "https://www.youtube.com/watch?v=v48gJFQvE1Y&ab_channel=BrockMesarich%7CAIforNonTechies"
        print(f"\nProcessing video...: {url}")
        result = summarizer.process_video(url)

        if result:
            print(f"\nVideo Title: {result['title']}")
            print("\nSummary:")
            print(result["summary"])

            # Interactive Q&A
            print("\nYou can now ask questions about the video (type 'quit' to exit)")
            while True:
                query = input("\nYour question: ").strip()
                if query.lower() == "quit":
                    break
                if query:
                    response = result["qa_chain"].invoke({"question": query})
                    print("\nAnswer:", response["answer"])

            # Option to see full transcript
            if input("\nWant to see the full transcript? (y/n): ").lower() == "y":
                print("\nFull Transcript:")
                print(result["full_transcript"])

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure required models and APIs are properly configured.")


if __name__ == "__main__":
    main()
