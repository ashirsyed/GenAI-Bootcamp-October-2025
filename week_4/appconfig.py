import os
from dotenv import load_dotenv

# Load Env Var from .env file
load_dotenv(override=True)

class AppConfig():

    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.bedrock_api_key = os.getenv("BEDROCK_API_KEY")
        self.aws_bearer_token = os.getenv("AWS_BEARER_TOKEN")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.bedrock_endpoint = os.getenv("BEDROCK_ENDPOINT")
        self.elevenlabs_api_url = os.getenv("ELEVENLABS_API_URL")
        

# Instantiate environment variables
app_config = AppConfig()