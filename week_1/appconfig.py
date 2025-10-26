import os
from dotenv import load_dotenv

# Load Env Var from .env file
load_dotenv(override=True)

class AppConfig():

    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

# Instantiate environment variables
app_config = AppConfig()