"""
Streamlit app to interact with the Groq API and OpenAI API
"""

from groq import Groq
from openai import OpenAI
from appconfig import app_config

class LLMApp:

    def __init__(self, groq_api_key: str=None, model: str="llama-3.3-70b-versatile", temperature: float=0.5, max_tokens: int=1024, system_prompt: str=None, chatbot_name: str="Ally"):

        
        """
        Initialize the LLMApp class
    
        Args:
            groq_api_key (str): The API key for Groq (optional, will use env variable if not provided)
            model (str): The model to use for the API
            chatbot_name (str): The name of the chatbot
        """

        # Initialize API keys
        self.groq_api_key = groq_api_key or app_config.groq_api_key
        self.openai_api_key = app_config.openai_api_key
        
        # Determine which provider to use based on model
        self.model = model
        self.use_openai = self._should_use_openai(self.model)

        # Store configuration
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.chatbot_name = chatbot_name
        self.conversation_history = []
        
        # Validate and initialize clients
        if self.use_openai:
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY in your .env file")
            self.client = OpenAI(api_key=self.openai_api_key)
            self.provider = "OpenAI"
        else:
            if not self.groq_api_key:
                raise ValueError("Groq API key is required. Please provide groq_api_key parameter or set GROQ_API_KEY in your .env file")
            self.client = Groq(api_key=self.groq_api_key)
            self.provider = "Groq"
    
    
    def _should_use_openai(self, model: str) -> bool:
        """
        Determine if the model should use OpenAI API
        
        Args:
            model (str): The model name
            
        Returns:
            bool: True if OpenAI should be used, False for Groq
        """
        openai_models = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4", "gpt-4-turbo", 
                        "gpt-4o", "gpt-3.5-turbo"]
        return any(model.lower().startswith(m.lower()) for m in openai_models)


    def chat(self, user_message, model: str=None, system_prompt: str=None, temperature: float=None, max_tokens: int=None):
        """
        Chat with the API
        
        Args:
            user_message (str): The message from the user
            model (str): The model to use (optional, defaults to current model)
            system_prompt (str): The system prompt for the API (optional, uses instance default if not provided)
            temperature (float): The temperature for the API (optional, uses instance default if not provided)
            max_tokens (int): The maximum number of tokens for the API (optional, uses instance default if not provided)
        
        Returns:
            str: The response from the API
        """
        
        # Update model if provided and different from current
        if model and model != self.model:
            self.set_model(model)
        
        # Update temperature and max_tokens if provided
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
        
        # Use instance defaults if not provided
        effective_temp = temperature if temperature is not None else self.temperature
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        effective_system_prompt = system_prompt if system_prompt is not None else self.system_prompt
        
        # Determine the system prompt to use
        if not effective_system_prompt:
            prompt = f"""You are {self.chatbot_name}, an intelligent and friendly AI assistant with a warm and approachable personality. 

                        Your key traits:
                        - You're genuinely curious and love learning about the world
                        - You communicate in a conversational, easy-to-understand manner
                        - You're enthusiastic about helping users achieve their goals
                        - You have a positive attitude and can make complex topics accessible
                        - You remember context from the conversation to provide coherent responses

                        Always be helpful, honest, and engaging. If you don't know something, admit it gracefully. Remember, your name is {self.chatbot_name}, so introduce yourself naturally when appropriate.
                        """
        else:
            prompt = effective_system_prompt
        
        messages = []

        # Add system prompt 
        messages.append(
            {
                "role": "system",
                "content": prompt
            }
        )

        # Add conversation history
        if self.conversation_history:
            messages.extend(self.conversation_history)

        # Add current user message
        user_message_dict = {
            "role": "user",
            "content": f"{user_message}"
        }
        messages.append(user_message_dict)

        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=effective_temp,
            max_tokens=effective_max_tokens
        )

        # Extract response
        response_text = response.choices[0].message.content

        # Add BOTH user message and assistant response to conversation history
        self.conversation_history.append(user_message_dict)
        self.conversation_history.append(
            {
                "role": "assistant",
                "content": f"{response_text}"
            }
        )

        return response_text

    def clear_conversation_history(self):
        """
        Clear the conversation history
        """
        
        self.conversation_history = []

    def get_conversation_history(self):
        """
        Get the conversation history
        """
        
        return self.conversation_history
    
    def get_provider_info(self):
        """
        Get information about the current provider and model
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "chatbot_name": self.chatbot_name
        }
    
    
    def set_model(self, model: str):
        use_openai = self._should_use_openai(model)
        old_provider = self.provider
        self.model = model
        self.use_openai = use_openai
        if use_openai:
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required for this model")
            self.client = OpenAI(api_key=self.openai_api_key)
            self.provider = "OpenAI"
        else:
            if not self.groq_api_key:
                raise ValueError("Groq API key is required for this model")
            self.client = Groq(api_key=self.groq_api_key)
            self.provider = "Groq"
        if old_provider != self.provider:
            print(f"\n[Switched from {old_provider} to {self.provider} for model: {model}]")


if __name__ == "__main__":
    """
    Main function to initialize the app
    """
    
    # Initialize LLMApp
    llm_app = LLMApp()

    # Create a text input for the user to enter their message
    provider_info = llm_app.get_provider_info()
    print(f"\n{'='*60}")
    print(f"Hello! I'm {provider_info['chatbot_name']}, your AI assistant.")
    print(f"Provider: {provider_info['provider']}")
    print(f"Model: {provider_info['model']}")
    print(f"{'='*60}")
    print("\nHow can I help you today? (Type 'exit' or 'quit' to leave)\n")
    
    while True:
        user_message = input("You: ")
        if user_message.lower() == "exit" or user_message.lower() == "quit":
            print(f"\n{llm_app.chatbot_name}: Goodbye! It was great chatting with you. Have a wonderful day!")
            break

        # Generate response
        response = llm_app.chat(user_message, model=llm_app.model)
        print(f"\n{llm_app.chatbot_name}: {response}\n")

    print(f"Conversation history: {llm_app.get_conversation_history()}")
        
        