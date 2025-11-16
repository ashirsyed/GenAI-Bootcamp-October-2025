from graph_builder import GraphBuilder
from langchain_groq import ChatGroq
from appconfig import app_config

import os

class AgenticBlogWriter:
    """
    A class to represent the agentic blog writer
    
    Attributes:
        llm: The language model to use for the API
        graph_builder: The graph builder to use for the API
    """
    def __init__(self, groq_api_key: str=None, model: str="llama-3.1-8b-instant"):
        """
        Initialize the AgenticBlogWriter class
        Args:
            groq_api_key: The API key for Groq (optional, will use env variable if not provided)
            model: The model to use for the API
        """
        if groq_api_key:
            app_config.groq_api_key=groq_api_key
        else:
            app_config.groq_api_key=os.getenv("GROQ_API_KEY")

        self.llm=ChatGroq(api_key=app_config.groq_api_key,model=model)
        self.graph_builder=GraphBuilder(self.llm)

    def write_blog(self, topic: str):
        """
        Write a blog based on the topic
        Args:
            topic: The topic of the blog
        """
        graph=self.graph_builder.setup_graph(usecase="topic")
        state=graph.invoke({"topic":topic})
        return state


if __name__=="__main__":
    agentic_blog_writer=AgenticBlogWriter()
    topic="AI and Machine Learning"
    state=agentic_blog_writer.write_blog(topic)
    print(f"Blog State: {state}")
