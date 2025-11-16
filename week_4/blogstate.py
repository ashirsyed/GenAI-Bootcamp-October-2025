from typing import TypedDict
from pydantic import BaseModel,Field

class Blog(BaseModel):
    title:str=Field(description="the title of the blog post")
    content:str=Field(description="The main content of the blog post")
    audio_url:str=Field(description="The URL of the audio file for the blog post")
    photo_url:str=Field(description="The URL of the photo file for the blog post")

class BlogState(TypedDict):
    topic:str
    blog:Blog
    current_language:str
