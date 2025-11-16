from blogstate import BlogState
import re
import requests
import os
import json
import base64
from datetime import datetime
from appconfig import app_config
import boto3


class BlogNode:
    """
    A class to represent the blog node
    """

    def __init__(self,llm):
        self.llm=llm

    
    def title_creation(self,state:BlogState):
        """
        create the title for the blog
        """

        if "topic" in state and state["topic"]:
            prompt="""
                   You are an expert blog content writer. Use Markdown formatting. Generate
                   a blog title for the {topic}. This title should be creative and SEO friendly.
                    Only return one line of the title, no other text or suggestions.
                   """
            
            sytem_message=prompt.format(topic=state["topic"])
            print(sytem_message)
            response=self.llm.invoke(sytem_message)
            #print(f"\n\n\n title Generation Response: {response.content}\n\n\n")
            return {"blog":{"title":response.content}}
        
    def content_generation(self,state:BlogState):
        if "topic" in state and state["topic"]:
            system_prompt = """You are expert blog writer. Use Markdown formatting.
            Generate a detailed blog content with detailed breakdown for the {topic} for a blog post of 100 words."""
            system_message = system_prompt.format(topic=state["topic"])
            response = self.llm.invoke(system_message)
            #print(f"\n\n\nContent Generation Response: {response.content}\n\n\n")
            return {"blog": {"title": state['blog']['title'], "content": response.content}}

    def audio_generation(self,state:BlogState):
        """
        Generate audio from the blog content using ElevenLabs (if API key available)
        Uses content from content_generation output
        """
        if "blog" in state and state["blog"] and "content" in state["blog"]:
            content = state["blog"]["content"]
            
            # Clean content - remove markdown formatting for better TTS
            # Remove markdown headers, bold, italic, links
            clean_content = re.sub(r'#+\s*', '', content)  # Remove headers
            clean_content = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_content)  # Remove bold
            clean_content = re.sub(r'\*([^*]+)\*', r'\1', clean_content)  # Remove italic
            clean_content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean_content)  # Remove links
            clean_content = re.sub(r'[^a-zA-Z0-9\s]', '', clean_content) # remove any special characters
            #clean_content = clean_content[:5000]  # Limit to 5000 chars for gTTS
            
            # Generate timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_filename = f"blog_audio_{timestamp}.mp3"
            audio_path = os.path.join("data", audio_filename)
            
            # Ensure data directory exists
            os.makedirs("data", exist_ok=True)
            
            try:
                # Try ElevenLabs first if API key is available (better quality)
                if app_config.elevenlabs_api_key:
                    url = app_config.elevenlabs_api_url
                    headers = {
                        "Accept": "audio/mpeg",
                        "Content-Type": "application/json",
                        "xi-api-key": app_config.elevenlabs_api_key
                    }
                    data = {
                        "text": clean_content,
                        "model_id": "eleven_monolingual_v1",
                        "voice_settings": {
                            "stability": 0.5,
                            "similarity_boost": 0.5
                        }
                    }
                    response = requests.post(url, json=data, headers=headers)
                    if response.status_code == 200:
                        with open(audio_path, "wb") as f:
                            f.write(response.content)
                        audio_url = os.path.abspath(audio_path)
                        return {"blog": {**state["blog"], "audio_url": audio_url}}
                
            except Exception as e:
                print(f"Error generating audio: {e}")
                
                # Return state without audio_url if generation fails
                return {"blog": state["blog"]}
        
        return {"blog": state.get("blog", {})}
        
    def photo_generation(self,state:BlogState):
        """
        Generate an image using AWS Bedrock Nova Canvas model based on the blog title from title_creation
        """
        if "blog" in state and state["blog"] and "title" in state["blog"]:
            title = state["blog"]["title"]
            
            # Clean title - remove markdown formatting
            clean_title = re.sub(r'#+\s*', '', title)  # Remove headers
            clean_title = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_title)  # Remove bold
            clean_title = re.sub(r'\*([^*]+)\*', r'\1', clean_title)  # Remove italic
            clean_title = re.sub(r'[^a-zA-Z0-9\s]', '', clean_title) # remove any special characters
            clean_title = clean_title.strip()
            
            # Generate timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"blog_image_{timestamp}.png"
            image_path = os.path.join("data", image_filename)
            
            # Ensure data directory exists
            os.makedirs("data", exist_ok=True)
            
            try:
                # Create image prompt based on title
                base_prompt = "Create a professional, high-quality blog post cover image related to: {}. The image should be visually appealing, modern, and relevant to the topic."
                image_prompt = base_prompt.format(clean_title)
                
                # Truncate to 1024 characters if needed
                if len(image_prompt) > 1024:
                    # Reserve space for the base prompt text
                    max_title_length = 1024 - len(base_prompt.format(""))
                    truncated_title = clean_title[:max_title_length].strip()
                    image_prompt = base_prompt.format(truncated_title)
                
                model_id = "amazon.nova-canvas-v1:0"
                
                try:
                    bedrock = boto3.client('bedrock-runtime', region_name=app_config.aws_region)
                    
                    payload = {
                        "taskType": "TEXT_IMAGE",
                        "textToImageParams": {
                            "text": image_prompt
                        }
                    }
                    
                    response = bedrock.invoke_model(
                        modelId=model_id,
                        body=json.dumps(payload),
                        contentType="application/json",
                        accept="application/json"
                    )
                    response_body = json.loads(response['body'].read())
                    
                except (ImportError, Exception) as e:
                    print(f"Error generating image with AWS Bedrock: {e}")
                    return {"blog": state["blog"]}
                                    
                images = response_body.get('images', [])
                if not images or not images[0]:
                    print(f"Error: No image data found in response. Available keys: {list(response_body.keys())}")
                    return {"blog": state["blog"]}
                
                image_data = images[0]          
                image_bytes = base64.b64decode(image_data)   
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                photo_url = os.path.abspath(image_path)
                return {"blog": {**state["blog"], "photo_url": photo_url}}
                    
            except Exception as e:
                print(f"Error generating image with AWS Bedrock: {e}")
    
                # Return state without photo_url if generation fails
                return {"blog": state["blog"]}
        
        return {"blog": state.get("blog", {})}
        