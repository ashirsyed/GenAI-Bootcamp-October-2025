import os
from typing import Optional
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_aimlapi import AimlapiLLM

from langchain_core.documents import Document
from newspaper import Article
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()


class NewsArticleSummarizer:
    def __init__(
        self,
        api_key: str = None,
        model_type: str = "aimlapi",
        model_name: str = "gpt-4o-mini",
    ):
        """
        Initialize the summarizer with choice of model
        Args:
            api_key: AIMLAPI API key (required for AIMLAPI models)
            model_type: 'aimlapi' or 'ollama'
            model_name: specific model name
        """
        self.model_type = model_type
        self.model_name = model_name

        # Setup LLM based on model type
        if model_type == "aimlapi":
            if not api_key:
                raise ValueError("API key is required for AIMLAPI models")
            os.environ["AIMLAPI_API_KEY"] = api_key
            self.llm = AimlapiLLM(
                model=model_name,
                temperature=0,
                max_tokens=2048,
            )
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Initialize text splitter for long articles
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200, length_function=len
        )

    def fetch_article(self, url: str) -> Optional[Article]:
        """
        Fetch article content using newspaper3k
        """
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article
        except Exception as e:
            print(f"Error fetching article: {e}")
            return None

    def create_documents(self, text: str) -> list[Document]:
        """
        Create LangChain documents from text
        Returns a list of Document objects split according to chunk_size and chunk_overlap
        """
        texts = self.text_splitter.split_text(text)
        docs = [Document(page_content=t) for t in texts]
        return docs
    
    def get_document_count(self, text: str) -> int:
        """
        Get the number of documents that would be created from a given text
        Useful for checking how many chunks an article will be split into
        """
        docs = self.create_documents(text)
        return len(docs)

    def summarize(self, url: str, summary_type: str = "detailed") -> dict:
        """
        Main summarization pipeline
        """
        # Fetch article
        article = self.fetch_article(url)
        if not article:
            return {"error": "Failed to fetch article"}

        # Create documents
        docs = self.create_documents(article.text)
        print(f"Article split into {len(docs)} document(s) for processing")

        # Define prompts based on summary type
        if summary_type == "detailed":
            map_prompt_template = """Write a detailed summary of the following text:
            "{text}"
            DETAILED SUMMARY:"""

            combine_prompt_template = """Write a detailed summary of the following text that combines the previous summaries:
            "{text}"
            FINAL DETAILED SUMMARY:"""
        else:  # concise summary
            map_prompt_template = """Write a concise summary of the following text:
            "{text}"
            CONCISE SUMMARY:"""

            combine_prompt_template = """Write a concise summary of the following text that combines the previous summaries:
            "{text}"
            FINAL CONCISE SUMMARY:"""

        # Create prompts
        map_prompt = PromptTemplate(
            template=map_prompt_template, input_variables=["text"]
        )

        combine_prompt = PromptTemplate(
            template=combine_prompt_template, input_variables=["text"]
        )

        # Create and run chain
        chain = load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True,
        )

        # Generate summary
        summary = chain.invoke(docs)

        return {
            "title": article.title,
            "authors": article.authors,
            "publish_date": article.publish_date,
            "summary": summary,
            "url": url,
            "model_info": {"type": self.model_type, "name": self.model_name},
        }


def main(url: str):
    # Example of using both models
    if not url:
        raise ValueError("URL is required")

    # Initialize both summarizers
    aimlapi_summarizer = NewsArticleSummarizer(
        api_key=os.getenv("AIMLAPI_API_KEY"),
        model_type="aimlapi",
        model_name="gpt-3.5-turbo-instruct",
    )

    # Formula: Number of documents ≈ (Article length - overlap) / (chunk_size - overlap)
    # Fetch article first to calculate document count
    article = aimlapi_summarizer.fetch_article(url)
    if article:
        # Calculate expected number of documents using the formula
        article_length = len(article.text)
        overlap = aimlapi_summarizer.text_splitter._chunk_overlap
        chunk_size = aimlapi_summarizer.text_splitter._chunk_size
        num_documents = (article_length - overlap) / (chunk_size - overlap)
        print(f"\nArticle length: {article_length} characters")
        print(f"Expected number of documents: {num_documents:.1f}")
        print(f"(Chunk size: {chunk_size}, Overlap: {overlap})")
    
    # Get summaries from both models
    print("\nGenerating AIMLAPI Summary...")
    aimlapi_summary = aimlapi_summarizer.summarize(url, summary_type="detailed")

    # Print results
    for summary, model in [(aimlapi_summary, "AIMLAPI")]:
        print(f"\n{model} Summary:")
        print("-" * 50)
        print(f"Title: {summary['title']}")
        print(f"Authors: {', '.join(summary['authors'])}")
        print(f"Published: {summary['publish_date']}")
        print(
            f"Model: {summary['model_info']['type']} - {summary['model_info']['name']}"
        )
        print(f"Summary:\n{summary['summary']}")

        # Print first document content
        print("\nFirst Document Content:")
        print(summary["summary"]["input_documents"][0].page_content)

        print("\nSecond Document Content:")
        print(summary["summary"]["input_documents"][1].page_content)

        # Check how many documents were created
        # input_docs = summary["summary"].get("input_documents", [])
        # num_docs = len(input_docs)
        # print(f"\nNumber of documents created: {num_docs}")
        
        # # Print document contents (safely handle any number of documents)
        # if num_docs > 0:
        #     print("\nDocument Contents:")
        #     print("-" * 50)
        #     for i, doc in enumerate(input_docs, 1):
        #         print(f"\nDocument {i} ({len(doc.page_content)} characters):")
        #         print(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
        # else:
        #     print("\nNo documents found in summary output.")


if __name__ == "__main__":
    url = "https://www.artificialintelligence-news.com/news/us-china-ai-chip-race-cambricons-first-profit-lands/"
    main(url)
