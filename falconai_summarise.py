import os
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline


import os
import time
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser




# Initialize Falconsai summarizer pipeline
summarizer = pipeline("summarization", model="Falconsai/text_summarization")

def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname == "youtu.be":
        return parsed.path[1:]
    elif parsed.hostname in ["www.youtube.com", "youtube.com"]:
        if parsed.path == "/watch":
            return parse_qs(parsed.query)['v'][0]
        elif parsed.path.startswith("/embed/"):
            return parsed.path.split("/")[2]
    raise ValueError("Invalid YouTube URL")

def load_youtube_transcript(url):
    video_id = extract_video_id(url)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    full_text = " ".join([entry['text'] for entry in transcript])
    return [Document(page_content=full_text)]

def load_webpage(url):
    loader = WebBaseLoader(url)
    return loader.load()

def split_chunks(docs, chunk_size=1000, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def summarize_chunks(chunks):
    summaries = []
    for chunk in chunks:
        try:
            summary_result = summarizer(chunk.page_content, max_length=120, min_length=30, do_sample=False)
            summaries.append(summary_result[0]['summary_text'])
        except Exception as e:
            print(f"⚠️ Error summarizing chunk: {e}")
    return " ".join(summaries)

def handle_link(url: str):
    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc or "youtu.be" in parsed.netloc:
        print("🔗 Detected YouTube link.")
        docs = load_youtube_transcript(url)
    else:
        print("🔗 Detected Web/Blog link.")
        docs = load_webpage(url)

    chunks = split_chunks(docs)
    summary = summarize_chunks(chunks)
    return summary

# Example usage
if __name__ == "__main__":
    test_url = "https://www.youtube.com/watch?v=TIFjafya_eQ"  # Replace with your URL
    final_summary = handle_link(test_url)
    print("\n📝 Final Summary:\n", final_summary)
