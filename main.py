import streamlit as st
from google.generativeai import configure, GenerativeModel
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyBymOuraxBaTfeWM5dAGLldWNUq7bEm5oY"
configure(api_key=GEMINI_API_KEY)
gemini_model = GenerativeModel("gemini-1.5-flash")

# Function to extract transcript from YouTube
def get_youtube_transcript(video_url):
    try:
        video_id = video_url.split("v=")[1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry["text"] for entry in transcript])
    except Exception as e:
        return f"Error fetching transcript: {e}"

# In-memory storage for transcript embeddings
transcript_store = []

def store_transcript(transcript):
    """Store transcript in an in-memory list"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_text(transcript)
    transcript_store.extend(split_docs)  # Store transcripts in memory
    return split_docs

def generate_blog(video_url):
    transcript = get_youtube_transcript(video_url)
    if "Error" in transcript:
        return transcript
    
    store_transcript(transcript)
    
    # Generate summary using Gemini
    summary_prompt = f"Summarize the key points from the following transcript: {transcript}"
    summary = gemini_model.generate_content(summary_prompt).text
    
    # Generate blog post based on summary
    blog_prompt = f"Write a detailed blog post based on the following summary: {summary}"
    blog_content = gemini_model.generate_content(blog_prompt).text
    
    return blog_content

# Streamlit UI
st.title("📹 YouTube Blog Writer with LangChain & Gemini")
video_url = st.text_input("Enter YouTube Video URL:")

if st.button("Generate Blog"):
    if "youtube.com" in video_url or "youtu.be" in video_url:
        with st.spinner("Fetching transcript..."):
            blog_content = generate_blog(video_url)
        
        if "Error" in blog_content:
            st.error(blog_content)
        else:
            st.success("✅ Blog Generated Successfully!")
            st.text_area("Generated Blog Content:", blog_content, height=300)
            st.download_button("Download Blog", blog_content, file_name="generated_blog.md", mime="text/markdown")
    else:
        st.error("❌ Please enter a valid YouTube URL!")
