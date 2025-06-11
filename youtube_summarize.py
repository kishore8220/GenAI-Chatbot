import os
import whisper
import tempfile
import shutil
import subprocess
import gradio as gr
import textwrap

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv

load_dotenv()

# --- UTILITIES ---
def is_tool_installed(name):
    return shutil.which(name) is not None


def download_youtube_audio(url, output_dir):
    if not is_tool_installed("yt-dlp"):
        raise EnvironmentError("yt-dlp is not installed.")
    if not is_tool_installed("ffmpeg"):
        raise EnvironmentError("ffmpeg is not installed.")

    audio_path = os.path.join(output_dir, "audio.%(ext)s")
    subprocess.run([
        "yt-dlp",
        "-x", "--audio-format", "mp3",
        "-o", audio_path,
        url
    ], check=True)

    for file in os.listdir(output_dir):
        if file.startswith("audio.") and (file.endswith(".mp3") or file.endswith(".m4a")):
            return os.path.join(output_dir, file)

    raise FileNotFoundError("Audio not found.")


def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]


def create_vector_db(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = splitter.create_documents([text])
    db = FAISS.from_documents(docs, OllamaEmbeddings(model="nomic-embed-text"))
    return db


def get_summary(db, query):
    docs = db.similarity_search(query, k=4)
    context = " ".join([doc.page_content for doc in docs])
    chat = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("Use only this transcript: {docs}"),
        HumanMessagePromptTemplate.from_template("Answer: {question}")
    ])

    chain = LLMChain(llm=chat, prompt=prompt)
    return chain.run(question=query, docs=context)


# --- GRADIO INTERFACE ---
def summarize_youtube_video(url):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = download_youtube_audio(url, tmpdir)
            transcript = transcribe_audio(audio_path)
            db = create_vector_db(transcript)
            summary = get_summary(db, "Summarize this video in detail")

            return (
                textwrap.fill(transcript, width=100),
                textwrap.fill(summary, width=100)
            )

    except Exception as e:
        return f"[ERROR] {e}", ""


iface = gr.Interface(
    fn=summarize_youtube_video,
    inputs=gr.Textbox(label="Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=xxxx"),
    outputs=[
        gr.Textbox(label="Transcript"),
        gr.Textbox(label="Detailed Summary"),
    ],
    title="🎬 YouTube Video Summarizer",
    description="Extracts audio from YouTube, transcribes it using Whisper, and summarizes with LLM.",
    theme="soft",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(share=True)
