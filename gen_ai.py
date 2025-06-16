import os
import tempfile
import textwrap
import uuid
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import TypedDict, Annotated

import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

from langchain_ollama import ChatOllama
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import PromptTemplate
from langgraph.graph import add_messages, StateGraph, END
from langgraph.prebuilt import ToolNode

from youtube_summarize import download_youtube_audio, transcribe_audio, create_vector_db, get_summary
from dotenv import load_dotenv

load_dotenv()

# ------------------- Pinecone Setup ------------------------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "chat-history"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------- Memory Save & Fetch ------------------------
def save_to_pinecone(user_msg, bot_response, thread_id="default_thread"):
    combined_text = f"User: {user_msg}\nBot: {bot_response}"
    embedding = embedder.encode(combined_text).tolist()

    index.upsert([
        {
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {
                "thread_id": thread_id,
                "user_input": user_msg,
                "bot_response": bot_response,
                "timestamp": datetime.now().isoformat()
            }
        }
    ])

def fetch_context_from_pinecone(thread_id: str, top_k: int = 5) -> list:
    # Embed any dummy text to get a valid query vector (required by Pinecone)
    query_vector = embedder.encode("dummy query").tolist()
    
    results = index.query(
        vector=query_vector,
        top_k=50,  # large enough to fetch full history
        filter={"thread_id": {"$eq": thread_id}},
        include_metadata=True
    )

    # Sort by timestamp
    sorted_matches = sorted(
        results.matches,
        key=lambda x: x.metadata["timestamp"]
    )[:top_k]

    messages = []
    for match in sorted_matches:
        meta = match.metadata
        messages.append({"role": "user", "content": meta["user_input"]})
        messages.append({"role": "assistant", "content": meta["bot_response"]})

    return messages


# ------------------- Chat State ------------------------
class BasicChatbot(TypedDict):
    messages: Annotated[list, add_messages]

# ------------------- Tool: Get System Date & Time --------------------
def get_system_date_and_time(_input: str) -> str:
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    return f"🕒 Current Date & Time (IST): {now.strftime('%Y-%m-%d %I:%M:%S %p')}"

date_time_tool = Tool.from_function(
    name="get_system_date_and_time",
    description="Returns the current system date and time in IST",
    func=get_system_date_and_time
)

# ------------------- Tool: YouTube Summarizer ------------------------
def youtube_summarizer_tool(url: str) -> str:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = download_youtube_audio(url, tmpdir)
            transcript = transcribe_audio(audio_path)
            db = create_vector_db(transcript)
            summary = get_summary(db, "Summarize this video in detail")
            return f"📝 Summary:\n{textwrap.fill(summary, width=100)}"
    except Exception as e:
        return f"[ERROR] {e}"

youtube_tool = Tool.from_function(
    name="summarize_youtube_video",
    description="Summarizes a YouTube video from its URL using Whisper and LLM. Input must be a YouTube URL.",
    func=youtube_summarizer_tool
)

# ------------------- Tool: Webpage Summarizer ------------------------
def summarize_web_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        if not text or len(text) < 300:
            return "❌ Unable to extract meaningful content."
        db = create_vector_db(text)
        summary = get_summary(db, "Summarize the webpage in detail")
        return f"🌐 Summary:\n{textwrap.fill(summary, width=100)}"
    except Exception as e:
        return f"[ERROR while summarizing URL] {e}"

summarize_web_tool = Tool.from_function(
    name="summarize_web_url",
    description="Summarizes content of any given webpage URL (HTML articles, docs, blogs). Input must be a valid URL.",
    func=summarize_web_url
)

# ------------------- Tool: Tavily Search -----------------------------
tavily_api_key = os.environ.get("TAVILY_API_KEY")
tavily_tool = TavilySearchResults(api_key=tavily_api_key)

# ------------------- LLM + Tools Setup ------------------------
llm = ChatOllama(model="qwen3")
tools = [date_time_tool, tavily_tool, youtube_tool, summarize_web_tool]
llm_tools = llm.bind_tools(tools)

# ------------------- LangGraph Nodes ------------------------
def chatbot(state: BasicChatbot):
    response = llm_tools.invoke(state["messages"])
    state["messages"].append(response)
    return {"messages": state["messages"]}

def tool_router(state: BasicChatbot):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    else:
        return END

tool_node = ToolNode(tools=tools)

graph = StateGraph(BasicChatbot)
graph.add_node("chatbot", chatbot)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("chatbot")
graph.add_conditional_edges("chatbot", tool_router)
graph.add_edge("tool_node", "chatbot")
app = graph.compile()

# ------------------- CLI Chat Interface ------------------------
def chat_with_bot(message: str, thread_id: str = "default_thread", is_custom_thread: bool = True):
    if message.strip().lower() == "!clear":
        return "✅ Chat history clearing not yet implemented."

    past_messages = fetch_context_from_pinecone(thread_id)
    print("Thread_id:", thread_id)
    past_messages.append({"role": "user", "content": message})
    state = {"messages": past_messages}

    result = app.invoke(state)
    response_msg = result["messages"][-1]
    clean_content = re.sub(r"<think>.*?</think>", "", response_msg.content, flags=re.DOTALL).strip()

    # Only save to Pinecone if not default (i.e. custom or UUID session)
    if is_custom_thread or thread_id != "default_thread":
        save_to_pinecone(message, clean_content, thread_id)

    return clean_content


# ------------------- Entry Point ------------------------
if __name__ == "__main__":
    user_input_thread = input("Enter thread ID (or press Enter for a new session): ").strip()
    thread_id = user_input_thread if user_input_thread else str(uuid.uuid4())
    is_custom_thread = bool(user_input_thread)  # To check later if user supplied a custom ID
    while True:
        user_input = input("👤 You: ")
        if user_input.strip().lower() == "exit":
            print("👋 Exiting chatbot.")
            break
        print("🤖 Bot:", chat_with_bot(user_input, thread_id, is_custom_thread))


