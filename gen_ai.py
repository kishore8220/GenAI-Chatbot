import os
import uuid
import re
import gradio as gr
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo
from youtube_summarize import download_youtube_audio, transcribe_audio, create_vector_db, get_summary

from langchain_ollama import ChatOllama
from langchain.agents import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langgraph.graph import add_messages, StateGraph, END
from langgraph.prebuilt import ToolNode

from bs4 import BeautifulSoup
import requests
import textwrap
import tempfile
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ------------------- Pinecone Setup ------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
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
    query_vector = embedder.encode("dummy query").tolist()
    results = index.query(
        vector=query_vector,
        top_k=50,
        filter={"thread_id": {"$eq": thread_id}},
        include_metadata=True
    )
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

class BasicChatbot(dict):
    messages: list

def get_system_date_and_time(_input: str) -> str:
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    return f"🕒 Current Date & Time (IST): {now.strftime('%Y-%m-%d %I:%M:%S %p')}"

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

date_time_tool = Tool.from_function(
    name="get_system_date_and_time",
    description="Returns the current system date and time in IST",
    func=get_system_date_and_time
)
youtube_tool = Tool.from_function(
    name="summarize_youtube_video",
    description="Summarizes a YouTube video from its URL using Whisper and LLM. Input must be a YouTube URL.",
    func=youtube_summarizer_tool
)
summarize_web_tool = Tool.from_function(
    name="summarize_web_url",
    description="Summarizes content of any given webpage URL (HTML articles, docs, blogs). Input must be a valid URL.",
    func=summarize_web_url
)
tavily_api_key = os.environ.get("TAVILY_API_KEY")
tavily_tool = TavilySearchResults(api_key=tavily_api_key)

llm = ChatOllama(model="qwen3")
tools = [date_time_tool, tavily_tool, youtube_tool, summarize_web_tool]
llm_tools = llm.bind_tools(tools)

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

def gr_chat(message, history, thread_id):
    past_messages = fetch_context_from_pinecone(thread_id)
    past_messages.append({"role": "user", "content": message})
    state = {"messages": past_messages}
    
    result = app.invoke(state)
    response_msg = result["messages"][-1]
    clean_content = re.sub(r"<think>.*?</think>", "", response_msg.content, flags=re.DOTALL).strip()
    
    save_to_pinecone(message, clean_content, thread_id)

    # ✅ Only return the assistant response
    return clean_content

# ------------------- README Generator Chain ---------------------------
def generate_readme_from_code(file):
    try:
        if not file.name.endswith(".py"):
            return "[ERROR] Only .py files are supported.", "", None

        with open(file.name, "r", encoding="utf-8") as f:
            code = f.read()

        prompt_template = PromptTemplate(
            input_variables=["code"],
            template="""You are a helpful assistant. Read the following Python code and generate a professional, clean, and informative README.md content for it.

Python Code:
{code}

README:
""")
        chain = prompt_template | llm
        readme = chain.invoke({"code": code})

        if hasattr(readme, "content"):
            readme = readme.content

        # Save to temp .md file
        readme_file = tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode='w', encoding='utf-8')
        readme_file.write(readme)
        readme_file.close()

        return code, readme, readme_file.name

    except Exception as e:
        return f"[ERROR] {str(e)}", "", None





def new_session():
    return str(uuid.uuid4())


chat_ui = gr.ChatInterface(
    fn=gr_chat,
    additional_inputs=[
        gr.Textbox(value=new_session, label="Thread ID")
    ],
    title="🔗 Tool-Enabled Chatbot",
    description="Chat with a tool-enhanced LLM agent.",
)

readme_tab = gr.Interface(
    fn=generate_readme_from_code,
    inputs=gr.File(label="📎 Upload a Python (.py) file", file_types=[".py"]),
    outputs=[
        gr.Code(label="📄 Uploaded Python Code", language="python"),
        gr.Markdown(label="📘 Generated README.md"),
        gr.File(label="📥 Download README.md")
    ],
    title="📄 README Generator",
    description="Upload a `.py` file to generate a professional README.md"
)

demo = gr.TabbedInterface(
    interface_list=[chat_ui, readme_tab],
    tab_names=["💬 Chatbot", "📄 README Generator"]
)

if __name__ == "__main__":
    demo.launch()
