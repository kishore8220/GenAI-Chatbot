import os
import tempfile
import textwrap
from datetime import datetime
from zoneinfo import ZoneInfo

import gradio as gr
import requests
from bs4 import BeautifulSoup

from langchain_ollama import ChatOllama
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import PromptTemplate

from youtube_summarize import download_youtube_audio, transcribe_audio, create_vector_db, get_summary

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

# ------------------- LLM Setup ------------------------
llm = ChatOllama(model="qwen3")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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


# ------------------- Tool: Tavily Search -----------------------------
tavily_api_key = os.environ.get("TAVILY_API_KEY")
tavily_tool = TavilySearchResults(api_key=tavily_api_key)

# ------------------- Agent Setup -------------------------------------
tools = [date_time_tool, tavily_tool, youtube_tool, summarize_web_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

def chatbot_interface(message, history):
    try:
        return agent.run(message)
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ------------------- Gradio UI Setup ---------------------------------
chat_tab = gr.ChatInterface(
    fn=chatbot_interface,
    title="🤖 GenAI Chatbot with Tools",
    description=" Web Search + YouTube & Website summarizer"
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
    interface_list=[chat_tab, readme_tab],
    tab_names=["💬 Chatbot", "📄 README Generator"]
)

if __name__ == "__main__":
    demo.launch()
