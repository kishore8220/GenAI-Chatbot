# 🤖 GenAI Chatbot with Tools

A smart, multi-functional chatbot powered by **LangChain**, **Gradio**, and **Ollama**, equipped with tool-augmented capabilities to handle web search, summarize YouTube videos and webpages, and even generate `README.md` files from Python code.

---

## 🚀 Features

### 🧠 Chatbot with Tool Integration

An intelligent conversational agent that can:

* 🔍 Perform real-time **web searches** via Tavily API.
* 🕒 Fetch **current IST date and time**.
* 📺 **Summarize YouTube videos** using Whisper transcription + vector database.
* 🌐 **Summarize webpage content** using BeautifulSoup and LLM.

Built using LangChain’s `Conversational ReAct Agent`, enabling it to reason and invoke tools based on the query.

### 📄 README.md Generator

* Upload any `.py` file.
* The model reads your code and generates a **clean and professional `README.md`**.
* Built using LangChain PromptTemplates and `ChatOllama`.

---

## 📦 Tech Stack

| Tech/Tool          | Purpose                                                |
| ------------------ | ------------------------------------------------------ |
| **LangChain**      | Agent framework, tool abstraction, and prompt chaining |
| **Ollama (Qwen3)** | LLM backend                                            |
| **Gradio**         | UI interface with tabbed layout                        |
| **Whisper**        | YouTube video transcription                            |
| **BeautifulSoup**  | Web scraping                                           |
| **Tavily**         | External search engine API                             |

---

## 🧪 How It Works

### 🔧 Tools Initialization

Each tool (YouTube, Web Summary, DateTime, Tavily Search) is wrapped using `langchain.agents.Tool`.

### 🤖 Agent Setup

An agent is initialized with all tools using:

```python
initialize_agent(..., agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION)
```

### 🗂️ Gradio UI

Two functional tabs:

* **Chatbot Tab**: Interactive AI assistant with tool usage.
* **README Generator Tab**: Upload `.py` file → Get generated `README.md`.

---

## 🖥️ Run Locally

```bash
git clone https://github.com/your-username/genai-chatbot.git
cd genai-chatbot
pip install -r requirements.txt
python app.py
```

---

## 📎 Screenshots

> *(You can insert screenshots here after running the UI)*

---

## ✨ Future Ideas

* Add support for PDF and DOC summarization.
* Integrate memory persistence.
* Enable user-uploaded audio summarization.
