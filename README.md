# 🔗 Tool-Enhanced LLM Chatbot & 📄 README Generator

This project combines the power of a **LangGraph-based conversational AI** with tool-augmented capabilities such as YouTube summarization, web article summarization, date/time reporting, and web search. It also includes a utility to **auto-generate professional `README.md` files** from uploaded Python source code.

---

## 🌟 Features

### 💬 Tool-Enhanced Chatbot
- Powered by `LangGraph`, `LangChain`, and `Ollama` (LLM model: `qwen3`)
- Integrates useful tools:
  - ⏰ **Date/Time Tool** – Returns IST time
  - 🔍 **Tavily Search Tool** – Real-time web search using Tavily API
  - 📺 **YouTube Summarizer** – Summarizes YouTube videos using Whisper + LLM
  - 🌐 **Web URL Summarizer** – Extracts and summarizes content from web pages
- Uses Pinecone for **chat memory persistence**
- Chat state managed using `StateGraph` with tool routing
- Responsive UI using `Gradio`

### 📄 README Generator
- Upload a `.py` file to generate a well-structured and professional `README.md`
- Uses prompt chaining with `LangChain` to extract useful documentation from code
- Provides downloadable markdown output

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/kishore8220/GenAI-Chatbot.git
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file with the following content:
```env
PINECONE_API_KEY=your-pinecone-api-key
TAVILY_API_KEY=your-tavily-api-key
```

---

## ▶️ Run the App
```bash
python gen_ai.py
```
This will launch a `Gradio` UI with two tabs:
- 💬 Chatbot
- 📄 README Generator

---

## 🧠 Technologies Used

- **LangChain**: For prompt chaining, tools, and agents
- **LangGraph**: For building conversational workflows
- **Ollama (ChatOllama)**: For hosting and running the LLM
- **Pinecone**: For persistent memory embedding storage
- **SentenceTransformers**: For vector encoding (`MiniLM-L6-v2`)
- **Tavily Search**: For real-time information retrieval
- **Gradio**: Web UI framework
- **Whisper**: For YouTube video transcription
- **BeautifulSoup + Requests**: For web content extraction

---

## 🧪 Example Use Cases

- Ask for the current IST time
- Paste a YouTube URL and get a detailed summary
- Input a blog or article link and receive a concise summary
- Upload a Python file and receive a ready-to-use `README.md` file

---

## 📂 File Structure

```bash
.
├── gen_ai.py                      # Main application file
├── .env                        # API keys for Pinecone and Tavily
├── requirements.txt            # Python dependencies
├── youtube_summarize.py       # Helper module for YouTube audio transcription and vector DB
└── ...
```

---

## 🔐 Security Notes
- Ensure `.env` is **never** checked into version control
- Use a secure key management strategy for production deployments

---

## 📜 License

This project is open-source and available under the [Apache 2.0](LICENSE).
