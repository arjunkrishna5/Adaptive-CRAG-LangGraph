# 🧠 Adaptive Self-Corrective RAG (CRAG) with LangGraph

An advanced Agentic Retrieval-Augmented Generation (RAG) system built with **LangGraph**, featuring:
- **Vector Search** with local ChromaDB & HuggingFace embeddings
- **Document Relevance Grading** to filter irrelevant context
- **Autonomous Query Rewriting & Web Search Fallback** (via Tavily / DuckDuckGo)
- **Self-Reflective Hallucination Checking** before answering
- **Interactive UI** with real-time thought-process visualization in Streamlit
- **Scientific Evaluation** with Ragas

---

## 🚀 Quick Setup (Phase 0)

### 1. Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Setup Environment Keys
```powershell
Copy-Item .env.example .env
```
