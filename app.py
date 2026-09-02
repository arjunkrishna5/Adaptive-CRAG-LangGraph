"""
app.py
------
Interactive Streamlit User Interface for the Adaptive Self-Corrective RAG (CRAG) system.

Features:
- Real-time "Agent Thought Process" visualizer using st.status and LangGraph streaming.
- 1-click preset demo query buttons for testing both local retrieval and live web fallback.
- Inspection drawer displaying retrieved source document chunks and metadata.
- Observability and API configuration status indicators.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from src.state import GraphState
from src.graph import crag_app

# Load environment variables
load_dotenv()

# ============================================================================
# 1. Page Configuration & Custom Styling
# ============================================================================

st.set_page_config(
    page_title="Adaptive CRAG | LangGraph",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished interface styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .stStatusWidget {
        border-radius: 8px;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .badge-active { background-color: #DEF7EC; color: #03543F; }
    .badge-inactive { background-color: #FDE8E8; color: #9B1C1C; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# 2. Session State Initialization
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "preset_prompt" not in st.session_state:
    st.session_state.preset_prompt = None


# ============================================================================
# 3. Sidebar: Architecture & 1-Click Demo Buttons
# ============================================================================

with st.sidebar:
    st.header("🧠 System Control Center")
    st.caption("Adaptive Corrective RAG state machine powered by LangGraph.")

    # API Status Check
    st.subheader("🔌 API Health Status")
    groq_key = os.getenv("GROQ_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    tavily_key = os.getenv("TAVILY_API_KEY", "")

    groq_ok = bool(groq_key and not groq_key.startswith("gsk_your_"))
    openai_ok = bool(openai_key and not openai_key.startswith("sk-your_"))
    tavily_ok = bool(tavily_key and not tavily_key.startswith("tvly-your_"))

    llm_status = "Groq (Llama 3.3 70B)" if groq_ok else ("OpenAI (GPT-4o-mini)" if openai_ok else "Not configured")
    search_status = "Tavily Search API" if tavily_ok else "DuckDuckGo Fallback"

    st.markdown(
        f"• **LLM Engine:** `{llm_status}`<br>"
        f"• **Search Fallback:** `{search_status}`<br>"
        f"• **Vector DB:** `ChromaDB (Local MiniLM)`",
        unsafe_allow_html=True,
    )

    st.divider()

    # 1-Click Preset Test Prompts
    st.subheader("🎯 1-Click Test Scenarios")
    st.caption("Click a scenario below to test local retrieval vs. web search fallback:")

    if st.button("🟢 In-Domain Test (Local Nexus-X)", use_container_width=True):
        st.session_state.preset_prompt = "What is the memory requirement for the Nexus-X Protocol?"

    if st.button("🌐 Out-of-Domain Test (Web Fallback)", use_container_width=True):
        st.session_state.preset_prompt = "What are the latest key features announced in Python 3.13?"

    if st.button("📚 Theory Test (Naive RAG Limitations)", use_container_width=True):
        st.session_state.preset_prompt = "What are the three critical failure modes of standard Naive RAG?"

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.preset_prompt = None
        st.rerun()


# ============================================================================
# 4. Main Chat Interface
# ============================================================================

st.markdown('<div class="main-title">🧠 Adaptive Self-Corrective RAG (CRAG)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A self-healing LangGraph state machine featuring document relevance grading, '
    'autonomous query rewriting, web search fallback, and hallucination evaluation.</div>',
    unsafe_allow_html=True,
)

# Display historical conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📄 View Retrieved Source Context"):
                for i, src in enumerate(message["sources"], 1):
                    src_type = src.metadata.get("source", "ChromaDB Chunk")
                    st.markdown(f"**Source #{i} ({src_type}):**")
                    st.code(src.page_content, language="markdown")

# Handle input (either from chat_input or 1-click button)
prompt_input = st.chat_input("Ask a question about the system or any topic...")
prompt_to_run = prompt_input or st.session_state.preset_prompt

if prompt_to_run:
    # Reset preset trigger
    st.session_state.preset_prompt = None

    # Append user prompt
    st.session_state.messages.append({"role": "user", "content": prompt_to_run})
    with st.chat_message("user"):
        st.markdown(prompt_to_run)

    # Agent Execution Container
    with st.chat_message("assistant"):
        final_answer = ""
        retrieved_sources = []

        # Real-time Thought Process Visualizer
        with st.status("🧠 Agent is analyzing and executing workflow...", expanded=True) as status:
            initial_state: GraphState = {
                "question": prompt_to_run,
                "generation": "",
                "web_search_needed": False,
                "documents": [],
                "retry_count": 0,
            }

            try:
                # Stream each node output as it finishes execution
                for output in crag_app.stream(initial_state):
                    for node_name, node_state in output.items():
                        if node_name == "retrieve":
                            docs = node_state.get("documents", [])
                            st.write(f"🔍 **[Retrieve]** Pulled `{len(docs)}` candidate chunks from local ChromaDB.")

                        elif node_name == "grade_documents":
                            docs = node_state.get("documents", [])
                            web_needed = node_state.get("web_search_needed", False)
                            st.write(
                                f"⚖️ **[Grade Documents]** Preserved `{len(docs)}` relevant chunks. "
                                f"Web search needed: **`{web_needed}`**"
                            )

                        elif node_name == "transform_query":
                            new_query = node_state.get("question", "")
                            st.write(f"✍️ **[Transform Query]** Rewrote search terms to: *\"{new_query}\"*")

                        elif node_name == "web_search":
                            st.write("🌐 **[Web Search]** Executed live external web search to augment context.")

                        elif node_name == "generate":
                            final_answer = node_state.get("generation", "")
                            retrieved_sources = node_state.get("documents", [])
                            st.write("📝 **[Generate & Inspect]** Synthesized answer and verified factual grounding.")

                status.update(label="✅ Agent Reasoning Complete!", state="complete", expanded=False)

            except Exception as e:
                status.update(label="⚠️ Workflow Encountered an Issue", state="error", expanded=True)
                final_answer = f"Error during execution: {e}"

        # Display Final Synthesized Answer
        st.markdown(final_answer)

        # Display Retrieved Sources Drawer
        if retrieved_sources:
            with st.expander("📄 View Retrieved Source Context"):
                for i, src in enumerate(retrieved_sources, 1):
                    src_type = src.metadata.get("source", "ChromaDB Chunk")
                    st.markdown(f"**Source #{i} ({src_type}):**")
                    st.code(src.page_content, language="markdown")

        # Save assistant turn to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "sources": retrieved_sources,
        })
