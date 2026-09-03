"""
nodes.py
--------
Defines the functional action nodes for the Adaptive Corrective RAG (CRAG) workflow:
1. retrieve: Fetches candidate documents from the local ChromaDB vector store.
2. grade_documents: Evaluates retrieved chunks, discards irrelevant context, and flags web fallback.
3. transform_query: Rewrites conversational or failed queries into optimized search keywords.
4. web_search: Queries live external search engines (Tavily with DuckDuckGo fallback).
5. generate: Synthesizes the final grounded answer from curated context.
"""

import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchResults

from src.state import GraphState
from src.vectorstore import get_retriever
from src.evaluators import get_judge_llm, get_retrieval_grader

# Load environment variables
load_dotenv()


# ============================================================================
# 1. Retrieve Node
# ============================================================================

def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieves the top-k most relevant document chunks from the local ChromaDB vector store.

    Args:
        state: The current GraphState containing the user question.

    Returns:
        Dictionary update containing the retrieved documents.
    """
    print("--- [NODE] RETRIEVE: Querying local ChromaDB vector store ---", flush=True)
    question = state["question"]
    retriever = get_retriever(k=3)
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


# ============================================================================
# 2. Grade Documents Node
# ============================================================================

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Evaluates each retrieved document chunk against the user question using
    the structured Retrieval Grader LLM judge. Filters out irrelevant context.

    Args:
        state: The current GraphState containing question and documents.

    Returns:
        Dictionary update with filtered documents and web_search_needed flag.
    """
    print("--- [NODE] GRADE DOCUMENTS: Evaluating chunk relevance ---")
    question = state["question"]
    documents = state.get("documents", [])
    
    retrieval_grader = get_retrieval_grader()
    filtered_docs: List[Document] = []
    web_search_needed = False

    for doc in documents:
        try:
            score = retrieval_grader.invoke(
                {"question": question, "document": doc.page_content}
            )
            grade = score.binary_score.lower()
            if grade == "yes":
                print("  -> Grade: [RELEVANT] - Retaining chunk")
                filtered_docs.append(doc)
            else:
                print("  -> Grade: [IRRELEVANT] - Discarding chunk")
        except Exception as e:
            print(f"  -> Grading fallback due to error ({e}) - Retaining chunk")
            filtered_docs.append(doc)

    # Web search is only required if all local chunks were filtered out
    web_search_needed = (len(filtered_docs) == 0)
    if web_search_needed:
        print("  -> All local chunks were irrelevant. Web search flagged as mandatory.")
    else:
        print(f"  -> Preserved {len(filtered_docs)} relevant chunk(s). Local context sufficient; skipping web search.")

    return {"documents": filtered_docs, "web_search_needed": web_search_needed}


# ============================================================================
# 3. Transform Query Node (Query Rewriter)
# ============================================================================

def transform_query(state: GraphState) -> Dict[str, Any]:
    """
    Transforms and optimizes the user query into an effective search engine keyword query.

    Args:
        state: The current GraphState containing the user question.

    Returns:
        Dictionary update with the rewritten question.
    """
    print("--- [NODE] TRANSFORM QUERY: Rewriting query for web search ---")
    question = state["question"]
    llm = get_judge_llm(temperature=0)

    system_prompt = (
        "You are an expert query optimizer for search engines.\n"
        "Look at the input question and formulate an improved, keyword-rich search query\n"
        "that will return the best possible search engine results. Do not include conversational filler.\n"
        "Output ONLY the revised query string."
    )

    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Initial Question:\n\n{question}\n\nFormulate an improved search query:"),
        ]
    )

    query_rewriter = re_write_prompt | llm | StrOutputParser()
    try:
        better_query = query_rewriter.invoke({"question": question}).strip()
        print(f"  -> Original: '{question}'")
        print(f"  -> Optimized: '{better_query}'")
    except Exception as e:
        print(f"  -> Rewriter fallback due to error: {e}")
        better_query = question

    return {"question": better_query}


# ============================================================================
# 4. Web Search Node
# ============================================================================

def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Executes an external live web search to retrieve fresh, external context.
    Prefers Tavily Search API; gracefully falls back to DuckDuckGo if no Tavily API key is set.

    Args:
        state: The current GraphState containing the transformed query and existing docs.

    Returns:
        Dictionary update with combined documents and web_search_needed set to False.
    """
    print("--- [NODE] WEB SEARCH: Executing live internet search ---")
    question = state["question"]
    documents = state.get("documents", [])
    tavily_api_key = os.getenv("TAVILY_API_KEY", "")

    search_content = ""
    
    # Try Tavily first if key is configured
    if tavily_api_key and not tavily_api_key.startswith("tvly-your_"):
        try:
            print("  -> Executing search via Tavily API...")
            tavily_tool = TavilySearchResults(max_results=3)
            search_results = tavily_tool.invoke({"query": question})
            if isinstance(search_results, list):
                search_content = "\n\n".join(
                    [res.get("content", "") for res in search_results if isinstance(res, dict)]
                )
            else:
                search_content = str(search_results)
        except Exception as e:
            print(f"  -> Tavily search failed ({e}), falling back to DuckDuckGo...")
            search_content = ""

    # Fallback to DuckDuckGo if Tavily wasn't used or failed
    if not search_content:
        try:
            print("  -> Executing search via DuckDuckGo fallback...")
            ddg_tool = DuckDuckGoSearchResults(max_results=3)
            search_content = ddg_tool.run(question)
        except Exception as e:
            print(f"  -> Web search fallback error: {e}")
            search_content = f"No external web results found for query: {question}"

    web_document = Document(page_content=search_content, metadata={"source": "web_search"})
    documents.append(web_document)

    return {"documents": documents, "web_search_needed": False}


# ============================================================================
# 5. Generate Node (RAG Answer Synthesis)
# ============================================================================

def generate(state: GraphState) -> Dict[str, Any]:
    """
    Synthesizes a factually grounded answer using the curated document context.

    Args:
        state: The current GraphState containing question and verified documents.

    Returns:
        Dictionary update containing the generated answer.
    """
    print("--- [NODE] GENERATE: Synthesizing final answer from context ---")
    question = state["question"]
    documents = state.get("documents", [])
    
    # Format all document contents into a single context string
    context_str = "\n\n---\n\n".join([doc.page_content for doc in documents])
    
    llm = get_judge_llm(temperature=0)

    system_prompt = (
        "You are an expert assistant for question-answering tasks.\n"
        "Use ONLY the following pieces of retrieved context to answer the question.\n"
        "If you do not know the answer based on the provided context, state that you do not know.\n"
        "Keep the answer factual, precise, and concise.\n\n"
        "Context:\n"
        "{context}"
    )

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question: {question}\n\nAnswer:"),
        ]
    )

    rag_chain = rag_prompt | llm | StrOutputParser()
    
    try:
        generation = rag_chain.invoke({"context": context_str, "question": question})
    except Exception as e:
        generation = f"Generation failed due to error: {e}"

    return {"generation": generation}


if __name__ == "__main__":
    print("--- Testing Graph Action Nodes (Module Verification) ---")
    sample_state: GraphState = {
        "question": "What is the memory requirement for the Nexus-X Protocol?",
        "generation": "",
        "web_search_needed": False,
        "documents": [],
        "retry_count": 0,
    }

    # 1. Test Retrieve Node
    retrieval_output = retrieve(sample_state)
    print(f"[OK] Retrieve Node fetched {len(retrieval_output['documents'])} chunks.")

    # 2. Test Grade Documents Node
    graded_output = grade_documents(retrieval_output)
    print(f"[OK] Grade Node preserved {len(graded_output['documents'])} chunks. Web search needed: {graded_output['web_search_needed']}")

    print("[SUCCESS] Graph Action Nodes initialized and verified successfully!")
