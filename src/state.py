"""
state.py
--------
Defines the central state schema (GraphState) for the Adaptive Corrective RAG (CRAG) workflow.

The GraphState acts as the shared working memory across all nodes and edges in the
LangGraph state machine. Each node receives this state, executes its dedicated logic,
and returns a dictionary of state updates.
"""

from typing import List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Represents the shared memory state of the Adaptive CRAG graph.

    Attributes:
        question: The current question or transformed search query.
        generation: The generated response from the LLM synthesizer.
        web_search_needed: Boolean flag indicating whether external web fallback is required.
        documents: A list of relevant Document objects preserved for context generation.
        retry_count: An integer counter tracking self-correction attempts to prevent infinite cycles.
    """
    question: str
    generation: str
    web_search_needed: bool
    documents: List[Document]
    retry_count: int
