"""
graph.py
--------
Assembles and compiles the complete Adaptive Corrective RAG (CRAG) state machine.
Integrates all action nodes, fixed transitions, and self-reflective conditional edges
into an executable LangGraph application.
"""

from langgraph.graph import END, StateGraph
from src.state import GraphState
from src.nodes import (
    retrieve,
    grade_documents,
    transform_query,
    web_search,
    generate,
)
from src.edges import decide_to_generate, grade_generation


def create_crag_graph():
    """
    Constructs and compiles the Adaptive CRAG StateGraph.

    Returns:
        CompiledStateGraph runnable instance.
    """
    workflow = StateGraph(GraphState)

    # 1. Register Action Nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)

    # 2. Define Entry Point & Fixed Transitions
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # 3. Define Conditional Routing Edge (Local vs. Web Fallback)
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )

    # 4. Define Web Search Sequence Transitions
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")

    # 5. Define Self-Correction Inspection Edge (Hallucination & Relevance)
    workflow.add_conditional_edges(
        "generate",
        grade_generation,
        {
            "generate": "generate",
            "transform_query": "transform_query",
            "end": END,
        },
    )

    # Compile into executable state machine
    app = workflow.compile()
    return app


# Singleton compiled app instance
crag_app = create_crag_graph()


def run_crag(question: str) -> dict:
    """
    Executes a question end-to-end through the Adaptive CRAG graph.

    Args:
        question: User query string.

    Returns:
        Final GraphState dictionary containing the synthesized answer and metadata.
    """
    initial_state: GraphState = {
        "question": question,
        "generation": "",
        "web_search_needed": False,
        "documents": [],
        "retry_count": 0,
    }
    final_state = crag_app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    print("--- [GRAPH TEST] Compiling & Verifying Adaptive CRAG LangGraph ---")
    
    # 1. Verify graph compilation
    print(f"[OK] StateGraph successfully compiled: {type(crag_app)}")
    
    # 2. Print Mermaid Graph Definition for visualization
    try:
        mermaid_graph = crag_app.get_graph().draw_mermaid()
        print("\n--- Compiled Mermaid Graph Flow ---")
        print(mermaid_graph)
        print("-" * 35)
    except Exception as e:
        print(f"[NOTE] Mermaid drawing skipped: {e}")
    
    print("\n[SUCCESS] Phase 4 complete! The LangGraph state machine is fully compiled and ready.")
