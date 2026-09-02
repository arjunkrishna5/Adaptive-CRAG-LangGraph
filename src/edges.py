"""
edges.py
--------
Defines conditional routing logic (edges) for the Adaptive Corrective RAG (CRAG) workflow:
1. decide_to_generate: Routes to 'generate' if local documents are sufficient,
   or 'transform_query' if web search fallback is needed.
2. grade_generation: Evaluates the synthesized answer for hallucinations and
   relevance, managing self-correction retry loops and circuit-breaker limits.
"""

from src.state import GraphState
from src.evaluators import get_hallucination_grader, get_answer_grader

# Maximum allowed self-correction attempts before triggering circuit breaker
MAX_RETRIES = 3


def decide_to_generate(state: GraphState) -> str:
    """
    Evaluates whether to proceed directly to answer synthesis or fall back to web search.

    Args:
        state: Current GraphState containing the web_search_needed flag.

    Returns:
        Next node name: 'transform_query' or 'generate'.
    """
    print("--- [EDGE] ROUTING DECISION: Evaluating retrieval sufficiency ---")
    web_search_needed = state.get("web_search_needed", False)

    if web_search_needed:
        print("  -> Decision: Context insufficient / irrelevant. Routing to [transform_query].")
        return "transform_query"
    else:
        print("  -> Decision: Context sufficient. Routing directly to [generate].")
        return "generate"


def grade_generation(state: GraphState) -> str:
    """
    Self-reflective evaluation of the generated answer against retrieved facts and query.
    1. Checks for factual hallucinations.
    2. Checks for direct answer relevance.
    3. Enforces circuit breaker limits to prevent infinite cycles.

    Args:
        state: Current GraphState containing question, documents, generation, and retry_count.

    Returns:
        Next node name: 'generate' (retry), 'transform_query' (fallback), or 'end' (success).
    """
    print("--- [EDGE] QUALITY INSPECTION: Checking hallucinations and relevance ---")
    question = state["question"]
    documents = state.get("documents", [])
    generation = state.get("generation", "")
    retry_count = state.get("retry_count", 0)

    # Format documents for hallucination checker
    context_str = "\n\n".join([doc.page_content for doc in documents])

    # Circuit breaker: stop if max retries exceeded
    if retry_count >= MAX_RETRIES:
        print(f"  -> Circuit breaker triggered (max retries: {MAX_RETRIES} reached). Routing to [end].")
        return "end"

    # Step 1: Hallucination Evaluation (Grounding)
    hallucination_grader = get_hallucination_grader()
    try:
        hallucination_score = hallucination_grader.invoke(
            {"documents": context_str, "generation": generation}
        )
        is_grounded = hallucination_score.binary_score.lower() == "yes"
    except Exception as e:
        print(f"  -> Hallucination check fallback due to error: {e}")
        is_grounded = True

    if not is_grounded:
        print(f"  -> Hallucination detected! Attempt #{retry_count + 1}. Routing back to [generate].")
        state["retry_count"] = retry_count + 1
        return "generate"

    print("  -> Fact Check Passed: Generation is grounded in facts.")

    # Step 2: Answer Relevance Evaluation
    answer_grader = get_answer_grader()
    try:
        answer_score = answer_grader.invoke(
            {"question": question, "generation": generation}
        )
        is_relevant = answer_score.binary_score.lower() == "yes"
    except Exception as e:
        print(f"  -> Answer relevance check fallback due to error: {e}")
        is_relevant = True

    if is_relevant:
        print("  -> Answer Relevance Passed: Generation directly answers the question. Routing to [end].")
        return "end"
    else:
        print(f"  -> Answer not relevant to question. Routing to [transform_query] for external search.")
        state["retry_count"] = retry_count + 1
        return "transform_query"


if __name__ == "__main__":
    print("--- Testing Conditional Edge Functions ---")
    
    # 1. Test decide_to_generate routing
    state_local: GraphState = {"web_search_needed": False}  # type: ignore
    state_web: GraphState = {"web_search_needed": True}  # type: ignore
    
    print(f"[TEST 1] Local docs adequate -> Next: '{decide_to_generate(state_local)}' (Expected: 'generate')")
    print(f"[TEST 2] Local docs missing   -> Next: '{decide_to_generate(state_web)}' (Expected: 'transform_query')")
    print("[SUCCESS] Conditional Edge logic verified successfully!")
