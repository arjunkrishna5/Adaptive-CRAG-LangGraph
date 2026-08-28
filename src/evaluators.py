"""
evaluators.py
-------------
Implements structured LLM judges for the Adaptive Corrective RAG (CRAG) system:
1. Document Relevance Grader: Evaluates if retrieved chunks are relevant to the question.
2. Hallucination Grader: Evaluates if the generated answer is grounded in the retrieved facts.
3. Answer Relevance Grader: Evaluates if the generated answer resolves the user's question.

Uses Pydantic schemas to enforce strictly typed, deterministic binary ('yes' | 'no') outputs.
"""

import os
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


# ============================================================================
# 1. Pydantic Schemas for Structured Output
# ============================================================================

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: Literal["yes", "no"] = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucination check in generation against context."""
    binary_score: Literal["yes", "no"] = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score for evaluating whether the answer addresses the question."""
    binary_score: Literal["yes", "no"] = Field(
        description="Answer addresses the user's question, 'yes' or 'no'"
    )


# ============================================================================
# 2. LLM Initialization Helper
# ============================================================================

def get_judge_llm(temperature: float = 0):
    """
    Initializes and returns the judge LLM.
    Prefers Groq (Llama 3.3 70B) for high-speed inference; falls back to OpenAI (GPT-4o-mini).
    """
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")

    # Check for valid Groq key (not empty or placeholder)
    if groq_api_key and not groq_api_key.startswith("gsk_your_"):
        return ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=temperature,
            api_key=groq_api_key,
        )

    # Check for valid OpenAI key
    if openai_api_key and not openai_api_key.startswith("sk-your_"):
        return ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=temperature,
            api_key=openai_api_key,
        )

    # Default fallback: Groq instance (will read from env directly if configured later)
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=temperature,
    )


# ============================================================================
# 3. Judge Chains Implementation
# ============================================================================

def get_retrieval_grader(llm=None):
    """
    Creates and returns the Document Relevance Grader chain.
    """
    if llm is None:
        llm = get_judge_llm(temperature=0)

    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    system_prompt = (
        "You are an expert evaluator assessing the relevance of a retrieved document to a user question.\n"
        "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.\n"
        "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
    )

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader


def get_hallucination_grader(llm=None):
    """
    Creates and returns the Hallucination Grader chain (Grounding check).
    """
    if llm is None:
        llm = get_judge_llm(temperature=0)

    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    system_prompt = (
        "You are an expert fact-checking evaluator assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.\n"
        "Give a binary score 'yes' or 'no'. 'yes' means that the answer is fully grounded in and supported by the document context.\n"
        "'no' means the answer makes ungrounded claims or introduces facts not present in the documents."
    )

    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Set of facts:\n\n{documents}\n\nLLM generation:\n\n{generation}"),
        ]
    )

    hallucination_grader = hallucination_prompt | structured_llm_grader
    return hallucination_grader


def get_answer_grader(llm=None):
    """
    Creates and returns the Answer Relevance Grader chain.
    """
    if llm is None:
        llm = get_judge_llm(temperature=0)

    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    system_prompt = (
        "You are an expert evaluator assessing whether an answer addresses and resolves the user's specific question.\n"
        "Give a binary score 'yes' or 'no'. 'yes' means the answer directly resolves the question.\n"
        "'no' means the answer is off-topic, evasive, or fails to address the question."
    )

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "User question:\n\n{question}\n\nLLM generation:\n\n{generation}"),
        ]
    )

    answer_grader = answer_prompt | structured_llm_grader
    return answer_grader


if __name__ == "__main__":
    print("--- Testing Structured Evaluators & Pydantic Schemas ---")

    # 1. Test Pydantic validation mechanics directly
    test_doc_grade = GradeDocuments(binary_score="yes")
    print(f"[OK] GradeDocuments Schema valid: {test_doc_grade.model_dump()}")

    test_hallucination_grade = GradeHallucinations(binary_score="no")
    print(f"[OK] GradeHallucinations Schema valid: {test_hallucination_grade.model_dump()}")

    test_answer_grade = GradeAnswer(binary_score="yes")
    print(f"[OK] GradeAnswer Schema valid: {test_answer_grade.model_dump()}")

    # 2. Test prompt template construction
    try:
        r_grader = get_retrieval_grader()
        h_grader = get_hallucination_grader()
        a_grader = get_answer_grader()
        print("[SUCCESS] All 3 LLM Judge Chains built successfully!")
    except Exception as e:
        print(f"[NOTE] LLM Judge initialized with configuration: {e}")
