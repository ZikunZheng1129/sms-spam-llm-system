"""Top-level mode runners.

Provides:
    - run_advanced_rag(user_query, classifier_mode)
    - run_guarded_fallback(user_query)        # Phase 1, simple orchestration
    - run_all_modes(user_query)
"""

from __future__ import annotations

from typing import Any

from src.base_modes import run_base_llm, run_basic_rag
from src.graph import build_graph


def run_advanced_rag(user_query: str, classifier_mode: str = "base") -> dict[str, Any]:
    """Run the advanced agentic RAG workflow with the requested classifier."""
    graph = build_graph()

    initial_state = {
        "user_query": user_query,
        "retrieved_docs": [],
        "filtered_docs": [],
        "classification": "",
        "explanation": "",
        "verification": "",
        "needs_retry": False,
        "retry_count": 0,
        "final_answer": "",
        "classifier_mode": classifier_mode,
    }

    result = graph.invoke(initial_state)

    return {
        "mode": f"advanced_rag_{classifier_mode}",
        "user_query": user_query,
        "final_answer": result["final_answer"],
        "classification": result["classification"],
        "explanation": result["explanation"],
        "verification": result["verification"],
        "filtered_docs": result["filtered_docs"],
    }


def run_guarded_fallback(user_query: str) -> dict[str, Any]:
    """LoRA-first orchestration with verifier-driven fallback to the base agent.

    Phase 1 implementation: simple two-call orchestration on top of the
    existing graph. A dedicated fallback subgraph may replace this in a later
    phase.

    Steps:
        1. Run advanced_rag(classifier_mode="lora").
        2. Read its verification result.
        3. If verification == "supported", keep the LoRA result.
        4. Otherwise, run advanced_rag(classifier_mode="base") and use it.

    Returns a dict with these top-level keys:
        - mode
        - user_query
        - initial_lora_prediction
        - lora_verification
        - fallback_used
        - final_prediction
        - final_answer
        - lora_result
        - base_result  (only present when fallback_used is True)
    """
    lora_result = run_advanced_rag(user_query, classifier_mode="lora")
    initial_lora_prediction = (lora_result.get("classification") or "").strip().lower()
    lora_verification = (lora_result.get("verification") or "").strip().lower()

    if lora_verification == "supported":
        final_prediction = initial_lora_prediction
        fallback_used = False
        final_answer = (
            f"Final Prediction: {(final_prediction or 'unknown').upper()}\n"
            f"Initial LoRA Prediction: {(initial_lora_prediction or 'unknown').upper()}\n"
            f"LoRA Verification: {lora_verification or 'unknown'}\n"
            f"Fallback Used: {fallback_used}\n"
            f"Source: advanced_rag_lora"
        )
        return {
            "mode": "guarded_fallback",
            "user_query": user_query,
            "initial_lora_prediction": initial_lora_prediction,
            "lora_verification": lora_verification,
            "fallback_used": fallback_used,
            "final_prediction": final_prediction,
            "final_answer": final_answer,
            "lora_result": lora_result,
        }

    base_result = run_advanced_rag(user_query, classifier_mode="base")
    final_prediction = (base_result.get("classification") or "").strip().lower()
    fallback_used = True
    base_verification = (base_result.get("verification") or "").strip().lower()
    final_answer = (
        f"Final Prediction: {(final_prediction or 'unknown').upper()}\n"
        f"Initial LoRA Prediction: {(initial_lora_prediction or 'unknown').upper()}\n"
        f"LoRA Verification: {lora_verification or 'unknown'}\n"
        f"Fallback Used: {fallback_used}\n"
        f"Base Verification: {base_verification or 'unknown'}\n"
        f"Source: advanced_rag_base (fallback)"
    )
    return {
        "mode": "guarded_fallback",
        "user_query": user_query,
        "initial_lora_prediction": initial_lora_prediction,
        "lora_verification": lora_verification,
        "fallback_used": fallback_used,
        "final_prediction": final_prediction,
        "final_answer": final_answer,
        "lora_result": lora_result,
        "base_result": base_result,
    }


def run_all_modes(user_query: str) -> dict[str, Any]:
    """Run all comparison modes for one query (used by run_comparison.py)."""
    return {
        "base_llm": run_base_llm(user_query),
        "basic_rag": run_basic_rag(user_query),
        "advanced_rag_base": run_advanced_rag(user_query, classifier_mode="base"),
        "advanced_rag_lora": run_advanced_rag(user_query, classifier_mode="lora"),
        "guarded_fallback": run_guarded_fallback(user_query),
    }
