from src.base_modes import run_base_llm, run_basic_rag
from src.graph import build_graph


def run_advanced_rag(user_query: str, classifier_mode: str = "base") -> dict:
    """
    Run the advanced agentic RAG workflow with either:
    - classifier_mode="base"
    - classifier_mode="lora"
    """
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


def run_all_modes(user_query: str) -> dict:
    """
    Run all four required comparison modes for one query.
    """
    return {
        "base_llm": run_base_llm(user_query),
        "basic_rag": run_basic_rag(user_query),
        "advanced_rag_base": run_advanced_rag(user_query, classifier_mode="base"),
        "advanced_rag_lora": run_advanced_rag(user_query, classifier_mode="lora"),
    }
