from typing import Literal

from langgraph.graph import StateGraph, END
from langchain_core.runnables.graph import MermaidDrawMethod

from src.config import MAX_RETRIES
from src.agents import (
    AgentState,
    retrieve_from_pinecone_node,
    filter_evidence_node,
    classify_sms_base_node,
    classify_sms_lora_node,
    verify_classification_node,
    finalize_output_node,
)


def route_classifier(state: AgentState) -> Literal["base", "lora"]:
    """
    Route to the correct classifier backend.
    """
    if state["classifier_mode"] == "lora":
        return "lora"
    return "base"


def route_after_verification(state: AgentState) -> Literal["retry", "finalize"]:
    """
    Decide whether to retry retrieval/classification or finalize the answer.
    """
    if state["needs_retry"] and state["retry_count"] < MAX_RETRIES:
        return "retry"
    return "finalize"


def build_graph():
    """
    Build and compile the LangGraph workflow.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve_from_pinecone", retrieve_from_pinecone_node)
    workflow.add_node("filter_evidence", filter_evidence_node)
    workflow.add_node("classify_sms_base", classify_sms_base_node)
    workflow.add_node("classify_sms_lora", classify_sms_lora_node)
    workflow.add_node("verify_classification", verify_classification_node)
    workflow.add_node("finalize_output", finalize_output_node)

    workflow.set_entry_point("retrieve_from_pinecone")

    workflow.add_edge("retrieve_from_pinecone", "filter_evidence")

    workflow.add_conditional_edges(
        "filter_evidence",
        route_classifier,
        {
            "base": "classify_sms_base",
            "lora": "classify_sms_lora",
        },
    )

    workflow.add_edge("classify_sms_base", "verify_classification")
    workflow.add_edge("classify_sms_lora", "verify_classification")

    workflow.add_conditional_edges(
        "verify_classification",
        route_after_verification,
        {
            "retry": "retrieve_from_pinecone",
            "finalize": "finalize_output",
        },
    )

    workflow.add_edge("finalize_output", END)

    return workflow.compile()


def save_graph_png(output_path: str = "outputs/graph.png"):
    """
    Save a PNG visualization of the compiled LangGraph.
    """
    graph = build_graph()
    png_bytes = graph.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API
    )

    with open(output_path, "wb") as f:
        f.write(png_bytes)

    return output_path
