from langchain_openai import ChatOpenAI

from src.config import DEFAULT_TEMPERATURE, TOP_K_RETRIEVAL
from src.pinecone_utils import (
    initialize_pinecone,
    initialize_embedding_model,
    retrieve_from_pinecone,
    format_retrieved_docs,
)


def get_base_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=DEFAULT_TEMPERATURE)


def run_base_llm(user_query: str) -> dict:
    """
    Base LLM only: no retrieval, no RAG.
    """
    llm = get_base_llm()

    prompt = f"""Classify the following SMS message as spam or ham.

SMS:
{user_query}

Return your answer in exactly this format:
Label: <spam or ham>
Explanation: <short explanation>
"""

    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)

    return {
        "mode": "base_llm",
        "user_query": user_query,
        "raw_output": content,
    }


def run_basic_rag(user_query: str) -> dict:
    """
    Basic RAG: retrieve similar examples, then ask the base LLM to classify.
    """
    llm = get_base_llm()
    index = initialize_pinecone()
    model = initialize_embedding_model()

    retrieved_docs = retrieve_from_pinecone(
        query_text=user_query,
        index=index,
        model=model,
        top_k=TOP_K_RETRIEVAL,
    )

    docs_text = format_retrieved_docs(retrieved_docs)

    prompt = f"""Classify the following SMS message as spam or ham using the retrieved examples as evidence.

SMS:
{user_query}

Retrieved examples:
{docs_text}

Return your answer in exactly this format:
Label: <spam or ham>
Explanation: <short explanation>
"""

    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)

    return {
        "mode": "basic_rag",
        "user_query": user_query,
        "retrieved_docs": retrieved_docs,
        "raw_output": content,
    }