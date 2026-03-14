import os
from typing import List, Dict, Any

from fastembed import TextEmbedding
from pinecone import Pinecone

from src.config import PINECONE_INDEX_NAME, EMBEDDING_MODEL_NAME


def initialize_pinecone(index_name: str = PINECONE_INDEX_NAME):
    """
    Initialize Pinecone client and return the target index.
    Requires PINECONE_API_KEY to be set in the environment.
    """
    api_key = os.environ["PINECONE_API_KEY"]
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)


def initialize_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """
    Initialize the embedding model used both for indexing and querying.
    """
    return TextEmbedding(model_name=model_name)


def embed_text(text: str, model: TextEmbedding) -> List[float]:
    """
    Embed a single query text and return it as a list of floats.
    """
    return next(model.embed([text])).tolist()


def retrieve_from_pinecone(
    query_text: str,
    index,
    model: TextEmbedding,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Query Pinecone using the embedded query text and return
    a simplified list of retrieval results.
    """
    query_vec = embed_text(query_text, model)

    response = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True
    )

    results = []
    for match in response["matches"]:
        results.append(
            {
                "id": match["id"],
                "score": float(match["score"]),
                "label": match["metadata"].get("label", ""),
                "text": match["metadata"].get("text", "")
            }
        )

    return results


def format_retrieved_docs(docs: List[Dict[str, Any]]) -> str:
    """
    Format retrieved documents into a readable string for prompting/debugging.
    """
    if not docs:
        return "No retrieved documents."

    formatted = []
    for i, doc in enumerate(docs, start=1):
        formatted.append(
            f"Document {i}:\n"
            f"ID: {doc['id']}\n"
            f"Score: {doc['score']:.4f}\n"
            f"Label: {doc['label']}\n"
            f"Text: {doc['text']}\n"
        )

    return "\n".join(formatted)
