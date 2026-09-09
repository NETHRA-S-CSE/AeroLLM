import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure backend directory is in sys.path for clean package imports
CURRENT_DIR = Path(__file__).parent.resolve()
BACKEND_DIR = CURRENT_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from rag.retriever import retrieve
from llm.context_builder import build_context
from llm.prompts import build_full_prompt
from llm.llm_client import LLMClient, generate


def answer_query(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Executes the complete AeroLLM RAG pipeline:
    Query -> FAISS Retriever -> Top-K FAA SDR Reports -> Context Builder -> Prompt Builder -> LLM Client -> Structured Output

    Args:
        query (str): User search query string.
        top_k (int): Number of top historical SDR reports to retrieve. Default is 5.

    Returns:
        Dict[str, Any]: Result dictionary containing:
            - 'query': User input query
            - 'answer': Generated LLM structured answer
            - 'retrieved_sources': List of retrieved SDR report records
            - 'report_ids': List of retrieved report IDs
            - 'scores': List of vector similarity scores
    """
    # 1. Retrieve top-K reports
    retrieved_docs = retrieve(query, top_k=top_k)

    # 2. Build context and full prompt payload
    prompt_payload = build_full_prompt(query, retrieved_docs)

    # 3. Generate answer via LLM Client
    client = LLMClient()
    llm_answer = client.generate(prompt_payload["full_prompt"])

    # Extract summary lists
    report_ids = [doc.get("id", "N/A") for doc in retrieved_docs]
    scores = [float(doc.get("score", 0.0)) for doc in retrieved_docs]

    return {
        "query": query,
        "answer": llm_answer,
        "retrieved_sources": retrieved_docs,
        "report_ids": report_ids,
        "scores": scores
    }


if __name__ == "__main__":
    # Quick self-test run
    result = answer_query("hydraulic reservoir low pressure during takeoff", top_k=2)
    print(f"Pipeline executed successfully. Retrieved Report IDs: {result['report_ids']}")
