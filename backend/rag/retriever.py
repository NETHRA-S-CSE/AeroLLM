import json
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

# Resolve paths relative to this file's location (backend/rag/)
BASE_DIR = Path(__file__).parent.resolve()
DEFAULT_INDEX_PATH = BASE_DIR / "vectorstore" / "test" / "faa_sdr.index"
DEFAULT_METADATA_PATH = BASE_DIR / "vectorstore" / "test" / "metadata.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level singletons for lazy loading
_index: Optional[faiss.Index] = None
_metadata: Optional[List[Dict[str, Any]]] = None
_model: Optional[SentenceTransformer] = None


def load_resources(
    index_path: Path = DEFAULT_INDEX_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    model_name: str = MODEL_NAME
) -> None:
    """
    Lazy loads the FAISS index, metadata, and embedding model into memory once.
    """
    global _index, _metadata, _model

    if _index is None:
        path = Path(index_path)
        if not path.is_absolute() and not path.exists():
            path = BASE_DIR / index_path
        if not path.exists():
            raise FileNotFoundError(f"FAISS index file not found at: {path}")
        _index = faiss.read_index(str(path))

    if _metadata is None:
        path = Path(metadata_path)
        if not path.is_absolute() and not path.exists():
            path = BASE_DIR / metadata_path
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            _metadata = json.load(f)

    if _model is None:
        _model = SentenceTransformer(model_name)


def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieves the top-K relevant documents for a given search query using FAISS vector search.

    Args:
        query (str): The search query string.
        top_k (int): Number of top results to return. Default is 5.

    Returns:
        List[Dict[str, Any]]: List of dictionary results containing:
            - 'id': Report ID string
            - 'score': Similarity score float
            - 'text': Full text of the document
            - 'metadata': Metadata dictionary
    """
    load_resources()

    # Convert query to normalized embedding
    query_embedding = _model.encode([query], normalize_embeddings=True)

    # Search FAISS index
    scores, indices = _index.search(query_embedding, top_k)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_metadata):
            continue
        record = _metadata[idx]
        results.append({
            "id": record["id"],
            "score": float(score),
            "text": record["text"],
            "metadata": record.get("metadata", {})
        })

    return results


if __name__ == "__main__":
    # Simple self-test run
    sample_results = retrieve("hydraulic reservoir low pressure during takeoff", top_k=2)
    print(f"Retrieved {len(sample_results)} sample results successfully.")
