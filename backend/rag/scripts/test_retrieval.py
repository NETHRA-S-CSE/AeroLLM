import json
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer


INDEX_FILE = Path("rag/vectorstore/test/faa_sdr.index")
METADATA_FILE = Path("rag/vectorstore/test/metadata.json")


print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_FILE))

print("Loading metadata...")
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def search(query, top_k=5):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    print("\n" + "=" * 90)
    print(f"QUERY: {query}")
    print("=" * 90)

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]), 1
    ):

        record = metadata[idx]

        print(f"\n--- RESULT {rank} ---")
        print(f"Similarity : {score:.4f}")
        print(f"Report ID  : {record['id']}")

        print("\nMetadata:")
        for key, value in record["metadata"].items():
            print(f"  {key}: {value}")

        print("\nReport:")
        print(record["text"])


queries = [
    "hydraulic reservoir low pressure during takeoff",
    "engine vibration and component replacement",
    "emergency lights inoperative and replaced",
]


for query in queries:
    search(query, top_k=5)