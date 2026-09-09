import json
import numpy as np
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer

INPUT_FILE = Path("rag/data/documents/faa_sdr_documents.jsonl")
OUTPUT_DIR = Path("rag/vectorstore/test")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_DOCUMENTS = 10000
BATCH_SIZE = 64

print("Loading embedding model...")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Reading documents...")

texts = []
metadata = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for i, line in enumerate(f):

        if i >= MAX_DOCUMENTS:
            break

        record = json.loads(line)

        texts.append(record["text"])
        metadata.append(record)

print(f"Documents loaded: {len(texts):,}")

print("\nGenerating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.asarray(embeddings, dtype="float32")

print(f"Embedding shape: {embeddings.shape}")

print("\nBuilding FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print(f"FAISS vectors: {index.ntotal:,}")

print("\nSaving index...")

faiss.write_index(
    index,
    str(OUTPUT_DIR / "faa_sdr.index")
)

with open(
    OUTPUT_DIR / "metadata.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        ensure_ascii=False
    )

print("\n================================")
print("FAISS TEST COMPLETE")
print("================================")
print(f"Vectors: {index.ntotal:,}")
print(f"Index: {OUTPUT_DIR / 'faa_sdr.index'}")
print(f"Metadata: {OUTPUT_DIR / 'metadata.json'}")