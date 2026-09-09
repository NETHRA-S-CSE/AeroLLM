import json
from pathlib import Path

file_path = Path("rag/data/documents/faa_sdr_documents.jsonl")

count = 0
total_chars = 0

print("Inspecting generated documents...\n")

with open(file_path, "r", encoding="utf-8") as f:

    for line in f:
        record = json.loads(line)

        count += 1
        total_chars += len(record["text"])

        if count <= 3:
            print("=" * 80)
            print(f"DOCUMENT {count}")
            print("=" * 80)
            print(record["text"])
            print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Total documents: {count:,}")
print(f"Total characters: {total_chars:,}")
print(f"Average characters/document: {total_chars / count:,.0f}")