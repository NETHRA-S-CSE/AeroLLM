import sys
from pathlib import Path

# Add backend directory to sys.path
CURRENT_DIR = Path(__file__).parent.resolve()
BACKEND_DIR = CURRENT_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from llm.rag_pipeline import answer_query

test_queries = [
    "hydraulic reservoir low pressure during takeoff",
    "engine vibration and component replacement",
    "emergency lights inoperative and replaced",
    "hydraulic leak and pump replacement",
    "aircraft electrical wiring failure"
]


def run_pipeline_tests():
    print("=" * 80)
    print("NovaTRix AeroLLM - Complete End-to-End RAG Pipeline Test")
    print("=" * 80)

    for idx, query in enumerate(test_queries, 1):
        result = answer_query(query, top_k=5)

        print("\n" + "=" * 40)
        print("QUERY")
        print("=" * 40)
        print(f"[{idx}] {result['query']}")

        print("\n" + "=" * 40)
        print("RETRIEVED SOURCES")
        print("=" * 40)
        for rank, doc in enumerate(result["retrieved_sources"], 1):
            meta = doc.get("metadata", {})
            make = meta.get("aircraft_make", "N/A")
            model = meta.get("aircraft_model", "N/A")
            part = meta.get("part_name", "N/A")
            score = doc.get("score", 0.0)
            print(f"{rank}. {doc['id']} (Score: {score:.4f}) | Aircraft: {make} {model} | Part: {part}")

        print("\n" + "=" * 40)
        print("LLM RESPONSE")
        print("=" * 40)
        print(result["answer"])
        print("-" * 80)


if __name__ == "__main__":
    run_pipeline_tests()
