import sys
from pathlib import Path

# Add backend/rag to sys.path if needed
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from retriever import retrieve

test_queries = [
    "hydraulic reservoir low pressure during takeoff",
    "engine vibration and component replacement",
    "emergency lights inoperative and replaced"
]

def run_tests():
    print("=" * 80)
    print("AeroLLM RAG Retriever Test Suite")
    print("=" * 80)

    for query_idx, query in enumerate(test_queries, 1):
        print(f"\nQUERY {query_idx}: \"{query}\"")
        print("-" * 80)
        
        results = retrieve(query, top_k=5)
        print(f"Top {len(results)} results:\n")

        for rank, item in enumerate(results, 1):
            meta = item.get("metadata", {})
            aircraft_make = meta.get("aircraft_make", "N/A")
            aircraft_model = meta.get("aircraft_model", "N/A")
            aircraft_info = f"{aircraft_make} {aircraft_model}".strip()
            
            part_name = meta.get("part_name", "N/A")
            component_name = meta.get("component_name", "N/A")
            component_info = f"{component_name} (Part: {part_name})"
            
            raw_text = item.get("text", "").replace("\n", " ")
            text_preview = raw_text[:150] + "..." if len(raw_text) > 150 else raw_text

            print(f"  [{rank}] Report ID     : {item['id']}")
            print(f"      Similarity Score: {item['score']:.4f}")
            print(f"      Aircraft        : {aircraft_info}")
            print(f"      Component       : {component_info}")
            print(f"      Text Preview    : {text_preview}")
            print()

if __name__ == "__main__":
    run_tests()
