import sys
from pathlib import Path

# Add backend directory to sys.path for relative package resolution
CURRENT_DIR = Path(__file__).parent.resolve()
BACKEND_DIR = CURRENT_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from rag.retriever import retrieve
from llm.context_builder import build_context
from llm.prompts import build_full_prompt

def test_context_pipeline():
    test_query = "hydraulic reservoir low pressure during takeoff"

    print("=" * 80)
    print("AeroLLM Context & Prompt Generation Pipeline Test")
    print("=" * 80)
    print(f"\n1. INPUT QUERY: \"{test_query}\"\n")

    # Step 1: Retrieve top 5 documents
    print("2. RETRIEVING TOP 5 DOCUMENTS...")
    retrieved_docs = retrieve(test_query, top_k=5)
    print(f"   Retrieved {len(retrieved_docs)} records from FAISS vectorstore.\n")

    # Step 2: Build context
    print("3. BUILDING CONTEXT BLOCK...")
    context_str = build_context(retrieved_docs)
    print(f"   Context built successfully ({len(context_str)} characters).\n")

    # Step 3: Build final prompt
    print("4. GENERATING FULL LLM PROMPT PAYLOAD...")
    prompt_payload = build_full_prompt(test_query, retrieved_docs)
    
    print("\n" + "=" * 80)
    print("GENERATED FULL PROMPT PAYLOAD")
    print("=" * 80)
    print(prompt_payload["full_prompt"])
    print("=" * 80)

if __name__ == "__main__":
    test_context_pipeline()
