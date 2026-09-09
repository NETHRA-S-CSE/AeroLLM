import sys
import json
from pathlib import Path

# Add backend directory to sys.path
CURRENT_DIR = Path(__file__).parent.resolve()
BACKEND_DIR = CURRENT_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app

def test_api_input_output():
    client = TestClient(app)

    test_queries = [
        "A320 hydraulic low pressure during takeoff",
        "engine vibration and component replacement",
        "Boeing 737 electrical wiring failure"
    ]

    print("=" * 80)
    print("AeroLLM Input & Output End-to-End API Test")
    print("=" * 80)

    for idx, query in enumerate(test_queries, 1):
        input_payload = {
            "query": query,
            "top_k": 5
        }

        print(f"\n--- TEST CASE {idx} ---")
        print("\n[INPUT REQUEST] POST /query:")
        print(json.dumps(input_payload, indent=2))

        response = client.post("/query", json=input_payload)
        
        print(f"\n[OUTPUT RESPONSE] HTTP Status: {response.status_code}")
        if response.status_code == 200:
            res_json = response.json()
            print(json.dumps(res_json, indent=2))
        else:
            print(response.text)
        print("-" * 80)

if __name__ == "__main__":
    test_api_input_output()
