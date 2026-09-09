import sys
from pathlib import Path

# Add backend directory to sys.path
CURRENT_DIR = Path(__file__).parent.resolve()
BACKEND_DIR = CURRENT_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app


def test_api_endpoints():
    client = TestClient(app)

    print("=" * 80)
    print("NovaTRix AeroLLM - FastAPI Backend API Test Suite")
    print("=" * 80)

    # 1. Test GET /health
    print("\n1. TESTING GET /health...")
    health_resp = client.get("/health")
    print(f"   HTTP Status Code: {health_resp.status_code}")
    print(f"   Response Payload: {health_resp.json()}")
    
    assert health_resp.status_code == 200
    assert health_resp.json() == {"status": "ok"}
    print("   [OK] GET /health PASSED!")

    # 2. Test POST /query
    print("\n2. TESTING POST /query...")
    test_payload = {"query": "A320 hydraulic low pressure during takeoff", "top_k": 5}
    query_resp = client.post("/query", json=test_payload)
    print(f"   HTTP Status Code: {query_resp.status_code}")

    res_json = query_resp.json()

    print("\n" + "=" * 40)
    print("API RESPONSE PAYLOAD")
    print("=" * 40)
    print(f"Query  : {res_json.get('query')}")
    print("Sources:")
    for src in res_json.get("sources", []):
        print(f"  - ID: {src['id']} | Score: {src['score']}")

    print("\nLLM Answer Response:")
    print(res_json.get("answer"))
    print("=" * 40)

    # Schema assertions
    assert query_resp.status_code == 200
    assert res_json["query"] == test_payload["query"]
    assert len(res_json["sources"]) == 5
    assert "Fault / Observation:" in res_json["answer"]
    assert "Relevant Report IDs:" in res_json["answer"]

    print("\n   [OK] POST /query PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_api_endpoints()
