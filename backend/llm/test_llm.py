import sys
from pathlib import Path

# Add backend directory to sys.path
CURRENT_DIR = Path(__file__).parent.resolve()
BACKEND_DIR = CURRENT_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from llm.llm_client import LLMClient, generate
from llm.prompts import SYSTEM_PROMPT

def test_llm_client():
    print("=" * 80)
    print("NovaTRix AeroLLM - LLM Client Test Suite")
    print("=" * 80)

    # Sample mock prompt simulating a context payload
    sample_prompt = f"""=== SYSTEM PROMPT ===
{SYSTEM_PROMPT}

=== USER PROMPT ===
USER QUERY:
hydraulic reservoir low pressure during takeoff

RETRIEVED HISTORICAL SDR EVIDENCE:
================================================================================
--- DOCUMENT [1] ---
REPORT ID: FAA_SDR_9557
Similarity: 0.4595
Aircraft: BOEING 7272Q9
JASC Code: 2926
Component / Part: Part: RESERVOIR | P/N: 69171354
Report Content:
ON CLIMB, LOST STANDBY HYDRAULIC QUANITY. REMOVED AND REPLACED STANDBY HYDRAULIC RESERVOIR AND LEFT WING TAI DUCT.

--- DOCUMENT [2] ---
REPORT ID: FAA_SDR_8887
Similarity: 0.4530
Aircraft: AIRBUS A320211
JASC Code: 2913
Component / Part: Part: PUMP | P/N: 12652
Report Content:
DURING CLIMBOUT FROM DTW, RECEIVED 'HYD B SYS LO PR' MESSAGE. REPLACED THE BLUE HYDRAULIC SYSTEM ELECTRIC PUMP. OPERATIONAL CHECK OK.
================================================================================
"""

    print("\n1. Testing module-level generate(prompt) interface...")
    response = generate(sample_prompt)

    print("\n" + "=" * 80)
    print("GENERATED LLM RESPONSE")
    print("=" * 80)
    print(response)
    print("=" * 80)

    # Validate output schema fields
    required_fields = [
        "Fault / Observation:",
        "Aircraft:",
        "Affected System:",
        "Affected Component:",
        "Historical Evidence:",
        "Reported Maintenance Action:",
        "Relevant Report IDs:",
        "Confidence:",
        "Notes:"
    ]

    missing_fields = [field for field in required_fields if field not in response]

    print("\n2. VERIFYING SCHEMA COMPLIANCE...")
    if not missing_fields:
        print("   [OK] ALL REQUIRED SCHEMA FIELDS ARE PRESENT!")
    else:
        print(f"   [FAIL] MISSING FIELDS: {missing_fields}")

if __name__ == "__main__":
    test_llm_client()
