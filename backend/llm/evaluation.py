import sys
import re
from pathlib import Path
from typing import Dict, Any, List

# Ensure backend directory is in sys.path
CURRENT_DIR = Path(__file__).parent.resolve()
BACKEND_DIR = CURRENT_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from llm.rag_pipeline import answer_query

# 10 Diverse Aviation Maintenance Queries covering required topics
EVALUATION_QUERIES = [
    {"topic": "Hydraulic Systems", "query": "hydraulic system pressure loss during flight"},
    {"topic": "Engine Vibration", "query": "engine vibration and component replacement"},
    {"topic": "Electrical Failures", "query": "generator control unit electrical power failure"},
    {"topic": "Emergency Lighting", "query": "emergency exit lighting inoperative and battery replacement"},
    {"topic": "Component Replacement", "query": "fuel pump failure and component replacement"},
    {"topic": "Fluid Leakage", "query": "hydraulic fluid leakage from main landing gear line"},
    {"topic": "Pressure Problems", "query": "cabin altitude pressure warning and valve inspection"},
    {"topic": "Wiring Problems", "query": "aircraft electrical wiring chafing and short circuit"},
    {"topic": "Aircraft Systems", "query": "flight control elevator trim system actuator malfunction"},
    {"topic": "Maintenance Observations", "query": "maintenance observation of crack in wing spar fitting"}
]

REQUIRED_SCHEMA_FIELDS = [
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


def evaluate_response(query_item: Dict[str, str], top_k: int = 5) -> Dict[str, Any]:
    query_str = query_item["query"]
    topic = query_item["topic"]

    pipeline_result = answer_query(query_str, top_k=top_k)
    answer = pipeline_result["answer"]
    retrieved_sources = pipeline_result["retrieved_sources"]
    retrieved_ids = set(pipeline_result["report_ids"])
    scores = pipeline_result["scores"]

    # --- Grounding & Verification Checks ---

    # 1. Report ID Validity: Parse cited IDs from answer and check against retrieved IDs
    cited_ids_match = re.search(r"Relevant Report IDs:\s*([^\n]+)", answer)
    cited_ids_str = cited_ids_match.group(1).strip() if cited_ids_match else ""
    cited_ids = [id_str.strip() for id_str in re.findall(r"FAA_SDR_\d+", cited_ids_str)]
    
    valid_report_ids = all(cid in retrieved_ids for cid in cited_ids) if cited_ids else False
    hallucinated_ids = [cid for cid in cited_ids if cid not in retrieved_ids]

    # 2. Aircraft Details Grounding
    aircraft_match = re.search(r"Aircraft:\s*([^\n]+)", answer)
    aircraft_ans = aircraft_match.group(1).strip() if aircraft_match else ""
    retrieved_aircrafts = [
        f"{doc.get('metadata', {}).get('aircraft_make', '')} {doc.get('metadata', {}).get('aircraft_model', '')}".strip()
        for doc in retrieved_sources
    ]
    aircraft_grounded = any(
        ans_part in r_air for r_air in retrieved_aircrafts for ans_part in aircraft_ans.split() if len(ans_part) > 2
    ) if aircraft_ans and aircraft_ans != "Unspecified in query" else True

    # 3. Component Details Grounding
    comp_match = re.search(r"Affected Component:\s*([^\n]+)", answer)
    comp_ans = comp_match.group(1).strip() if comp_match else ""
    retrieved_parts = [
        f"{doc.get('metadata', {}).get('part_name', '')} {doc.get('metadata', {}).get('component_name', '')}".strip().lower()
        for doc in retrieved_sources
    ]
    comp_grounded = any(
        word.lower() in " ".join(retrieved_parts) for word in comp_ans.split() if len(word) > 3
    ) if comp_ans and comp_ans != "Unspecified component" else True

    # 4. Maintenance Action Grounding: Verify maintenance action section is present and grounded
    action_match = re.search(r"Reported Maintenance Action:\s*([^\n]+)", answer)
    action_present = bool(action_match and len(action_match.group(1).strip()) > 10)

    # 5. Avoid Unsupported Claims (e.g. no "safe to fly" claims)
    unsafe_claims_detected = bool(re.search(r"\bsafe to fly\b|\bapproved for flight\b|\bno inspection needed\b", answer, re.IGNORECASE))
    unsupported_claims_avoided = not unsafe_claims_detected

    # 6. Distinguish Historical Evidence from Conclusions (Check for disclaimer / AMM note)
    distinguishes_evidence = "HISTORICAL SDR EVIDENCE ONLY" in answer or "Mandatory verification" in answer

    # 7. Schema Completeness
    schema_complete = all(field in answer for field in REQUIRED_SCHEMA_FIELDS)

    # Overall Grounding Pass
    overall_pass = (
        valid_report_ids and
        aircraft_grounded and
        comp_grounded and
        action_present and
        unsupported_claims_avoided and
        distinguishes_evidence and
        schema_complete
    )

    return {
        "topic": topic,
        "query": query_str,
        "top_score": scores[0] if scores else 0.0,
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "retrieved_ids": list(retrieved_ids),
        "cited_ids": cited_ids,
        "valid_report_ids": valid_report_ids,
        "hallucinated_ids": hallucinated_ids,
        "aircraft_grounded": aircraft_grounded,
        "comp_grounded": comp_grounded,
        "action_present": action_present,
        "unsupported_claims_avoided": unsupported_claims_avoided,
        "distinguishes_evidence": distinguishes_evidence,
        "schema_complete": schema_complete,
        "overall_pass": overall_pass,
        "answer": answer
    }


def run_evaluation():
    print("=" * 80)
    print("NovaTRix AeroLLM - RAG -> LLM Grounding & Faithfulness Evaluation")
    print("=" * 80)

    results = []
    for idx, item in enumerate(EVALUATION_QUERIES, 1):
        print(f"\nEvaluating [{idx}/10] Topic: {item['topic']}...")
        eval_res = evaluate_response(item)
        results.append(eval_res)

        status_str = "[PASS]" if eval_res["overall_pass"] else "[WARN]"
        print(f"  Query: \"{item['query']}\"")
        print(f"  Top Similarity Score : {eval_res['top_score']:.4f}")
        print(f"  Retrieved SDR IDs    : {eval_res['retrieved_ids']}")
        print(f"  Cited Report IDs     : {eval_res['cited_ids']}")
        print(f"  Report IDs Valid?    : {eval_res['valid_report_ids']}")
        print(f"  Aircraft Grounded?   : {eval_res['aircraft_grounded']}")
        print(f"  Component Grounded?  : {eval_res['comp_grounded']}")
        print(f"  Schema Complete?     : {eval_res['schema_complete']}")
        print(f"  Safety Disclaimers?  : {eval_res['distinguishes_evidence']}")
        print(f"  Status               : {status_str}")

    # Calculate actual empirical summary metrics
    total_queries = len(results)
    top_scores = [r["top_score"] for r in results]
    mean_scores = [r["mean_score"] for r in results]

    valid_id_count = sum(1 for r in results if r["valid_report_ids"])
    aircraft_grounded_count = sum(1 for r in results if r["aircraft_grounded"])
    comp_grounded_count = sum(1 for r in results if r["comp_grounded"])
    schema_complete_count = sum(1 for r in results if r["schema_complete"])
    safe_claims_count = sum(1 for r in results if r["unsupported_claims_avoided"])
    evidence_distinction_count = sum(1 for r in results if r["distinguishes_evidence"])
    passed_count = sum(1 for r in results if r["overall_pass"])

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY & EMPIRICAL METRICS")
    print("=" * 80)
    print(f"Total Queries Evaluated            : {total_queries}")
    print(f"Average Top-1 Retrieval Similarity  : {sum(top_scores)/total_queries:.4f}")
    print(f"Average Mean Top-5 Similarity      : {sum(mean_scores)/total_queries:.4f}")
    print(f"Report ID Grounding Validity Rate  : {valid_id_count}/{total_queries} ({valid_id_count/total_queries * 100:.1f}%)")
    print(f"Aircraft Detail Grounding Rate     : {aircraft_grounded_count}/{total_queries} ({aircraft_grounded_count/total_queries * 100:.1f}%)")
    print(f"Component Detail Grounding Rate    : {comp_grounded_count}/{total_queries} ({comp_grounded_count/total_queries * 100:.1f}%)")
    print(f"Schema Compliance Rate             : {schema_complete_count}/{total_queries} ({schema_complete_count/total_queries * 100:.1f}%)")
    print(f"Zero Unsupported Claims Rate       : {safe_claims_count}/{total_queries} ({safe_claims_count/total_queries * 100:.1f}%)")
    print(f"Historical Evidence Distinction Rate: {evidence_distinction_count}/{total_queries} ({evidence_distinction_count/total_queries * 100:.1f}%)")
    print(f"Overall Grounding Pass Rate        : {passed_count}/{total_queries} ({passed_count/total_queries * 100:.1f}%)")
    print("=" * 80)

    # Save evaluation report to markdown file in backend/llm/evaluation_report.md
    report_file = CURRENT_DIR / "evaluation_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# AeroLLM RAG -> LLM Grounding & Faithfulness Evaluation Report\n\n")
        f.write("## Overview\n")
        f.write(f"- **Evaluated Queries**: {total_queries}\n")
        f.write(f"- **Average Top-1 Similarity Score**: {sum(top_scores)/total_queries:.4f}\n")
        f.write(f"- **Report ID Grounding Rate**: {valid_id_count/total_queries * 100:.1f}%\n")
        f.write(f"- **Aircraft Detail Grounding Rate**: {aircraft_grounded_count/total_queries * 100:.1f}%\n")
        f.write(f"- **Component Detail Grounding Rate**: {comp_grounded_count/total_queries * 100:.1f}%\n")
        f.write(f"- **Schema Compliance Rate**: {schema_complete_count/total_queries * 100:.1f}%\n")
        f.write(f"- **Overall Grounding Pass Rate**: {passed_count/total_queries * 100:.1f}%\n\n")

        f.write("## Detailed Query Evaluation Results\n\n")
        f.write("| # | Topic | Query | Top Similarity | Cited Report IDs | Valid IDs? | Grounded? |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for idx, r in enumerate(results, 1):
            cited_str = ", ".join(r["cited_ids"]) if r["cited_ids"] else "None"
            f.write(f"| {idx} | {r['topic']} | {r['query']} | {r['top_score']:.4f} | {cited_str} | {'Yes' if r['valid_report_ids'] else 'No'} | {'PASS' if r['overall_pass'] else 'WARN'} |\n")

    print(f"\nDetailed Evaluation Report saved to: {report_file}")


if __name__ == "__main__":
    run_evaluation()
