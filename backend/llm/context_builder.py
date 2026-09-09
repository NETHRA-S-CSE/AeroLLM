from typing import List, Dict, Any

def build_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    """
    Converts a list of retrieved document dictionaries into a clean, compact text context block for LLM prompting.

    Args:
        retrieved_docs (List[Dict[str, Any]]): Documents returned by backend.rag.retriever.retrieve()

    Returns:
        str: Formatted context string containing structured details for each retrieved report.
    """
    if not retrieved_docs:
        return "No relevant historical maintenance reports found."

    context_blocks = []
    for idx, doc in enumerate(retrieved_docs, 1):
        report_id = doc.get("id", "N/A")
        score = doc.get("score", 0.0)
        meta = doc.get("metadata", {})

        make = meta.get("aircraft_make", "Not available")
        model = meta.get("aircraft_model", "Not available")
        aircraft = f"{make} {model}".strip() if make != "Not available" or model != "Not available" else "Not available"

        jasc_code = meta.get("jasc_code", "Not available")
        part_name = meta.get("part_name", "Not available")
        part_number = meta.get("part_number", "Not available")
        component_name = meta.get("component_name", "Not available")
        
        comp_details = []
        if component_name != "Not available":
            comp_details.append(f"Component: {component_name}")
        if part_name != "Not available":
            comp_details.append(f"Part: {part_name}")
        if part_number != "Not available":
            comp_details.append(f"P/N: {part_number}")
        
        component_summary = " | ".join(comp_details) if comp_details else "Not available"
        raw_text = doc.get("text", "").strip()

        block = (
            f"REPORT ID: {report_id}\n"
            f"Similarity: {score:.4f}\n"
            f"Aircraft: {aircraft}\n"
            f"JASC Code: {jasc_code}\n"
            f"Component / Part: {component_summary}\n"
            f"Report Content:\n{raw_text}\n"
        )
        context_blocks.append(f"--- DOCUMENT [{idx}] ---\n" + block)

    return "\n".join(context_blocks)
