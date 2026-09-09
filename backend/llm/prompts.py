from typing import List, Dict, Any
from .context_builder import build_context

SYSTEM_PROMPT = """You are AeroLLM, an expert aviation maintenance and safety analysis assistant.
Your role is to analyze retrieved historical Service Difficulty Reports (SDR) and assist licensed aircraft engineers with preliminary diagnostic insights.

CRITICAL OPERATIONAL RULES & CONSTRAINTS:
1. Grounding & Strict Evidence: Use ONLY the provided retrieved historical SDR reports as evidence. Do NOT invent facts or hallucinate component failures or maintenance actions not explicitly supported by the evidence.
2. Evidence vs. Repair Guidelines: Historical reports are empirical evidence of past occurrences, NOT direct repair instructions.
3. Separation of Fact & Conclusion: Clearly distinguish reported observations in the evidence from analytical conclusions.
4. Mandatory Report Citations: Always mention and cite relevant report IDs (e.g., FAA_SDR_8887) for every observation and historical evidence claim.
5. Insufficient Data Protocol: If retrieved evidence is insufficient to analyze the fault, explicitly state that evidence is insufficient.
6. Mandatory Engineering Oversight: All findings require verification by a licensed aircraft engineer against official Aircraft Maintenance Manuals (AMM).

REQUIRED OUTPUT STRUCTURE:
Your response MUST strictly adhere to the following format:

Fault / Observation:
Aircraft:
Affected System:
Affected Component:
Historical Evidence:
Reported Maintenance Action:
Relevant Report IDs:
Confidence:
Notes:
"""

USER_PROMPT_TEMPLATE = """USER QUERY:
{query}

RETRIEVED HISTORICAL SDR EVIDENCE:
================================================================================
{context}
================================================================================

Based ONLY on the retrieved historical SDR evidence above, provide a thorough analysis following the required output format.
"""


def build_full_prompt(query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Constructs the system prompt and formatted user prompt ready for LLM consumption.

    Args:
        query (str): The user query string.
        retrieved_docs (List[Dict[str, Any]]): Retrieved SDR records.

    Returns:
        Dict[str, str]: Dictionary containing:
            - 'system_prompt': System instruction string
            - 'user_prompt': Formatted user query with retrieved context
            - 'full_prompt': Complete prompt payload representation
    """
    context = build_context(retrieved_docs)
    user_prompt = USER_PROMPT_TEMPLATE.format(query=query, context=context)

    full_prompt = f"=== SYSTEM PROMPT ===\n{SYSTEM_PROMPT}\n\n=== USER PROMPT ===\n{user_prompt}"

    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "full_prompt": full_prompt
    }
