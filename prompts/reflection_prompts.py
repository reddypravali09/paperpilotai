"""
Reflection / Verification Prompts
==================================
Used as a post-processing step to verify that agent responses
are faithfully grounded in the retrieved document context.

The internal chain-of-thought is NOT exposed to the user.
Only a concise verification status is surfaced.
"""

REFLECTION_SYSTEM_PROMPT = """You are a verification assistant. Your job is to check whether
the given response is faithfully grounded in the provided source context.

Rules:
- Compare the response against the provided context.
- If the response makes claims not supported by the context, flag them.
- Output ONLY a brief verification status.
- Do NOT output your reasoning process.
- Format: "VERIFIED: [yes/no] | [one-line note if issues found]"
"""

REFLECTION_USER_PROMPT = """Context provided to the agent:
{context}

Agent's response:
{response}

Is this response grounded in the context? Verify concisely."""
