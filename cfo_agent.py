import json
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from schemas import CFOAnalysis


# CONFIG

load_dotenv()

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEPRECATED_GROQ_MODELS = {
    "llama-3.1-8b-instant",
}

configured_model = os.getenv(
    "GROQ_MODEL",
    DEFAULT_GROQ_MODEL,
).strip()

if configured_model in DEPRECATED_GROQ_MODELS:
    print(
        f"Warning: GROQ_MODEL={configured_model!r} is deprecated on Groq. "
        f"Using {DEFAULT_GROQ_MODEL!r} instead.",
        file=sys.stderr,
    )
    configured_model = DEFAULT_GROQ_MODEL


llm = ChatGroq(
    model=configured_model,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_tokens=1000,
    include_reasoning=False,
    reasoning_effort="low",
)


# SYSTEM PROMPT

SYSTEM_PROMPT = """
You are the CFO synthesis agent for an automotive business.

You receive two completed analyses:
1. Automotive registration evidence from VAHAN.
2. Macroeconomic evidence from macroeconomic tools.

Rules:
- Use only the supplied analyses.
- Do not invent numbers or claim that one factor caused another.
- Connect registration demand, financing, input costs, energy, and currency
  only when supported by the supplied evidence.
- Make the distinction between evidence and recommendation clear.
- If one analysis is missing or contains caveats, preserve that limitation.
- Keep every list item concise.
- Populate every schema field.
- Use [] for any list field when the supplied evidence does not support it.
- Every list field must contain strings only.
- Never add fields outside the provided schema.
"""


def synthesize_analysis(
    user_query: str,
    automotive_analysis,
    macro_analysis,
) -> CFOAnalysis:
    """
    Combine specialist outputs into the final CFOAnalysis using Groq's
    strict JSON Schema structured-output mode.
    """

    structured_llm = llm.with_structured_output(
        CFOAnalysis,
        method="json_schema",
        strict=True,
    )

    evidence = {
        "user_query": user_query,
        "automotive_analysis": automotive_analysis.model_dump(),
        "macro_analysis": macro_analysis.model_dump(),
    }

    return structured_llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=json.dumps(
                evidence,
                ensure_ascii=False,
            )
        ),
    ])