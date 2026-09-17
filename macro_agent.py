import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq

from schemas import MacroAnalysis

from macro_tools import (
    get_repo_rate,
    get_inflation,
    get_wpi_metrics,
    get_iip_motor_vehicles,
    get_forex_rates,
    get_brent,
    get_macro_snapshot,
)


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
    max_tokens=900,
    include_reasoning=False,
    reasoning_effort="low",
)


# TOOLS

MACRO_TOOLS = [
    get_repo_rate,
    get_inflation,
    get_wpi_metrics,
    get_iip_motor_vehicles,
    get_forex_rates,
    get_brent,
    get_macro_snapshot,
]

TOOLS_BY_NAME = {
    tool.name: tool
    for tool in MACRO_TOOLS
}


# SYSTEM PROMPT

SYSTEM_PROMPT = """
You are the Macroeconomic Intelligence Agent for aiCFO.

Your source of truth is the macroeconomic data exposed through your tools.

Your domain includes:
- RBI repo rate
- CPI inflation
- Steel, rubber and aluminium WPI
- IIP Motor Vehicles
- USD/INR, EUR/INR and JPY/INR
- Brent crude

Your job is to determine which macroeconomic data is relevant to the user's
question and interpret it in an automotive context.

Rules:
1. Use tools whenever factual data is required.
2. Never invent numbers.
3. Prefer specific tools when the question concerns one metric.
4. Use get_macro_snapshot when the question asks for a broad overview of the
   automotive macro environment.
5. Distinguish observed data from your interpretation.
6. Do not treat a correlation as proof of causation.
7. Pay attention to the reporting period of every metric.
8. If information is missing, say so.
9. Do not use VAHAN-specific analysis here; that belongs to the Automotive Agent.
10. Keep the reasoning focused on automotive relevance.
"""


# AGENT

def create_agent():
    """Create the Macro Agent with access to macro tools."""

    return llm.bind_tools(MACRO_TOOLS)


# TOOL-CALLING LOOP

def collect_macro_context(
    query: str,
    max_iterations: int = 5,
):
    """
    Let Groq select and use macroeconomic tools.

    Returns the conversation containing:
        - the question
        - model tool calls
        - tool results
    """

    agent = create_agent()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    for _ in range(max_iterations):
        model_messages = messages

        if len(messages) > 6:
            model_messages = messages[:2] + messages[-4:]

        response = agent.invoke(model_messages)
        messages.append(response)

        if not response.tool_calls:
            return messages

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            tool = TOOLS_BY_NAME.get(tool_name)

            if tool is None:
                tool_result = {
                    "error": f"Unknown tool: {tool_name}"
                }
            else:
                try:
                    tool_result = tool.invoke(tool_args)
                except Exception as exc:
                    tool_result = {
                        "error": str(exc)
                    }

            tool_result_text = str(tool_result)

            if len(tool_result_text) > 3000:
                tool_result_text = (
                    tool_result_text[:3000]
                    + "\n[Tool result truncated for model context.]"
                )

            messages.append(
                ToolMessage(
                    content=tool_result_text,
                    tool_call_id=tool_call_id,
                )
            )

    print(
        f"Warning: Macro Agent reached the {max_iterations} "
        "tool-call iterations. Synthesizing from collected macro evidence.",
        file=sys.stderr,
    )

    return messages


# STRUCTURED ANALYSIS

def extract_macro_evidence(messages) -> str:
    """Extract bounded macro tool results for final synthesis."""

    evidence = []

    for message in messages:
        if isinstance(message, ToolMessage):
            evidence.append(message.content)

    return "\n\n".join(evidence)[:6000]


def generate_analysis(
    query: str,
    messages,
) -> MacroAnalysis:
    """
    Convert retrieved macroeconomic evidence into a structured MacroAnalysis.
    """

    macro_evidence = extract_macro_evidence(messages)

    if not macro_evidence.strip():
        return MacroAnalysis(
            summary="No macroeconomic evidence was retrieved for this request.",
            demand_environment=[],
            cost_environment=[],
            financing_environment=[],
            currency_environment=[],
            key_metrics=[],
            caveats=[
                "No macroeconomic tool returned usable evidence for the request."
            ],
        )

    structured_llm = llm.with_structured_output(
        MacroAnalysis,
        method="json_schema",
        strict=True,
    )

    analysis_messages = [
        SystemMessage(
            content=(
                "You are the final Macroeconomic Intelligence analyst for "
                "an automotive business.\n\n"
                "Use only the macroeconomic evidence provided below.\n"
                "Do not invent missing numbers.\n"
                "Separate factual observations from interpretation.\n"
                "Explain why observed macro conditions matter for automotive "
                "demand, financing, manufacturing, input costs, or currency "
                "exposure when supported by the evidence.\n"
                "Mention important reporting-period differences or missing "
                "information.\n\n"
                "Populate all schema fields. Use [] for a list field when the "
                "evidence does not support any items.\n"
                "Every list field must contain strings only.\n"
                "Never place a single string directly into a list field.\n"
                "Do not add fields outside the provided schema.\n\n"
                "Return no more than 3 items in each environment list, no more "
                "than 6 key_metrics, and no more than 3 caveats.\n"
                "Keep the summary under 40 words."
            )
        ),
        HumanMessage(
            content=(
                f"Original question:\n{query}\n\n"
                f"Macro evidence:\n{macro_evidence}"
            )
        ),
    ]

    return structured_llm.invoke(analysis_messages)


# PUBLIC API

def ask_macro_agent(
    query: str,
) -> MacroAnalysis:
    """Ask the Macro Agent a question."""

    messages = collect_macro_context(query)

    return generate_analysis(
        query=query,
        messages=messages,
    )


# TEST

if __name__ == "__main__":
    question = (
        "What is the current macroeconomic environment "
        "for the Indian automotive sector?"
    )

    result = ask_macro_agent(question)

    print("\n")
    print("=" * 70)
    print("MACROECONOMIC ANALYSIS")
    print("=" * 70)
    print()
    print(result.model_dump_json(indent=2))