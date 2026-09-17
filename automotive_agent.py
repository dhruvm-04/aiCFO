import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq
from schemas import AutomotiveAnalysis
from automotive_tools import (
    get_fuel_data,
    get_fuel_trend,
    get_maker_data,
    get_maker_trend,
    get_norms_data,
    get_norms_trend,
    get_vehicle_category_data,
    get_vehicle_category_trend,
    get_vehicle_class_data,
    get_vehicle_class_trend,
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
    max_tokens=800,
    include_reasoning=False,
    reasoning_effort="low",
)

# TOOLS

VAHAN_TOOLS = [
    get_maker_data,
    get_maker_trend,
    get_fuel_data,
    get_fuel_trend,
    get_vehicle_category_data,
    get_vehicle_category_trend,
    get_vehicle_class_data,
    get_vehicle_class_trend,
    get_norms_data,
    get_norms_trend,
]

TOOLS_BY_NAME = {
    tool.name: tool
    for tool in VAHAN_TOOLS
}

# PROMPT

SYSTEM_PROMPT = """
You are the Automotive Intelligence Agent for aiCFO.

Your only source of truth is the local VAHAN dataset exposed through
your tools.

You can analyze:
- vehicle manufacturers / makers
- fuel types
- vehicle categories
- vehicle classes
- emission norms
- historical trends

Rules:
1. Use VAHAN tools whenever factual data is required.
2. Never invent VAHAN numbers.
3. Prefer trend tools for questions about changes over time.
4. Prefer data tools for questions about a specific year.
5. Pay attention to partial years.
6. Do not silently combine multiple VAHAN entities.
7. Clearly distinguish facts from interpretation.
8. Keep findings grounded in the retrieved data.
9. Mention important data limitations.
10. Do not answer from general knowledge when VAHAN data is required.
11. When a question concerns macroeconomics in the context of the automotive
    sector, provide the Automotive Agent's VAHAN perspective as well.
12. Use relevant VAHAN tools whenever the request contains an automotive
    entity, category, fuel, class, norm, year, or trend.
13. Only return no VAHAN findings when the request truly contains no usable
    automotive dimension.
"""

# AGENT

def create_agent():
    """Create the tool-calling Automotive Agent."""

    return llm.bind_tools(VAHAN_TOOLS)

# TOOL-CALLING LOOP

def collect_vahan_context(
    query: str,
    max_iterations: int = 5,
):
    """
    Let the LLM decide which VAHAN tools to call.

    Returns the complete conversation containing:
        - user question
        - tool calls
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

        # No more tools required.
        if not response.tool_calls:
            return messages

        # Execute requested tools.
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
        f"Warning: Automotive Agent reached the {max_iterations} "
        "tool-call iterations. Synthesizing from collected VAHAN evidence.",
        file=sys.stderr,
    )

    return messages

def extract_vahan_evidence(messages) -> str:
    """
    Extract only tool results from the research conversation.
    This keeps the final structured-output request small.
    """

    evidence = []

    for message in messages:
        if isinstance(message, ToolMessage):
            evidence.append(message.content)

    return "\n\n".join(evidence)[:6000]

# STRUCTURED ANALYSIS

def generate_analysis(
    query: str,
    messages,
) -> AutomotiveAnalysis:
    """
    Convert collected VAHAN evidence into structured AutomotiveAnalysis.

    Groq's strict JSON Schema mode is used here so Pydantic list fields are
    enforced by the model/API instead of relying on prompt instructions alone.
    """

    vahan_evidence = extract_vahan_evidence(messages)

    if not vahan_evidence.strip():
        return AutomotiveAnalysis(
            summary=(
                "The Automotive Agent found no VAHAN evidence applicable "
                "to this request."
            ),
            key_findings=[],
            trends=[],
            data_points=[],
            caveats=[
                "No automotive entity, category, fuel, class, norm, year, "
                "or trend was specific enough to query in VAHAN."
            ],
        )

    structured_llm = llm.with_structured_output(
        AutomotiveAnalysis,
        method="json_schema",
        strict=True,
    )

    analysis_messages = [
        SystemMessage(
            content=(
                "You are the final Automotive Intelligence analyst.\n\n"
                "Use only the VAHAN evidence provided below.\n"
                "Do not invent missing numbers.\n"
                "Keep facts separate from interpretation.\n\n"
                "If the question is primarily macroeconomic but has an "
                "automotive dimension, summarize only the relevant VAHAN "
                "evidence. Do not make macroeconomic claims unless they are "
                "supported by the provided VAHAN evidence.\n\n"
                "Populate all schema fields. Use [] for a list field when the "
                "evidence does not support any items.\n"
                "Every list field must contain strings only.\n"
                "Never place a single string directly into a list field.\n"
                "Do not add fields outside the provided schema.\n\n"
                "Include important numerical values in data_points.\n"
                "Mention missing or partial data in caveats.\n"
                "Keep every list item concise."
            )
        ),
        HumanMessage(
            content=(
                f"Original question:\n{query}\n\n"
                f"VAHAN evidence:\n{vahan_evidence}"
            )
        ),
    ]

    return structured_llm.invoke(analysis_messages)

# MAIN AGENT FUNCTION

def ask_automotive_agent(
    query: str,
) -> AutomotiveAnalysis:
    messages = collect_vahan_context(query)

    return generate_analysis(
        query=query,
        messages=messages,
    )