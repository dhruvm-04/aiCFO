import json
import sys

from langgraph.graph import StateGraph, START, END

from state import CFOState
from automotive_agent import ask_automotive_agent
from cfo_agent import synthesize_analysis
from macro_agent import ask_macro_agent


def automotive_node(state: CFOState):
    query = state["user_query"]

    result = ask_automotive_agent(query)

    return {
        "automotive_analysis": result
    }


def macro_node(state: CFOState):
    query = state["user_query"]

    result = ask_macro_agent(query)

    return {
        "macro_analysis": result
    }


def cfo_synthesis_node(state: CFOState):
    result = synthesize_analysis(
        user_query=state["user_query"],
        automotive_analysis=state["automotive_analysis"],
        macro_analysis=state["macro_analysis"],
    )

    return {
        "cfo_analysis": result
    }


# Build graph

builder = StateGraph(CFOState)

builder.add_node("automotive_agent", automotive_node)
builder.add_node("macro_agent", macro_node)
builder.add_node("cfo_synthesis", cfo_synthesis_node)

# Both agents start from the same user request.
builder.add_edge(START, "automotive_agent")
builder.add_edge(START, "macro_agent")

# Fan in after both specialists finish so synthesis receives both outputs.
builder.add_edge("automotive_agent", "cfo_synthesis")
builder.add_edge("macro_agent", "cfo_synthesis")
builder.add_edge("cfo_synthesis", END)

graph = builder.compile()

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip()

    if not query:
        print(
            "Enter your aiCFO analysis prompt: ",
            end="",
            file=sys.stderr,
        )
        query = sys.stdin.readline().strip()

    if not query:
        raise SystemExit(
            "A non-empty analysis prompt is required."
        )

    result = graph.invoke({
        "user_query": query
    })

    output = {
        "user_query": query,
        "automotive_analysis": result.get(
            "automotive_analysis"
        ).model_dump(),
        "macro_analysis": result.get(
            "macro_analysis"
        ).model_dump(),
        "cfo_analysis": result.get(
            "cfo_analysis"
        ).model_dump(),
    }

    print(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
    )