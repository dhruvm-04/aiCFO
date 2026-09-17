from typing import TypedDict, Optional
from schemas import AutomotiveAnalysis, CFOAnalysis, MacroAnalysis

class CFOState(TypedDict, total=False):
    # User request entering the graph
    user_query: str

    # Agent outputs written to the blackboard
    automotive_analysis: Optional[AutomotiveAnalysis]
    macro_analysis: Optional[MacroAnalysis]

    # Combined analysis produced after both specialist agents finish
    cfo_analysis: Optional[CFOAnalysis]

    # Basic execution metadata
    errors: list[str]