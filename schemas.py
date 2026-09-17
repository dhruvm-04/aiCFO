from pydantic import BaseModel, ConfigDict, Field

class AutomotiveAnalysis(BaseModel):
    """Structured output for the Automotive Intelligence Agent."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        description="Short summary of the automotive situation."
    )

    key_findings: list[str] = Field(
        description=(
            "Important findings directly supported by VAHAN data."
        )
    )

    trends: list[str] = Field(
        description="Relevant registration trends and changes."
    )

    data_points: list[str] = Field(
        description=(
            "Important numerical observations used in the analysis."
        )
    )

    caveats: list[str] = Field(
        description=(
            "Data limitations, ambiguity, missing information, "
            "or important qualifications."
        )
    )

class MacroAnalysis(BaseModel):
    """Structured output for the Macroeconomic Intelligence Agent."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        description=(
            "Short summary of the current macroeconomic environment "
            "relevant to the Indian automotive sector."
        )
    )

    demand_environment: list[str] = Field(
        description=(
            "Macro indicators affecting automotive demand. "
            "Only make demand claims when directly supported by the data."
        )
    )

    cost_environment: list[str] = Field(
        description=(
            "Macro indicators affecting automotive input, energy, "
            "logistics, and operating costs."
        )
    )

    financing_environment: list[str] = Field(
        description=(
            "Monetary-policy and financing conditions relevant to "
            "automotive manufacturers and consumers."
        )
    )

    currency_environment: list[str] = Field(
        description=(
            "Foreign-exchange conditions and their potential relevance "
            "to automotive imports and procurement."
        )
    )

    key_metrics: list[str] = Field(
        description=(
            "Important macroeconomic numerical observations used "
            "in the analysis."
        )
    )

    caveats: list[str] = Field(
        description=(
            "Data limitations, missing indicators, timing differences, "
            "or other important qualifications."
        )
    )


class CFOAnalysis(BaseModel):
    """Structured output for the final CFO synthesis agent."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        description=(
            "Short combined decision summary for the automotive business."
        )
    )

    cross_domain_insights: list[str] = Field(
        description=(
            "Insights supported by both automotive and macro analysis."
        )
    )

    recommendations: list[str] = Field(
        description="Practical actions supported by the combined evidence."
    )

    caveats: list[str] = Field(
        description=(
            "Important limitations in the combined analysis."
        )
    )