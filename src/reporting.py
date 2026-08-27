"""
Generates human-readable explainable rationale strictly from score contributors.
"""
from typing import List, Optional
from src.models import ScoreContributor, SemanticSignal


def generate_human_rationale(
    customer_id: str,
    risk_band: str,
    contributors: List[ScoreContributor],
    signal: Optional[SemanticSignal]
) -> str:
    """
    Generates explainable rationale in code from score contributors.
    """
    if not contributors:
        return f"{risk_band}: {customer_id} shows healthy engagement with no active risk contributors."

    top_reasons = [c.rule_description.lower() for c in contributors[:3]]
    joined_reasons = "; ".join(top_reasons)

    evidence_str = ""
    if signal and signal.evidence_spans:
        clean_excerpt = signal.evidence_spans[0].strip().replace("\n", " ")
        if len(clean_excerpt) > 100:
            clean_excerpt = clean_excerpt[:97] + "..."
        evidence_str = f" Semantic clue: \"{clean_excerpt}\""

    return f"{risk_band}: {customer_id} was prioritized due to {joined_reasons}.{evidence_str}"