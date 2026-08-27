"""
Pydantic data models enforcing input, semantic signal, and scored output contracts.
"""
from typing import Literal, Optional, List
from pydantic import BaseModel, Field, field_validator


# --- Input Contract ---
class RawCustomerRecord(BaseModel):
    customer_id: str = Field(description="Unique non-empty customer identifier")
    customer_name: Optional[str] = Field(default="De-identified Customer")
    interaction_text: str = Field(description="Support exchange or feedback (20-4000 chars)")
    issue_count_30d: int = Field(default=0, ge=0, le=99)
    payment_status: Literal["current", "overdue", "failed", "unknown"] = Field(default="current")
    failed_payments_30d: int = Field(default=0, ge=0, le=10)
    usage_change_pct: float = Field(default=0.0, ge=-100.0, le=100.0)
    csat_score: Optional[int] = Field(default=None, ge=1, le=5)
    last_contact_days: Optional[int] = Field(default=0, ge=0, le=365)

    @field_validator("customer_id")
    @classmethod
    def validate_customer_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("customer_id cannot be empty")
        return v.strip()

    @field_validator("interaction_text")
    @classmethod
    def validate_interaction_text(cls, v: str) -> str:
        # Contract (DESIGN.md §5): 20-4000 characters.
        stripped = v.strip() if v else ""
        if len(stripped) < 20:
            raise ValueError("interaction_text must be at least 20 characters")
        if len(stripped) > 4000:
            raise ValueError("interaction_text must be at most 4000 characters")
        return stripped


# --- Semantic Analysis Contract (from LLM) ---
class SemanticSignal(BaseModel):
    sentiment: Literal["very_negative", "negative", "neutral", "positive"]
    issue_type: Literal["billing", "support", "product", "usage", "cancellation", "other"]
    urgency: Literal["low", "medium", "high", "critical"]
    churn_intent: bool
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0 to 1.0")
    evidence_spans: List[str] = Field(default_factory=list, max_length=2, description="Up to 2 short excerpts")
    summary: str = Field(max_length=160, description="<= 160 character summary")


# --- LLM call outcome (honesty contract) ---
# Distinguishes a real analysis from an explicitly-labelled mock and from a
# failure. A failure must NEVER be silently replaced with fabricated analysis.
class SemanticResult(BaseModel):
    signal: Optional[SemanticSignal] = None
    source: Literal["live", "mock", "failed"]
    error: Optional[str] = Field(default=None, description="Provider error, for operator display only")


# --- Score Contributor Item ---
class ScoreContributor(BaseModel):
    signal_name: str
    rule_description: str
    points: int


# --- Complete Analyzed Customer Output ---
class AnalyzedCustomerRecord(BaseModel):
    customer_id: str
    customer_name: str
    interaction_text: str
    
    # Input structured fields
    issue_count_30d: int
    payment_status: str
    failed_payments_30d: int
    usage_change_pct: float
    csat_score: Optional[int]
    last_contact_days: Optional[int] = None

    # Semantic fields (or None if unavailable)
    semantic_status: Literal["available", "needs_review", "unavailable"]
    semantic_signal: Optional[SemanticSignal] = None
    analysis_source: Literal["live", "mock", "failed"] = "failed"
    semantic_error: Optional[str] = None

    # Deterministic scoring
    risk_score: int = Field(ge=0, le=100)
    risk_band: Literal["Critical", "High", "Medium", "Low"]
    contributors: List[ScoreContributor]
    scoring_version: str = "scoring_v1"

    # Suggested outcomes (prototype recommendations, require human review)
    rationale: str
    suggested_action: str