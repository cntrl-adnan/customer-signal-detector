import pandas as pd
from src.models import RawCustomerRecord, SemanticSignal
from src.risk_engine import RiskEngine


def test_deterministic_scoring_rules():
    engine = RiskEngine()

    # Critical Churn Customer: churn intent (25) + critical urgency (15) + very negative (15) + failed pay (15) + 2 failed (8) + 3 issues (12) + severe usage (15) + csat 1 (10)
    customer = RawCustomerRecord(
        customer_id="TEST-CRITICAL",
        interaction_text="Cancel my subscription immediately.",
        issue_count_30d=4,
        payment_status="failed",
        failed_payments_30d=2,
        usage_change_pct=-55.0,
        csat_score=1
    )
    signal = SemanticSignal(
        sentiment="very_negative",
        issue_type="cancellation",
        urgency="critical",
        churn_intent=True,
        confidence=0.95,
        evidence_spans=["Cancel my subscription immediately"],
        summary="Customer wants to cancel"
    )

    score, band, contributors, status = engine.calculate_score(customer, signal)

    # Max points sum is 115, capped at 100
    assert score == 100
    assert band == "Critical"
    assert status == "available"
    assert len(contributors) >= 5
    assert contributors[0].points >= contributors[1].points  # Descending order


def test_confidence_gate_excludes_semantic_points():
    """
    A low-confidence signal must contribute ZERO semantic points.

    DESIGN.md §4: "below 0.60, semantic points are excluded entirely and the
    row is marked Needs review; it scores on telemetry alone."

    The signal below is deliberately high-scoring (churn +25, critical urgency
    +15, very_negative +15 = 55 semantic points) but sits below the confidence
    threshold. Only the telemetry (overdue +12) may count.
    """
    engine = RiskEngine()
    customer = RawCustomerRecord(
        customer_id="TEST-LOW-CONF",
        interaction_text="Something vague and unclear about our account status.",
        issue_count_30d=0,
        payment_status="overdue",   # +12, the only points that should land
        failed_payments_30d=0,
        usage_change_pct=0.0,
        csat_score=None,
    )
    low_conf_signal = SemanticSignal(
        sentiment="very_negative",
        issue_type="cancellation",
        urgency="critical",
        churn_intent=True,
        confidence=0.45,            # below the 0.60 gate
        summary="Ambiguous note the model was not confident about",
    )

    score, band, contributors, status = engine.calculate_score(customer, low_conf_signal)

    assert status == "needs_review"

    # Telemetry alone: overdue payment = 12 points
    assert score == 12, f"expected telemetry-only score of 12, got {score}"
    assert band == "Low"

    # No semantic signal may appear among the contributors
    semantic_names = {"churn_intent", "urgency", "sentiment"}
    leaked = [c.signal_name for c in contributors if c.signal_name in semantic_names]
    assert not leaked, f"low-confidence semantic points leaked into score: {leaked}"


def test_semantic_unavailable_scores_on_telemetry_only():
    """A failed/absent semantic analysis must still produce a telemetry score."""
    engine = RiskEngine()
    customer = RawCustomerRecord(
        customer_id="TEST-NO-SIGNAL",
        interaction_text="A perfectly ordinary support message about our setup.",
        issue_count_30d=3,          # min(3,3)*4 = +12
        payment_status="failed",    # +15
        failed_payments_30d=1,      # min(1,2)*4 = +4
        usage_change_pct=-60.0,     # +15
        csat_score=2,               # +10
    )

    score, band, contributors, status = engine.calculate_score(customer, None)

    assert status == "unavailable"
    assert score == 56
    assert band == "High"
    assert all(c.signal_name not in {"churn_intent", "urgency", "sentiment"} for c in contributors)