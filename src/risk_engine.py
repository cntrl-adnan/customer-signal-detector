"""
Deterministic risk scoring engine adhering to scoring_v1 rules and confidence gates.
"""
import yaml
import os
from pathlib import Path
from typing import List, Tuple, Optional
from src.models import RawCustomerRecord, SemanticSignal, ScoreContributor

# Repo root, resolved from this file so the engine works from any CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCORING_CONFIG = PROJECT_ROOT / "config" / "scoring_v1.yaml"


class RiskEngine:
    """Calculates explainable 0-100 risk scores and determines contributors."""

    def __init__(self, config_path: Optional[str] = None):
        config_path = str(config_path or DEFAULT_SCORING_CONFIG)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Scoring config not found at: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.points = self.config["points"]
        self.thresholds = self.config["thresholds"]
        self.confidence_threshold = self.config.get("confidence_threshold", 0.60)
        # Stamped onto every scored record so a score stays interpretable
        # after the rules change (DESIGN.md §5).
        self.scoring_version = self.config.get("scoring_name", "scoring_v1")

    def calculate_score(
        self,
        customer: RawCustomerRecord,
        signal: Optional[SemanticSignal]
    ) -> Tuple[int, str, List[ScoreContributor], str]:
        """
        Calculates risk score, band, contributors, and semantic status.
        """
        contributors: List[ScoreContributor] = []
        raw_score = 0

        # Check semantic signal availability & confidence gate
        semantic_status = "available"
        has_valid_semantics = False

        if signal is None:
            semantic_status = "unavailable"
        elif signal.confidence < self.confidence_threshold:
            semantic_status = "needs_review"
        else:
            has_valid_semantics = True

        # --- 1. Semantic Signal Points ---
        # Only a confident, available signal may contribute points. A
        # needs_review (confidence < threshold) or unavailable signal scores on
        # telemetry alone — confidence gates points, it never scales them.
        if has_valid_semantics:
            # Churn Intent (+25)
            if signal.churn_intent:
                pts = self.points.get("churn_intent", 25)
                raw_score += pts
                contributors.append(ScoreContributor(
                    signal_name="churn_intent",
                    rule_description="Explicit churn intent detected in interaction",
                    points=pts
                ))

            # Urgency (+15 / +10 / +5)
            urgency_pts = self.points.get("urgency", {}).get(signal.urgency, 0)
            if urgency_pts > 0:
                raw_score += urgency_pts
                contributors.append(ScoreContributor(
                    signal_name="urgency",
                    rule_description=f"{signal.urgency.capitalize()} operational urgency",
                    points=urgency_pts
                ))

            # Sentiment (+15 / +10)
            sentiment_pts = self.points.get("sentiment", {}).get(signal.sentiment, 0)
            if sentiment_pts > 0:
                raw_score += sentiment_pts
                contributors.append(ScoreContributor(
                    signal_name="sentiment",
                    rule_description=f"{signal.sentiment.replace('_', ' ').capitalize()} sentiment tone",
                    points=sentiment_pts
                ))

        # --- 2. Structured Data Points ---

        # Payment status (+15 / +12)
        payment_pts = self.points.get("payment_status", {}).get(customer.payment_status, 0)
        if payment_pts > 0:
            raw_score += payment_pts
            contributors.append(ScoreContributor(
                signal_name="payment_status",
                rule_description=f"Payment status is {customer.payment_status}",
                points=payment_pts
            ))

        # Failed payments (min(failed_payments_30d, 2) * 4)
        failed_cap = self.points.get("failed_payments_max_cap", 2)
        failed_mult = self.points.get("failed_payments_multiplier", 4)
        failed_pts = min(customer.failed_payments_30d, failed_cap) * failed_mult
        if failed_pts > 0:
            raw_score += failed_pts
            contributors.append(ScoreContributor(
                signal_name="failed_payments",
                rule_description=f"{customer.failed_payments_30d} failed payment attempts (30d)",
                points=failed_pts
            ))

        # Recent issues (min(issue_count_30d, 3) * 4)
        issues_cap = self.points.get("recent_issues_max_cap", 3)
        issues_mult = self.points.get("recent_issues_multiplier", 4)
        issues_pts = min(customer.issue_count_30d, issues_cap) * issues_mult
        if issues_pts > 0:
            raw_score += issues_pts
            contributors.append(ScoreContributor(
                signal_name="recent_issues",
                rule_description=f"{customer.issue_count_30d} logged support issues (30d)",
                points=issues_pts
            ))

        # Usage decline (<= -50%: +15, -25% to -49%: +10)
        usage_cfg = self.points.get("usage_decline", {})
        if customer.usage_change_pct <= usage_cfg.get("severe_threshold", -50.0):
            pts = usage_cfg.get("severe_points", 15)
            raw_score += pts
            contributors.append(ScoreContributor(
                signal_name="usage_decline",
                rule_description=f"Severe usage drop ({customer.usage_change_pct}%)",
                points=pts
            ))
        elif customer.usage_change_pct <= usage_cfg.get("moderate_threshold", -25.0):
            pts = usage_cfg.get("moderate_points", 10)
            raw_score += pts
            contributors.append(ScoreContributor(
                signal_name="usage_decline",
                rule_description=f"Moderate usage drop ({customer.usage_change_pct}%)",
                points=pts
            ))

        # CSAT (<= 2: +10, 3: +5)
        csat_cfg = self.points.get("csat", {})
        if customer.csat_score is not None:
            if customer.csat_score <= csat_cfg.get("very_low_threshold", 2):
                pts = csat_cfg.get("very_low_points", 10)
                raw_score += pts
                contributors.append(ScoreContributor(
                    signal_name="csat_score",
                    rule_description=f"Low satisfaction score ({customer.csat_score}/5)",
                    points=pts
                ))
            elif customer.csat_score == csat_cfg.get("low_threshold", 3):
                pts = csat_cfg.get("low_points", 5)
                raw_score += pts
                contributors.append(ScoreContributor(
                    signal_name="csat_score",
                    rule_description=f"Neutral satisfaction score ({customer.csat_score}/5)",
                    points=pts
                ))

        # Clamp score between 0 and 100
        final_score = min(100, max(0, raw_score))

        # Determine Risk Band
        if final_score >= self.thresholds.get("critical", 75):
            band = "Critical"
        elif final_score >= self.thresholds.get("high", 50):
            band = "High"
        elif final_score >= self.thresholds.get("medium", 25):
            band = "Medium"
        else:
            band = "Low"

        # Sort contributors in descending order of points
        contributors.sort(key=lambda c: c.points, reverse=True)

        return final_score, band, contributors, semantic_status