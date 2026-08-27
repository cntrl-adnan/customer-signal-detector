"""
Orchestrates data loading, LLM extraction, deterministic scoring, and reporting into AnalyzedCustomerRecords.
"""
from typing import List, Dict, Any, Tuple, Optional
from src.models import RawCustomerRecord, AnalyzedCustomerRecord
from src.data_loader import DataLoader
from src.llm_service import LLMService
from src.risk_engine import RiskEngine
from src.recommendation_engine import RecommendationEngine
from src.reporting import generate_human_rationale


class SignalAnalyzer:
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        risk_engine: Optional[RiskEngine] = None,
        rec_engine: Optional[RecommendationEngine] = None,
        mock_mode: bool = False
    ):
        self.llm_service = llm_service or LLMService(mock_mode=mock_mode)
        self.risk_engine = risk_engine or RiskEngine()
        self.rec_engine = rec_engine or RecommendationEngine()

    def analyze_record(self, raw: RawCustomerRecord) -> AnalyzedCustomerRecord:
        # 1. Semantic extraction — returns an explicit live/mock/failed outcome.
        #    On failure `signal` is None; we never substitute a fabricated one.
        result = self.llm_service.analyze_transcript(raw.interaction_text)
        signal = result.signal

        # 2. Deterministic risk calculation (semantic signal + telemetry)
        risk_score, risk_band, contributors, semantic_status = self.risk_engine.calculate_score(raw, signal)

        # 3. Suggested action & rationale, both derived deterministically
        issue_type = signal.issue_type if signal else "other"
        action = self.rec_engine.get_action(issue_type, risk_band)
        rationale = generate_human_rationale(raw.customer_id, risk_band, contributors, signal)

        return AnalyzedCustomerRecord(
            customer_id=raw.customer_id,
            customer_name=raw.customer_name or raw.customer_id,
            interaction_text=raw.interaction_text,
            issue_count_30d=raw.issue_count_30d,
            payment_status=raw.payment_status,
            failed_payments_30d=raw.failed_payments_30d,
            usage_change_pct=raw.usage_change_pct,
            csat_score=raw.csat_score,
            last_contact_days=raw.last_contact_days,
            semantic_status=semantic_status,
            semantic_signal=signal,
            analysis_source=result.source,
            semantic_error=result.error,
            risk_score=risk_score,
            risk_band=risk_band,
            contributors=contributors,
            scoring_version=self.risk_engine.scoring_version,
            rationale=rationale,
            suggested_action=action
        )

    def analyze_batch(self, records: List[RawCustomerRecord]) -> List[AnalyzedCustomerRecord]:
        results = [self.analyze_record(r) for r in records]
        # Sort score descending
        results.sort(key=lambda x: x.risk_score, reverse=True)
        return results

    def analyze_file(self, filepath_or_buffer) -> Tuple[List[AnalyzedCustomerRecord], List[Dict[str, Any]]]:
        raw_records, errors = DataLoader.load_from_csv(filepath_or_buffer)
        analyzed = self.analyze_batch(raw_records)
        return analyzed, errors