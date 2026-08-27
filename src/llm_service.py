"""
LLM Service module for bounded semantic signal extraction with retry, SHA-256 caching, and mock mode.
"""
import os
import re
import json
import time
import hashlib
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
from src.models import SemanticSignal, SemanticResult

load_dotenv()


class LLMService:
    """Service to interact with Google Gemini models for bounded semantic signal extraction."""

    # Fallback used only if neither an explicit model_name nor GEMINI_MODEL is set.
    DEFAULT_MODEL = "gemini-3.6-flash"

    # In-memory session cache: {"<mode>:<sha256>": SemanticResult}
    _session_cache: Dict[str, SemanticResult] = {}

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        mock_mode: bool = False
    ):
        self.mock_mode = mock_mode
        self.model_name = model_name or os.getenv("GEMINI_MODEL") or self.DEFAULT_MODEL
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.mock_mode:
            if not self.api_key:
                # If no API key is provided, automatically fallback to mock mode safely
                print("[WARNING] GEMINI_API_KEY not found. Running in MOCK MODE.")
                self.mock_mode = True
            else:
                self.client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _compute_hash(text: str) -> str:
        """Computes SHA-256 hash of normalized text for safe caching."""
        normalized = " ".join(text.strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _mock_analyze(self, transcript: str) -> SemanticSignal:
        """
        Deterministic offline mock analyzer for testing and resilient fallback.
        """
        text_lower = transcript.lower()

        def matches(patterns) -> bool:
            # Word-boundary matching. Plain substring checks produce false
            # positives: "down" matches "download", "fee" matches "coffee".
            return any(re.search(p, text_lower) for p in patterns)

        # Heuristic keywords for mock simulation. Prefix patterns (no trailing
        # \b) intentionally cover inflections: terminat -> terminate/termination.
        is_cancel = matches([
            r"\bcancel", r"\bterminat", r"\bcompetitor", r"\bswitch\b",
            r"\bleave\b", r"\bleaving\b", r"\bother vendors?\b",
            r"\banother vendor\b", r"\bnot renew", r"\bnon-renew",
        ])
        is_billing = matches([
            r"\bbill", r"\binvoice", r"\bcharge", r"\boverage",
            r"\bfees?\b", r"\brefund",
        ])
        is_outage = matches([
            r"\boutage", r"\bbugs?\b", r"\brate-limit", r"\berrors?\b",
            r"\bdown\b", r"\bdowntime\b", r"\bfail", r"\bcrash",
        ])
        is_praise = matches([
            r"\bhappy\b", r"\bgreat\b", r"\bthanks?\b", r"\bthank you\b",
            r"\blove\b", r"\bsmooth", r"\bappreciate",
        ])

        if is_praise and not is_cancel and not is_outage:
            return SemanticSignal(
                sentiment="positive",
                issue_type="other",
                urgency="low",
                churn_intent=False,
                confidence=0.92,
                evidence_spans=["team is very happy with the update"],
                summary="Positive feedback regarding platform performance and support."
            )

        if is_cancel:
            return SemanticSignal(
                sentiment="very_negative",
                issue_type="cancellation" if not is_billing else "billing",
                urgency="critical",
                churn_intent=True,
                confidence=0.95,
                evidence_spans=[transcript[:80]],
                summary="Customer expressing explicit cancellation or contract termination intent."
            )

        if is_billing:
            return SemanticSignal(
                sentiment="negative",
                issue_type="billing",
                urgency="high",
                churn_intent=False,
                confidence=0.88,
                evidence_spans=[transcript[:80]],
                summary="Disputed billing charges or invoice inquiry requiring resolution."
            )

        if is_outage:
            return SemanticSignal(
                sentiment="negative",
                issue_type="product",
                urgency="high",
                churn_intent=False,
                confidence=0.85,
                evidence_spans=[transcript[:80]],
                summary="Technical error or system outage impacting customer workflows."
            )

        return SemanticSignal(
            sentiment="neutral",
            issue_type="support",
            urgency="medium",
            churn_intent=False,
            confidence=0.75,
            evidence_spans=[],
            summary="General support inquiry or workflow feedback."
        )

    def analyze_transcript(self, transcript: str) -> SemanticResult:
        """
        Extracts schema-validated semantic signals from a transcript.

        Returns a SemanticResult carrying one of three honest outcomes:
          - source="live"   : a real, schema-validated model response
          - source="mock"   : an explicitly-requested offline simulation
          - source="failed" : the call failed; signal is None and error is set

        A failure is NEVER replaced with fabricated analysis (see DESIGN.md §4).
        """
        if not transcript or len(transcript.strip()) < 5:
            return SemanticResult(
                signal=None,
                source="failed",
                error="Transcript too short to analyze",
            )

        # Cache key includes the mode, so a mock result can never be served
        # to a live-mode request (or vice versa) for the same transcript.
        mode = "mock" if self.mock_mode else "live"
        cache_key = f"{mode}:{self._compute_hash(transcript)}"
        if cache_key in self._session_cache:
            return self._session_cache[cache_key]

        # Explicit mock mode — labelled as such, never passed off as live.
        if self.mock_mode:
            result = SemanticResult(signal=self._mock_analyze(transcript), source="mock")
            self._session_cache[cache_key] = result
            return result

        # Live Gemini API Call with 1 Retry
        prompt = f"""
You are an expert Customer Operations Semantic Intelligence AI.
Analyze the following customer interaction and extract structured fields strictly according to the schema.

Customer Interaction:
\"\"\"{transcript}\"\"\"

Return a valid JSON object matching:
- sentiment: 'very_negative' | 'negative' | 'neutral' | 'positive'
- issue_type: 'billing' | 'support' | 'product' | 'usage' | 'cancellation' | 'other'
- urgency: 'low' | 'medium' | 'high' | 'critical'
- churn_intent: boolean (true if explicit mention of canceling/leaving/competitor)
- confidence: float between 0.0 and 1.0
- evidence_spans: array of up to 2 short direct quotes
- summary: concise summary <= 160 characters
"""

        last_error: Optional[Exception] = None

        # One retry only, with a short backoff between attempts.
        for attempt in range(2):
            try:
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )

                raw_text = response.text.strip()
                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:]
                if raw_text.startswith("```"):
                    raw_text = raw_text[3:]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]

                parsed_dict = json.loads(raw_text.strip())
                signal = SemanticSignal.model_validate(parsed_dict)

                result = SemanticResult(signal=signal, source="live")
                self._session_cache[cache_key] = result
                return result

            except Exception as e:
                last_error = e
                if attempt == 0:
                    time.sleep(1.0)

        # Both attempts failed. Report it honestly — do NOT fabricate a result.
        # Failures are deliberately not cached, so a transient error can be
        # retried on the next run.
        error_detail = f"{type(last_error).__name__}: {last_error}"
        print(f"[LLM Service] semantic analysis unavailable — {error_detail}")
        return SemanticResult(signal=None, source="failed", error=error_detail)