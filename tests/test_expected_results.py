"""
Table-driven regression harness for the deterministic scoring contract.

Development Guide §10, minimum automated assertions:
  - a known fixture row produces the exact expected score, band, and top contributor
  - risk score is always an integer 0-100 and the band is one of four values
  - the loader reports invalid rows without exposing another customer's data
  - mock mode executes the complete pipeline without an API key

Expectations in data/expected_results.csv are MOCK-MODE values. Mock mode is
fully deterministic, so scores are asserted exactly. Live Gemini output varies
between runs and is deliberately not asserted here.
"""
from pathlib import Path

import pandas as pd
import pytest

from src.signal_analyzer import SignalAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = PROJECT_ROOT / "data" / "sample_customers.csv"
EXPECTED_CSV = PROJECT_ROOT / "data" / "expected_results.csv"

VALID_BANDS = {"Low", "Medium", "High", "Critical"}


@pytest.fixture(scope="module")
def analyzed_by_id():
    """Run the full pipeline once in mock mode; index results by customer_id."""
    analyzer = SignalAnalyzer(mock_mode=True)
    results, errors = analyzer.analyze_file(str(SAMPLE_CSV))
    assert errors == [], f"sample fixture should load cleanly, got: {errors}"
    return {r.customer_id: r for r in results}


def _expected_rows():
    df = pd.read_csv(EXPECTED_CSV)
    return [row for _, row in df.iterrows()]


@pytest.mark.parametrize(
    "expected",
    _expected_rows(),
    ids=lambda row: str(row["customer_id"]),
)
def test_fixture_row_scores_as_documented(expected, analyzed_by_id):
    """Each fixture row must produce its documented score, band, and top contributor."""
    cid = expected["customer_id"]
    assert cid in analyzed_by_id, f"{cid} missing from pipeline output"
    record = analyzed_by_id[cid]

    assert record.risk_band == expected["expected_band"], (
        f"{cid}: expected band {expected['expected_band']}, got {record.risk_band} "
        f"(score {record.risk_score})"
    )

    assert expected["min_score"] <= record.risk_score <= expected["max_score"], (
        f"{cid}: score {record.risk_score} outside expected range "
        f"[{expected['min_score']}, {expected['max_score']}]"
    )

    top_contributor = record.contributors[0].signal_name if record.contributors else "none"
    assert top_contributor == expected["primary_contributor"], (
        f"{cid}: expected top contributor '{expected['primary_contributor']}', "
        f"got '{top_contributor}'"
    )


def test_every_score_is_a_valid_integer_and_band(analyzed_by_id):
    """Risk score is always an int in 0-100; band is always one of four values."""
    for cid, record in analyzed_by_id.items():
        assert isinstance(record.risk_score, int), f"{cid}: score is not an int"
        assert 0 <= record.risk_score <= 100, f"{cid}: score {record.risk_score} out of range"
        assert record.risk_band in VALID_BANDS, f"{cid}: invalid band {record.risk_band}"


def test_contributors_are_sorted_by_points_descending(analyzed_by_id):
    """The rationale reads the top contributors, so ordering must be guaranteed."""
    for cid, record in analyzed_by_id.items():
        points = [c.points for c in record.contributors]
        assert points == sorted(points, reverse=True), f"{cid}: contributors not sorted"


def test_band_matches_score_thresholds(analyzed_by_id):
    """Band must agree with the score it was derived from (scoring_v1 thresholds)."""
    for cid, record in analyzed_by_id.items():
        score = record.risk_score
        if score >= 75:
            expected_band = "Critical"
        elif score >= 50:
            expected_band = "High"
        elif score >= 25:
            expected_band = "Medium"
        else:
            expected_band = "Low"
        assert record.risk_band == expected_band, (
            f"{cid}: score {score} should be {expected_band}, got {record.risk_band}"
        )


def test_mock_mode_completes_without_api_key(monkeypatch):
    """Mock mode must run the entire pipeline with no credentials present."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    analyzer = SignalAnalyzer(mock_mode=True)
    results, errors = analyzer.analyze_file(str(SAMPLE_CSV))

    assert errors == []
    assert len(results) == 14
    assert all(r.analysis_source == "mock" for r in results)
    assert all(r.suggested_action for r in results)
    assert all(r.rationale for r in results)


def test_results_are_deterministic():
    """The same fixture must produce identical scores across runs."""
    first = SignalAnalyzer(mock_mode=True).analyze_file(str(SAMPLE_CSV))[0]
    second = SignalAnalyzer(mock_mode=True).analyze_file(str(SAMPLE_CSV))[0]

    assert [(r.customer_id, r.risk_score, r.risk_band) for r in first] == \
           [(r.customer_id, r.risk_score, r.risk_band) for r in second]
