import pandas as pd
from src.signal_analyzer import SignalAnalyzer


def test_end_to_end_mock_pipeline():
    analyzer = SignalAnalyzer(mock_mode=True)
    results, errors = analyzer.analyze_file("data/sample_customers.csv")

    assert len(results) >= 12
    assert len(errors) == 0

    # Ensure results are sorted by risk_score descending
    scores = [r.risk_score for r in results]
    assert scores == sorted(scores, reverse=True)

    # Check that highest risk customer has rationale and suggested action
    top_customer = results[0]
    assert top_customer.risk_band in ["Critical", "High"]
    assert len(top_customer.rationale) > 10
    assert len(top_customer.suggested_action) > 10

    # Mock mode must be labelled as such — never passed off as a live analysis
    assert all(r.analysis_source == "mock" for r in results)