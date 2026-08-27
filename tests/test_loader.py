import io
import pandas as pd
import pytest
from src.data_loader import DataLoader
from src.models import RawCustomerRecord


def test_load_valid_sample_customers():
    records, errors = DataLoader.load_from_csv("data/sample_customers.csv")
    assert len(records) >= 12
    assert len(errors) == 0
    assert isinstance(records[0], RawCustomerRecord)
    assert records[0].customer_id == "CUST-001"


def test_loader_row_level_validation_error():
    # Test CSV with 1 valid row and 1 invalid row (empty customer_id and blank text)
    bad_csv = """customer_id,interaction_text,issue_count_30d,payment_status
CUST-VALID,"This is a valid support message with more than 10 characters.",2,current
,"",0,current
"""
    records, errors = DataLoader.load_from_csv(io.StringIO(bad_csv))
    assert len(records) == 1
    assert len(errors) == 1
    assert errors[0]["row_number"] == 3