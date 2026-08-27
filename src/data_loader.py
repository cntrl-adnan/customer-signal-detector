"""
Data loader module with row-level Pydantic validation and error collection.
"""
import pandas as pd
from typing import List, Tuple, Dict, Any
from src.models import RawCustomerRecord


class DataLoader:
    """Loads and validates customer CSV data according to the data contract."""

    REQUIRED_COLUMNS = ["customer_id", "interaction_text"]

    @classmethod
    def load_from_dataframe(cls, df: pd.DataFrame) -> Tuple[List[RawCustomerRecord], List[Dict[str, Any]]]:
        """
        Parses a pandas DataFrame into valid RawCustomerRecord models and collects row errors.
        """
        valid_records: List[RawCustomerRecord] = []
        validation_errors: List[Dict[str, Any]] = []

        # Standardize column names (lowercase, strip whitespace)
        df.columns = [c.strip().lower() for c in df.columns]

        # Check critical required columns
        for req in cls.REQUIRED_COLUMNS:
            if req not in df.columns:
                raise ValueError(f"Missing required CSV column: '{req}'")

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            row_number = idx + 2  # 1-indexed including header

            # Clean NaN values
            cleaned: Dict[str, Any] = {}
            for k, v in row_dict.items():
                if pd.isna(v):
                    cleaned[k] = None
                else:
                    cleaned[k] = v

            # Coerce types safely
            try:
                # Numeric defaults
                cleaned["issue_count_30d"] = int(cleaned.get("issue_count_30d") or 0)
                cleaned["failed_payments_30d"] = int(cleaned.get("failed_payments_30d") or 0)
                cleaned["usage_change_pct"] = float(cleaned.get("usage_change_pct") or 0.0)
                
                # CSAT can be None/blank
                if cleaned.get("csat_score") is not None and cleaned.get("csat_score") != "":
                    cleaned["csat_score"] = int(cleaned["csat_score"])
                else:
                    cleaned["csat_score"] = None

                if cleaned.get("last_contact_days") is not None and cleaned.get("last_contact_days") != "":
                    cleaned["last_contact_days"] = int(cleaned["last_contact_days"])
                else:
                    cleaned["last_contact_days"] = 0

                # Payment status enum default
                pay_status = str(cleaned.get("payment_status") or "current").strip().lower()
                if pay_status not in ["current", "overdue", "failed", "unknown"]:
                    pay_status = "unknown"
                cleaned["payment_status"] = pay_status

                # Validate with Pydantic
                record = RawCustomerRecord(**cleaned)
                valid_records.append(record)

            except Exception as e:
                validation_errors.append({
                    "row_number": row_number,
                    "customer_id": str(cleaned.get("customer_id", f"Row-{row_number}")),
                    "error_message": str(e)
                })

        return valid_records, validation_errors

    @classmethod
    def load_from_csv(cls, filepath_or_buffer) -> Tuple[List[RawCustomerRecord], List[Dict[str, Any]]]:
        """Loads and validates directly from a file path or file-like buffer."""
        df = pd.read_csv(filepath_or_buffer)
        return cls.load_from_dataframe(df)