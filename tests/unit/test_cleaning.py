"""Unit tests for dfast.data.cleaning module."""

import numpy as np
import pandas as pd
import pytest

from dfast.data.cleaning import (
    clean_acquisition,
    clean_performance,
    create_default_flag,
    remove_duplicates,
)


class TestCleanAcquisition:
    """Tests for the clean_acquisition function."""

    def test_handles_null_dscr(self, sample_acquisition_df: pd.DataFrame) -> None:
        """Missing DSCR values should be filled with median."""
        df = sample_acquisition_df.copy()
        df.loc[0, "original_dscr"] = np.nan
        df.loc[1, "original_dscr"] = np.nan

        result = clean_acquisition(df)

        assert result["original_dscr"].isna().sum() == 0

    def test_drops_rows_with_null_critical_fields(self, sample_acquisition_df: pd.DataFrame) -> None:
        """Rows missing loan_id or original_upb should be dropped."""
        df = sample_acquisition_df.copy()
        df.loc[0, "original_upb"] = np.nan

        result = clean_acquisition(df)

        assert len(result) == len(df) - 1

    def test_normalizes_state_to_uppercase(self, sample_acquisition_df: pd.DataFrame) -> None:
        """State codes should be uppercase two-letter."""
        df = sample_acquisition_df.copy()
        df.loc[0, "state"] = "ca"

        result = clean_acquisition(df)

        assert result.loc[0, "state"] == "CA"

    def test_normalizes_property_type(self, sample_acquisition_df: pd.DataFrame) -> None:
        """Property type should be title case."""
        df = sample_acquisition_df.copy()
        df.loc[0, "property_type"] = "apartment"

        result = clean_acquisition(df)

        assert result.loc[0, "property_type"] == "Apartment"


class TestCleanPerformance:
    """Tests for the clean_performance function."""

    def test_normalizes_dq_status_to_int(self, sample_performance_df: pd.DataFrame) -> None:
        """Delinquency status should be integers 0-3."""
        df = sample_performance_df.copy()
        result = clean_performance(df)

        assert result["delinquency_status"].dtype in [np.int64, np.int32, int]
        assert set(result["delinquency_status"].unique()).issubset({0, 1, 2, 3})

    def test_handles_string_dq_codes(self) -> None:
        """String delinquency codes like 'C' and 'F' should map correctly."""
        df = pd.DataFrame({
            "loan_id": ["LN1", "LN2", "LN3"],
            "reporting_period": pd.date_range("2020-01-01", periods=3, freq="MS"),
            "current_upb": [1e6, 1e6, 1e6],
            "delinquency_status": ["C", "1", "F"],
            "zero_balance_code": ["", "", ""],
        })

        result = clean_performance(df)

        assert result.loc[0, "delinquency_status"] == 0  # C = current
        assert result.loc[1, "delinquency_status"] == 1  # 1 = 30-day
        assert result.loc[2, "delinquency_status"] == 3  # F = foreclosure (90+)


class TestCreateDefaultFlag:
    """Tests for the create_default_flag function."""

    def test_d90_threshold(self) -> None:
        """Only delinquency_status >= 3 should be flagged as default."""
        df = pd.DataFrame({
            "loan_id": ["A", "B", "C", "D"],
            "delinquency_status": [0, 1, 2, 3],
        })

        result = create_default_flag(df)

        assert result["is_default"].tolist() == [0, 0, 0, 1]

    def test_all_current_no_defaults(self) -> None:
        """A portfolio with all current loans should have zero defaults."""
        df = pd.DataFrame({
            "loan_id": ["A", "B"],
            "delinquency_status": [0, 0],
        })

        result = create_default_flag(df)

        assert result["is_default"].sum() == 0


class TestRemoveDuplicates:
    """Tests for the remove_duplicates function."""

    def test_removes_exact_duplicates(self) -> None:
        """Duplicate (loan_id, reporting_period) pairs should be deduped."""
        df = pd.DataFrame({
            "loan_id": ["A", "A", "B"],
            "reporting_period": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-01"]),
            "value": [1, 2, 3],
        })

        result = remove_duplicates(df)

        assert len(result) == 2
        # Should keep the last occurrence
        assert result.loc[result["loan_id"] == "A", "value"].values[0] == 2

    def test_no_duplicates_unchanged(self, sample_performance_df: pd.DataFrame) -> None:
        """DataFrame without duplicates should pass through unchanged."""
        result = remove_duplicates(sample_performance_df)

        assert len(result) == len(sample_performance_df)
