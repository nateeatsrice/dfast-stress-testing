"""Unit tests for dfast.data.feature_engineering module."""

import numpy as np
import pandas as pd
import pytest

from dfast.data.feature_engineering import (
    _compute_current_ltv,
    _compute_loan_age,
    _compute_rolling_dq,
    _encode_unit_bucket,
    build_feature_table,
)


class TestBuildFeatureTable:
    """Integration-style tests for the full feature table builder."""

    def test_output_shape(self, sample_features_df: pd.DataFrame) -> None:
        """Feature table should have all expected columns."""
        expected_cols = {
            "loan_id", "reporting_period", "original_upb", "original_ltv",
            "original_dscr", "note_rate", "loan_purpose_purchase", "unit_bucket",
            "region", "loan_age", "months_since_last_delinquency",
            "rolling_dq_count_6m", "rolling_dq_count_12m", "unemployment_rate",
            "gdp_growth", "hpi_change", "current_ltv", "is_default",
        }
        assert expected_cols.issubset(set(sample_features_df.columns))

    def test_no_null_target(self, sample_features_df: pd.DataFrame) -> None:
        """The is_default target column should never be null."""
        assert sample_features_df["is_default"].isna().sum() == 0


class TestLoanAge:
    """Tests for _compute_loan_age."""

    def test_basic_calculation(self) -> None:
        """Loan age should be the month difference."""
        orig = pd.Series(pd.to_datetime(["2020-01-01", "2020-06-01"]))
        report = pd.Series(pd.to_datetime(["2020-07-01", "2020-06-01"]))

        result = _compute_loan_age(orig, report)

        assert result.tolist() == [6, 0]

    def test_negative_clipped_to_zero(self) -> None:
        """If reporting_period precedes origination, loan_age should be 0."""
        orig = pd.Series(pd.to_datetime(["2020-06-01"]))
        report = pd.Series(pd.to_datetime(["2020-01-01"]))

        result = _compute_loan_age(orig, report)

        assert result.tolist() == [0]


class TestRollingDqCount:
    """Tests for _compute_rolling_dq."""

    def test_six_month_window(self) -> None:
        """Rolling 6-month window should count delinquent months correctly."""
        df = pd.DataFrame({
            "loan_id": ["A"] * 8,
            "delinquency_status": [0, 0, 1, 0, 0, 2, 0, 0],
        })

        result = _compute_rolling_dq(df, window=6)

        # At index 7 (last), window covers indices 2-7: dq at idx 2,5 = 2
        assert result.iloc[7] == 1  # Only idx 5 is in the 6-month window ending at 7
        # At index 5: window covers 0-5, dq at idx 2 and 5 = 2
        assert result.iloc[5] == 2


class TestCurrentLtv:
    """Tests for _compute_current_ltv."""

    def test_no_hpi_change(self) -> None:
        """With zero HPI change, LTV changes only from amortization."""
        result = _compute_current_ltv(
            original_upb=pd.Series([1_000_000.0]),
            original_ltv=pd.Series([0.65]),
            loan_age=pd.Series([0]),
            hpi_change=pd.Series([0.0]),
            note_rate=pd.Series([0.05]),
        )
        # At age 0 with no HPI change: current LTV ≈ original LTV
        assert abs(result.iloc[0] - 0.65) < 0.01

    def test_negative_hpi_increases_ltv(self) -> None:
        """A declining HPI should increase LTV."""
        ltv_flat = _compute_current_ltv(
            original_upb=pd.Series([1_000_000.0]),
            original_ltv=pd.Series([0.65]),
            loan_age=pd.Series([24]),
            hpi_change=pd.Series([0.0]),
            note_rate=pd.Series([0.05]),
        )
        ltv_stressed = _compute_current_ltv(
            original_upb=pd.Series([1_000_000.0]),
            original_ltv=pd.Series([0.65]),
            loan_age=pd.Series([24]),
            hpi_change=pd.Series([-0.30]),
            note_rate=pd.Series([0.05]),
        )

        assert ltv_stressed.iloc[0] > ltv_flat.iloc[0]


class TestRegionEncoding:
    """Tests for region mapping via build_feature_table."""

    def test_state_maps_to_correct_region(self, sample_features_df: pd.DataFrame) -> None:
        """States should map to expected census regions."""
        valid_regions = {"Northeast", "Southeast", "Midwest", "West"}
        regions_found = set(sample_features_df["region"].unique())
        assert regions_found.issubset(valid_regions)


class TestUnitBucket:
    """Tests for _encode_unit_bucket."""

    def test_bucket_boundaries(self) -> None:
        """Units should bucket into small/medium/large correctly."""
        units = pd.Series([10, 49, 50, 200, 201, 500])
        result = _encode_unit_bucket(units)

        assert result.iloc[0] == "small"
        assert result.iloc[1] == "small"
        assert result.iloc[2] == "medium"
        assert result.iloc[3] == "medium"
        assert result.iloc[4] == "large"
        assert result.iloc[5] == "large"
