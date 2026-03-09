"""Shared pytest fixtures for DFAST test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_acquisition_df() -> pd.DataFrame:
    """Small acquisition DataFrame with 10 loans for unit tests."""
    rng = np.random.default_rng(0)
    n = 10
    return pd.DataFrame({
        "loan_id": [f"LN{i:06d}" for i in range(n)],
        "origination_date": pd.date_range("2019-01-01", periods=n, freq="MS"),
        "original_upb": rng.uniform(1_000_000, 10_000_000, n).round(2),
        "original_ltv": rng.uniform(0.50, 0.80, n).round(4),
        "original_dscr": rng.uniform(1.0, 1.8, n).round(4),
        "note_rate": rng.uniform(0.03, 0.06, n).round(5),
        "property_type": ["Apartment"] * 7 + ["Coop"] * 2 + ["Other"],
        "state": ["CA", "TX", "NY", "FL", "IL", "GA", "OH", "PA", "NJ", "VA"],
        "loan_purpose": ["Purchase"] * 6 + ["Refinance"] * 4,
        "number_of_units": [30, 60, 150, 250, 10, 80, 45, 200, 100, 20],
        "maturity_date": pd.date_range("2029-01-01", periods=n, freq="MS"),
    })


@pytest.fixture()
def sample_performance_df() -> pd.DataFrame:
    """Performance records for 3 loans: 2 current, 1 delinquent."""
    records = []
    for loan_idx in range(3):
        loan_id = f"LN{loan_idx:06d}"
        for month in range(12):
            if loan_idx == 2 and month >= 8:
                dq = 3  # Loan 2 goes 90+ DPD from month 8
            elif loan_idx == 1 and month >= 10:
                dq = 1  # Loan 1 goes 30-day DPD late
            else:
                dq = 0
            records.append({
                "loan_id": loan_id,
                "reporting_period": pd.Timestamp("2019-02-01") + pd.DateOffset(months=month),
                "current_upb": 5_000_000.0 * (1 - month / 360),
                "delinquency_status": dq,
                "zero_balance_code": "",
            })
    return pd.DataFrame(records)


@pytest.fixture()
def sample_macro_df() -> pd.DataFrame:
    """Monthly macro data for 2019."""
    dates = pd.date_range("2019-01-01", periods=14, freq="MS")
    return pd.DataFrame({
        "date": dates,
        "gdp_growth": [0.022] * 14,
        "unemployment_rate": [0.04] * 14,
        "hpi_change": np.linspace(0.0, 0.03, 14).round(5),
    })


@pytest.fixture()
def sample_features_df(sample_acquisition_df, sample_performance_df, sample_macro_df):
    """Pre-built feature table from sample data."""
    from dfast.data.cleaning import (
        clean_acquisition,
        clean_performance,
        create_default_flag,
    )
    from dfast.data.feature_engineering import build_feature_table

    acq = clean_acquisition(sample_acquisition_df)
    perf = clean_performance(sample_performance_df)
    perf = create_default_flag(perf)
    return build_feature_table(acq, perf, sample_macro_df)
