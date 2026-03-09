"""Data cleaning: null handling, type normalization, dedup, and default flag creation.

Each function is designed to be pure — it returns a new DataFrame rather than
mutating the input. Schema validation via Pandera is applied after cleaning.
"""

import logging

import numpy as np
import pandas as pd

from dfast.data.schemas import AcquisitionSchema, PerformanceSchema

logger = logging.getLogger(__name__)

# ── Delinquency mapping ──────────────────────────────────────
# Raw Fannie Mae data may encode delinquency as string codes.
# We normalize to integer buckets: 0=current, 1=30-day, 2=60-day, 3=90+.
_DQ_MAP: dict[str, int] = {
    "0": 0,
    "C": 0,
    "current": 0,
    "1": 1,
    "30": 1,
    "2": 2,
    "60": 2,
    "3": 3,
    "90": 3,
    "90+": 3,
    "F": 3,
    "R": 3,
}


def clean_acquisition(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw acquisition data: handle nulls, normalize dtypes, validate.

    Args:
        df: Raw acquisition DataFrame.

    Returns:
        Cleaned and validated acquisition DataFrame.
    """
    out = df.copy()

    # Ensure loan_id is a string
    out["loan_id"] = out["loan_id"].astype(str)

    # Parse dates if not already datetime
    for col in ["origination_date", "maturity_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    # Fill missing DSCR with median (common in MF data)
    if "original_dscr" in out.columns and out["original_dscr"].isna().any():
        median_dscr = out["original_dscr"].median()
        out["original_dscr"] = out["original_dscr"].fillna(median_dscr)
        logger.info("Filled %d missing DSCR values with median %.3f", df["original_dscr"].isna().sum(), median_dscr)

    # Normalize property_type to title case
    if "property_type" in out.columns:
        out["property_type"] = out["property_type"].str.strip().str.title()

    # Normalize state to uppercase two-letter
    if "state" in out.columns:
        out["state"] = out["state"].str.strip().str.upper()

    # Ensure numeric types
    for col in ["original_upb", "original_ltv", "note_rate"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["number_of_units"] = pd.to_numeric(out["number_of_units"], errors="coerce").fillna(0).astype(int)

    # Drop rows with critical nulls
    critical = ["loan_id", "origination_date", "original_upb", "original_ltv", "note_rate"]
    n_before = len(out)
    out = out.dropna(subset=[c for c in critical if c in out.columns])
    n_dropped = n_before - len(out)
    if n_dropped > 0:
        logger.info("Dropped %d rows with null critical fields", n_dropped)

    # Validate
    out = AcquisitionSchema.validate(out)
    logger.info("Acquisition cleaning complete: %d rows", len(out))
    return out


def clean_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw performance data: parse dates, normalize delinquency, validate.

    Args:
        df: Raw performance DataFrame.

    Returns:
        Cleaned and validated performance DataFrame.
    """
    out = df.copy()

    out["loan_id"] = out["loan_id"].astype(str)
    out["reporting_period"] = pd.to_datetime(out["reporting_period"], errors="coerce")

    # Normalize delinquency status to integer buckets
    out["delinquency_status"] = (
        out["delinquency_status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(_DQ_MAP)
        .fillna(0)
        .astype(int)
    )

    out["current_upb"] = pd.to_numeric(out["current_upb"], errors="coerce")

    # Fill missing zero_balance_code
    if "zero_balance_code" in out.columns:
        out["zero_balance_code"] = out["zero_balance_code"].fillna("")

    # Drop rows with null keys
    out = out.dropna(subset=["loan_id", "reporting_period"])

    out = PerformanceSchema.validate(out)
    logger.info("Performance cleaning complete: %d rows", len(out))
    return out


def create_default_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary ``is_default`` column using the D90 industry standard.

    A loan is considered in default if ``delinquency_status >= 3`` (90+ days
    past due).

    Args:
        df: Performance DataFrame with ``delinquency_status`` column.

    Returns:
        DataFrame with an additional ``is_default`` column.
    """
    out = df.copy()
    out["is_default"] = np.where(out["delinquency_status"] >= 3, 1, 0)

    default_rate = out["is_default"].mean()
    logger.info(
        "Default flag created: %.2f%% observations in default (%d / %d)",
        default_rate * 100,
        out["is_default"].sum(),
        len(out),
    )
    return out


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows on (loan_id, reporting_period).

    Keeps the last occurrence when duplicates are found.

    Args:
        df: DataFrame with ``loan_id`` and ``reporting_period`` columns.

    Returns:
        De-duplicated DataFrame.
    """
    n_before = len(df)
    out = df.drop_duplicates(subset=["loan_id", "reporting_period"], keep="last")
    n_removed = n_before - len(out)

    if n_removed > 0:
        logger.info("Removed %d duplicate (loan_id, reporting_period) rows", n_removed)

    return out.reset_index(drop=True)
