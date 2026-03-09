"""Feature engineering: build the modeling-ready feature table from cleaned data.

Joins acquisition and performance data, computes derived features (loan age,
rolling delinquency counts, macro-linked LTV), and encodes categoricals.
"""

import logging

import numpy as np
import pandas as pd

from dfast.data.schemas import FeatureSchema

logger = logging.getLogger(__name__)

# ── Region mapping ────────────────────────────────────────────
_REGION_MAP: dict[str, str] = {
    # Northeast
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "NJ": "Northeast", "NY": "Northeast", "PA": "Northeast", "RI": "Northeast",
    "VT": "Northeast", "DE": "Northeast", "MD": "Northeast", "DC": "Northeast",
    # Southeast
    "AL": "Southeast", "AR": "Southeast", "FL": "Southeast", "GA": "Southeast",
    "KY": "Southeast", "LA": "Southeast", "MS": "Southeast", "NC": "Southeast",
    "SC": "Southeast", "TN": "Southeast", "VA": "Southeast", "WV": "Southeast",
    # Midwest
    "IL": "Midwest", "IN": "Midwest", "IA": "Midwest", "KS": "Midwest",
    "MI": "Midwest", "MN": "Midwest", "MO": "Midwest", "NE": "Midwest",
    "ND": "Midwest", "OH": "Midwest", "SD": "Midwest", "WI": "Midwest",
    # West
    "AK": "West", "AZ": "West", "CA": "West", "CO": "West",
    "HI": "West", "ID": "West", "MT": "West", "NV": "West",
    "NM": "West", "OK": "West", "OR": "West", "TX": "West",
    "UT": "West", "WA": "West", "WY": "West",
}


def _compute_loan_age(origination_date: pd.Series, reporting_period: pd.Series) -> pd.Series:
    """Calculate loan age in months between origination and reporting period.

    Args:
        origination_date: Series of origination dates.
        reporting_period: Series of reporting period dates.

    Returns:
        Series of integer loan ages (months).
    """
    delta = (reporting_period.dt.year - origination_date.dt.year) * 12 + (
        reporting_period.dt.month - origination_date.dt.month
    )
    return delta.clip(lower=0).astype(int)


def _compute_rolling_dq(
    perf_df: pd.DataFrame, window: int
) -> pd.Series:
    """Count months in delinquency within a trailing window per loan.

    Args:
        perf_df: Performance DataFrame sorted by (loan_id, reporting_period).
        window: Number of trailing months.

    Returns:
        Series with rolling delinquency counts.
    """
    dq_flag = (perf_df["delinquency_status"] > 0).astype(int)
    rolling = (
        dq_flag.groupby(perf_df["loan_id"])
        .rolling(window=window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .astype(int)
    )
    return rolling


def _compute_months_since_last_dq(perf_df: pd.DataFrame) -> pd.Series:
    """Months since the most recent delinquency event for each observation.

    Args:
        perf_df: Performance DataFrame sorted by (loan_id, reporting_period).

    Returns:
        Series with months since last delinquency; -1 if never delinquent.
    """
    result = pd.Series(index=perf_df.index, dtype="Int64")

    for loan_id, group in perf_df.groupby("loan_id"):
        months_since = pd.Series(index=group.index, dtype="Int64")
        counter = -1  # -1 means never delinquent

        for idx, row in group.iterrows():
            if row["delinquency_status"] > 0:
                counter = 0
            elif counter >= 0:
                counter += 1
            months_since.loc[idx] = counter

        result.loc[group.index] = months_since

    return result


def _encode_unit_bucket(units: pd.Series) -> pd.Series:
    """Bucket unit count: small (<50), medium (50–200), large (200+).

    Args:
        units: Series of integer unit counts.

    Returns:
        Categorical series with bucket labels.
    """
    return pd.cut(
        units,
        bins=[0, 49, 200, float("inf")],
        labels=["small", "medium", "large"],
        right=True,
    ).astype(str)


def _compute_current_ltv(
    original_upb: pd.Series,
    original_ltv: pd.Series,
    loan_age: pd.Series,
    hpi_change: pd.Series,
    note_rate: pd.Series,
) -> pd.Series:
    """Estimate current LTV under a given HPI change.

    Uses a simplified amortization assumption: straight-line amortization over
    a 30-year term. Property value is adjusted by cumulative HPI change.

    Args:
        original_upb: Original unpaid principal balance.
        original_ltv: Original loan-to-value ratio.
        loan_age: Age of loan in months.
        hpi_change: Cumulative HPI change since origination (e.g., -0.15 for 15% decline).
        note_rate: Annual note rate (used for amortization estimate).

    Returns:
        Estimated current LTV series.
    """
    # Estimate original property value
    original_property_value = original_upb / original_ltv.clip(lower=0.01)

    # Simplified remaining balance (straight-line over 360 months)
    term_months = 360
    amort_factor = (1 - (loan_age / term_months)).clip(lower=0.0)
    current_balance = original_upb * amort_factor

    # Stressed property value
    current_property_value = original_property_value * (1 + hpi_change)
    current_property_value = current_property_value.clip(lower=1.0)

    current_ltv = current_balance / current_property_value
    return current_ltv.clip(lower=0.0)


def build_feature_table(
    acquisition_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the modeling-ready feature table from cleaned inputs.

    Joins acquisition and performance data, merges macro variables by month,
    and computes all derived features.

    Args:
        acquisition_df: Cleaned acquisition DataFrame.
        performance_df: Cleaned performance DataFrame with ``is_default`` column.
        macro_df: Macro economic time series with columns ``date``,
            ``gdp_growth``, ``unemployment_rate``, ``hpi_change``.

    Returns:
        Feature DataFrame validated against ``FeatureSchema``.
    """
    logger.info(
        "Building feature table: %d acquisitions, %d performance records",
        len(acquisition_df),
        len(performance_df),
    )

    # Sort performance for rolling computations
    perf = performance_df.sort_values(["loan_id", "reporting_period"]).reset_index(drop=True)

    # ── Join acquisition onto performance ────────────────────
    merged = perf.merge(
        acquisition_df[
            [
                "loan_id",
                "origination_date",
                "original_upb",
                "original_ltv",
                "original_dscr",
                "note_rate",
                "property_type",
                "state",
                "loan_purpose",
                "number_of_units",
            ]
        ],
        on="loan_id",
        how="inner",
    )

    # ── Merge macro data by month ────────────────────────────
    macro = macro_df.copy()
    macro["year_month"] = pd.to_datetime(macro["date"]).dt.to_period("M")
    merged["year_month"] = merged["reporting_period"].dt.to_period("M")

    merged = merged.merge(
        macro[["year_month", "gdp_growth", "unemployment_rate", "hpi_change"]],
        on="year_month",
        how="left",
    )

    # Fill any missing macro values with baseline assumptions
    merged["gdp_growth"] = merged["gdp_growth"].fillna(0.02)
    merged["unemployment_rate"] = merged["unemployment_rate"].fillna(0.045)
    merged["hpi_change"] = merged["hpi_change"].fillna(0.0)

    # ── Loan-level features ──────────────────────────────────
    merged["loan_purpose_purchase"] = np.where(
        merged["loan_purpose"].str.lower().str.contains("purchase"), 1, 0
    )
    merged["unit_bucket"] = _encode_unit_bucket(merged["number_of_units"])
    merged["region"] = merged["state"].map(_REGION_MAP).fillna("West")

    # ── Behavioral features ──────────────────────────────────
    merged["loan_age"] = _compute_loan_age(
        merged["origination_date"], merged["reporting_period"]
    )

    # Re-sort for rolling computations after merge
    merged = merged.sort_values(["loan_id", "reporting_period"]).reset_index(drop=True)
    merged["rolling_dq_count_6m"] = _compute_rolling_dq(merged, window=6)
    merged["rolling_dq_count_12m"] = _compute_rolling_dq(merged, window=12)
    merged["months_since_last_delinquency"] = _compute_months_since_last_dq(merged)

    # ── Macro-linked features ────────────────────────────────
    merged["current_ltv"] = _compute_current_ltv(
        merged["original_upb"],
        merged["original_ltv"],
        merged["loan_age"],
        merged["hpi_change"],
        merged["note_rate"],
    )

    # ── Select and validate ──────────────────────────────────
    feature_cols = [
        "loan_id",
        "reporting_period",
        "original_upb",
        "original_ltv",
        "original_dscr",
        "note_rate",
        "loan_purpose_purchase",
        "unit_bucket",
        "region",
        "loan_age",
        "months_since_last_delinquency",
        "rolling_dq_count_6m",
        "rolling_dq_count_12m",
        "unemployment_rate",
        "gdp_growth",
        "hpi_change",
        "current_ltv",
        "is_default",
    ]

    features = merged[feature_cols].copy()
    features = FeatureSchema.validate(features)

    default_rate = features["is_default"].mean()
    logger.info(
        "Feature table built: %d rows, %.2f%% default rate",
        len(features),
        default_rate * 100,
    )
    return features
