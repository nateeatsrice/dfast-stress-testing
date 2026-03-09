"""Pandera DataFrameSchemas for raw and processed Fannie Mae Multifamily data.

Three schemas enforce data contracts at pipeline boundaries:
- AcquisitionSchema: raw origination-level loan attributes
- PerformanceSchema: monthly loan-level performance observations
- FeatureSchema: the final modeling-ready feature table
"""

import pandera as pa
from pandera import Column, DataFrameSchema, Check

# ── Raw Acquisition Data ─────────────────────────────────────

AcquisitionSchema = DataFrameSchema(
    columns={
        "loan_id": Column(str, nullable=False, unique=True),
        "origination_date": Column("datetime64[ns]", nullable=False),
        "original_upb": Column(float, Check.greater_than(0), nullable=False),
        "original_ltv": Column(float, Check.in_range(0.0, 1.5), nullable=False),
        "original_dscr": Column(float, Check.greater_than(0), nullable=True),
        "note_rate": Column(float, Check.in_range(0.0, 0.20), nullable=False),
        "property_type": Column(str, nullable=False),
        "state": Column(str, Check.str_length(2, 2), nullable=False),
        "loan_purpose": Column(str, nullable=False),
        "number_of_units": Column(int, Check.greater_than(0), nullable=False),
        "maturity_date": Column("datetime64[ns]", nullable=True),
    },
    coerce=True,
    strict=False,
    name="AcquisitionSchema",
)


# ── Raw Performance Data ─────────────────────────────────────

PerformanceSchema = DataFrameSchema(
    columns={
        "loan_id": Column(str, nullable=False),
        "reporting_period": Column("datetime64[ns]", nullable=False),
        "current_upb": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "delinquency_status": Column(int, Check.in_range(0, 3), nullable=False),
        "zero_balance_code": Column(str, nullable=True),
    },
    coerce=True,
    strict=False,
    name="PerformanceSchema",
)


# ── Modeling Feature Table ────────────────────────────────────

FeatureSchema = DataFrameSchema(
    columns={
        "loan_id": Column(str, nullable=False),
        "reporting_period": Column("datetime64[ns]", nullable=False),
        # Loan-level features
        "original_upb": Column(float, nullable=False),
        "original_ltv": Column(float, nullable=False),
        "original_dscr": Column(float, nullable=True),
        "note_rate": Column(float, nullable=False),
        "loan_purpose_purchase": Column(int, Check.isin([0, 1]), nullable=False),
        "unit_bucket": Column(str, Check.isin(["small", "medium", "large"]), nullable=False),
        "region": Column(
            str,
            Check.isin(["Northeast", "Southeast", "Midwest", "West"]),
            nullable=False,
        ),
        # Behavioral features
        "loan_age": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
        "months_since_last_delinquency": Column(int, nullable=True),
        "rolling_dq_count_6m": Column(int, Check.in_range(0, 6), nullable=False),
        "rolling_dq_count_12m": Column(int, Check.in_range(0, 12), nullable=False),
        # Macro features
        "unemployment_rate": Column(float, nullable=False),
        "gdp_growth": Column(float, nullable=False),
        "hpi_change": Column(float, nullable=False),
        "current_ltv": Column(float, nullable=False),
        # Target
        "is_default": Column(int, Check.isin([0, 1]), nullable=False),
    },
    coerce=True,
    strict=False,
    name="FeatureSchema",
)
