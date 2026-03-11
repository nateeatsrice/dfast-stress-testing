"""Generate realistic synthetic Fannie Mae Multifamily loan data for development.

Produces three pipe-delimited CSV files:
    - acquisition.csv (5,000 loans)
    - performance.csv (monthly performance records)
    - macro.csv (monthly macro time series 2015-2024)

Usage:
    python scripts/generate_sample_data.py
    # or: make generate-data
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────
NUM_LOANS = 5_000
OUTPUT_DIR = Path("./data")
SEED = 42

# State weights approximate Fannie Mae MF portfolio concentration
STATE_WEIGHTS: dict[str, float] = {
    "CA": 0.14, "TX": 0.10, "FL": 0.08, "NY": 0.07, "IL": 0.05,
    "GA": 0.04, "NC": 0.04, "VA": 0.04, "OH": 0.03, "PA": 0.03,
    "AZ": 0.03, "CO": 0.03, "WA": 0.03, "NJ": 0.03, "TN": 0.03,
    "MD": 0.02, "MN": 0.02, "MI": 0.02, "MO": 0.02, "IN": 0.02,
    "SC": 0.02, "MA": 0.02, "OR": 0.02, "AL": 0.01, "NV": 0.01,
    "UT": 0.01, "KY": 0.01, "CT": 0.01, "OK": 0.01, "LA": 0.01,
    "NM": 0.005, "NE": 0.005, "WI": 0.005, "AR": 0.005, "KS": 0.005,
}

PROPERTY_TYPES = ["Apartment", "Coop", "Manufactured", "Other"]
PROPERTY_WEIGHTS = [0.70, 0.10, 0.05, 0.15]


def _generate_acquisition(rng: np.random.Generator) -> pd.DataFrame:
    """Generate synthetic acquisition records for multifamily loans."""
    logger.info("Generating %d acquisition records...", NUM_LOANS)

    states = list(STATE_WEIGHTS.keys())
    weights = np.array(list(STATE_WEIGHTS.values()))
    weights = weights / weights.sum()

    # Origination dates: uniform over 2015-01 to 2023-12
    orig_start = pd.Timestamp("2015-01-01")
    orig_end = pd.Timestamp("2023-12-31")
    orig_days = (orig_end - orig_start).days
    origination_dates = pd.to_datetime(
        orig_start.value + rng.integers(0, orig_days, size=NUM_LOANS) * 86_400_000_000_000
    )

    # UPB: lognormal, mean ~$5M
    log_upb = rng.normal(loc=np.log(5_000_000), scale=0.7, size=NUM_LOANS)
    original_upb = np.clip(np.exp(log_upb), 500_000, 50_000_000)

    # LTV: normal, mean 0.65
    original_ltv = np.clip(rng.normal(0.65, 0.10, size=NUM_LOANS), 0.40, 0.85)

    # DSCR: normal, mean 1.35
    original_dscr = np.clip(rng.normal(1.35, 0.20, size=NUM_LOANS), 0.90, 2.00)

    # Note rate: normal, mean 4.5%
    note_rate = np.clip(rng.normal(0.045, 0.008, size=NUM_LOANS), 0.025, 0.075)

    # Maturity: 7-10 years from origination
    term_years = rng.choice([7, 10], size=NUM_LOANS, p=[0.4, 0.6])
    maturity_dates = origination_dates + pd.to_timedelta(term_years * 365, unit="D")

    df = pd.DataFrame({
        "loan_id": [f"LN{i:06d}" for i in range(NUM_LOANS)],
        "origination_date": origination_dates,
        "original_upb": np.round(original_upb, 2),
        "original_ltv": np.round(original_ltv, 4),
        "original_dscr": np.round(original_dscr, 4),
        "note_rate": np.round(note_rate, 5),
        "property_type": rng.choice(PROPERTY_TYPES, size=NUM_LOANS, p=PROPERTY_WEIGHTS),
        "state": rng.choice(states, size=NUM_LOANS, p=weights),
        "loan_purpose": rng.choice(["Purchase", "Refinance"], size=NUM_LOANS, p=[0.55, 0.45]),
        "number_of_units": np.clip(
            rng.lognormal(mean=np.log(80), sigma=0.8, size=NUM_LOANS).astype(int), 5, 500
        ),
        "maturity_date": maturity_dates,
    })

    # Introduce ~2% missing DSCR values (realistic)
    mask = rng.random(NUM_LOANS) < 0.02
    df.loc[mask, "original_dscr"] = np.nan

    logger.info("Acquisition data: %d rows", len(df))
    return df


def _generate_performance(
    acquisition_df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Generate monthly performance records with realistic delinquency transitions."""
    logger.info("Generating performance records...")

    records: list[dict] = []
    cutoff = pd.Timestamp("2024-06-01")

    # Delinquency transition probabilities: P(new_state | current_state)
    # States: 0=current, 1=30-day, 2=60-day, 3=90+
    transition_probs = {
        0: {0: 0.98, 1: 0.02, 2: 0.00, 3: 0.00},
        1: {0: 0.60, 1: 0.10, 2: 0.30, 3: 0.00},
        2: {0: 0.20, 1: 0.00, 2: 0.30, 3: 0.50},
        3: {0: 0.10, 1: 0.00, 2: 0.10, 3: 0.80},
    }

    for _, loan in acquisition_df.iterrows():
        loan_id = loan["loan_id"]
        start = loan["origination_date"] + pd.DateOffset(months=1)
        # Generate 12–60 months of performance (capped at cutoff)
        n_months = min(rng.integers(12, 61), max(1, (cutoff.year - start.year) * 12 + cutoff.month - start.month))

        dq_state = 0
        for m in range(n_months):
            report_date = start + pd.DateOffset(months=m)
            if report_date > cutoff:
                break

            # Transition
            probs = transition_probs[dq_state]
            states = list(probs.keys())
            weights = list(probs.values())
            dq_state = rng.choice(states, p=weights)

            # Simple amortization for current_upb
            amort = 1.0 - (m / 360)
            current_upb = round(loan["original_upb"] * max(amort, 0.0), 2)

            zbc = ""
            if dq_state == 3 and rng.random() < 0.05:
                zbc = "03"  # third-party sale / liquidation

            records.append({
                "loan_id": loan_id,
                "reporting_period": report_date,
                "current_upb": current_upb,
                "delinquency_status": str(dq_state),
                "zero_balance_code": zbc,
            })

    df = pd.DataFrame(records)
    n_defaults = (df["delinquency_status"].astype(int) >= 3).sum()
    logger.info(
        "Performance data: %d rows, %d observations at 90+ DPD (%.2f%%)",
        len(df),
        n_defaults,
        n_defaults / len(df) * 100,
    )
    return df


def _generate_macro(rng: np.random.Generator) -> pd.DataFrame:
    """Generate synthetic monthly macro economic time series (2015–2024)."""
    logger.info("Generating macro time series...")

    dates = pd.date_range("2015-01-01", "2024-06-01", freq="MS")
    n = len(dates)

    # GDP growth: baseline ~2.2% with business-cycle variation
    gdp_trend = np.linspace(0.025, 0.018, n)
    gdp_noise = rng.normal(0, 0.003, n)
    # Add a mild recession around 2020 (COVID analog)
    covid_mask = (dates >= "2020-03-01") & (dates <= "2020-09-01")
    gdp_shock = np.zeros(n)
    gdp_shock[covid_mask] = -0.08
    gdp_growth = gdp_trend + gdp_noise + gdp_shock

    # Unemployment: baseline ~4%, spikes during recession
    unemp_trend = np.linspace(0.050, 0.038, n)
    unemp_noise = rng.normal(0, 0.002, n)
    unemp_shock = np.zeros(n)
    unemp_shock[covid_mask] = 0.06
    unemployment_rate = np.clip(unemp_trend + unemp_noise + unemp_shock, 0.03, 0.15)

    # HPI: cumulative appreciation ~3%/yr with dip during recession
    monthly_hpi = np.full(n, 0.03 / 12)
    monthly_hpi[covid_mask] = -0.02
    hpi_change = np.cumsum(monthly_hpi + rng.normal(0, 0.002, n))

    df = pd.DataFrame({
        "date": dates,
        "gdp_growth": np.round(gdp_growth, 5),
        "unemployment_rate": np.round(unemployment_rate, 5),
        "hpi_change": np.round(hpi_change, 5),
    })

    logger.info("Macro data: %d rows", len(df))
    return df


def main() -> None:
    """Generate all synthetic data files and write to OUTPUT_DIR."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    acquisition = _generate_acquisition(rng)
    performance = _generate_performance(acquisition, rng)
    macro = _generate_macro(rng)

    # Write pipe-delimited CSVs
    acquisition.to_csv(OUTPUT_DIR / "acquisition.csv", sep="|", index=False)
    performance.to_csv(OUTPUT_DIR / "performance.csv", sep="|", index=False)
    macro.to_csv(OUTPUT_DIR / "macro.csv", sep="|", index=False)

    logger.info("=" * 60)
    logger.info("Synthetic data written to %s/", OUTPUT_DIR)
    logger.info("  acquisition.csv : %d loans", len(acquisition))
    logger.info("  performance.csv : %d records", len(performance))
    logger.info("  macro.csv       : %d months", len(macro))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
