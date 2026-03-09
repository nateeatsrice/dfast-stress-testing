"""DFAST stress scenario definitions and overlay application.

Defines the three standard DFAST scenarios (baseline, adverse, severely adverse)
and provides functions to apply macro-economic overlays to the feature table,
producing stressed feature sets ready for model scoring.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from dfast.data.feature_engineering import _compute_current_ltv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StressScenario:
    """A single DFAST macro-economic stress scenario.

    Attributes:
        name: Human-readable scenario label.
        gdp_growth: Annualized GDP growth rate (e.g., -0.06 = -6%).
        unemployment_rate: National unemployment rate (e.g., 0.10 = 10%).
        hpi_change: Cumulative house price index change (e.g., -0.30 = -30%).
    """

    name: str
    gdp_growth: float
    unemployment_rate: float
    hpi_change: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary."""
        return asdict(self)


# ── Scenario Definitions ─────────────────────────────────────
# Aligned with Federal Reserve DFAST / FHFA supervisory scenarios

BASELINE = StressScenario(
    name="baseline",
    gdp_growth=0.02,
    unemployment_rate=0.045,
    hpi_change=0.00,
)

ADVERSE = StressScenario(
    name="adverse",
    gdp_growth=-0.02,
    unemployment_rate=0.07,
    hpi_change=-0.15,
)

SEVERELY_ADVERSE = StressScenario(
    name="severely_adverse",
    gdp_growth=-0.06,
    unemployment_rate=0.10,
    hpi_change=-0.30,
)

SCENARIOS: dict[str, StressScenario] = {
    "baseline": BASELINE,
    "adverse": ADVERSE,
    "severely_adverse": SEVERELY_ADVERSE,
}


def get_scenario(name: str) -> StressScenario:
    """Retrieve a scenario by name.

    Args:
        name: One of ``"baseline"``, ``"adverse"``, ``"severely_adverse"``.

    Returns:
        The corresponding ``StressScenario``.

    Raises:
        KeyError: If the scenario name is not recognized.
    """
    if name not in SCENARIOS:
        raise KeyError(
            f"Unknown scenario {name!r}. Choose from: {list(SCENARIOS.keys())}"
        )
    return SCENARIOS[name]


def apply_scenario_overlay(
    features_df: pd.DataFrame,
    scenario_name: str,
) -> pd.DataFrame:
    """Apply a stress scenario to the feature table.

    Replaces macro columns (gdp_growth, unemployment_rate, hpi_change) with
    the scenario values and recomputes ``current_ltv`` based on the stressed
    HPI.

    Args:
        features_df: The feature table with macro columns.
        scenario_name: Key into ``SCENARIOS``.

    Returns:
        A new DataFrame with macro columns overwritten and ``current_ltv``
        recalculated under the stressed HPI scenario.
    """
    scenario = get_scenario(scenario_name)
    out = features_df.copy()

    logger.info(
        "Applying scenario %r: GDP=%.1f%%, Unemp=%.1f%%, HPI=%.1f%%",
        scenario.name,
        scenario.gdp_growth * 100,
        scenario.unemployment_rate * 100,
        scenario.hpi_change * 100,
    )

    # Overwrite macro columns
    out["gdp_growth"] = scenario.gdp_growth
    out["unemployment_rate"] = scenario.unemployment_rate
    out["hpi_change"] = scenario.hpi_change

    # Recompute current LTV under stressed HPI
    out["current_ltv"] = _compute_current_ltv(
        out["original_upb"],
        out["original_ltv"],
        out["loan_age"],
        out["hpi_change"],
        out["note_rate"],
    )

    mean_ltv = out["current_ltv"].mean()
    logger.info(
        "Scenario %r applied: mean current LTV = %.3f",
        scenario.name,
        mean_ltv,
    )
    return out
