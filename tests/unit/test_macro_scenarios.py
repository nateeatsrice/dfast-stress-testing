"""Unit tests for dfast.data.macro_scenarios module."""

import numpy as np
import pandas as pd
import pytest

from dfast.data.macro_scenarios import (
    SCENARIOS,
    apply_scenario_overlay,
    get_scenario,
)


class TestScenarioDefinitions:
    """Tests for the scenario constants."""

    def test_all_three_scenarios_defined(self) -> None:
        """Baseline, adverse, and severely_adverse must all exist."""
        assert "baseline" in SCENARIOS
        assert "adverse" in SCENARIOS
        assert "severely_adverse" in SCENARIOS

    def test_scenarios_have_required_fields(self) -> None:
        """Each scenario must define gdp_growth, unemployment_rate, hpi_change."""
        for name, scenario in SCENARIOS.items():
            assert hasattr(scenario, "gdp_growth"), f"{name} missing gdp_growth"
            assert hasattr(scenario, "unemployment_rate"), f"{name} missing unemployment_rate"
            assert hasattr(scenario, "hpi_change"), f"{name} missing hpi_change"

    def test_severity_ordering(self) -> None:
        """Adverse scenarios should have worse macro conditions than baseline."""
        b = SCENARIOS["baseline"]
        a = SCENARIOS["adverse"]
        sa = SCENARIOS["severely_adverse"]

        assert a.gdp_growth < b.gdp_growth
        assert sa.gdp_growth < a.gdp_growth

        assert a.unemployment_rate > b.unemployment_rate
        assert sa.unemployment_rate > a.unemployment_rate

        assert a.hpi_change < b.hpi_change
        assert sa.hpi_change < a.hpi_change

    def test_get_unknown_scenario_raises(self) -> None:
        """Requesting an unknown scenario name should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown scenario"):
            get_scenario("apocalypse")


class TestApplyScenarioOverlay:
    """Tests for apply_scenario_overlay."""

    def test_baseline_preserves_macro_values(self, sample_features_df: pd.DataFrame) -> None:
        """Baseline scenario should set macro columns to baseline values."""
        result = apply_scenario_overlay(sample_features_df, "baseline")
        baseline = SCENARIOS["baseline"]

        assert (result["gdp_growth"] == baseline.gdp_growth).all()
        assert (result["unemployment_rate"] == baseline.unemployment_rate).all()
        assert (result["hpi_change"] == baseline.hpi_change).all()

    def test_adverse_shifts_unemployment(self, sample_features_df: pd.DataFrame) -> None:
        """Adverse scenario should set unemployment to 7%."""
        result = apply_scenario_overlay(sample_features_df, "adverse")

        assert (result["unemployment_rate"] == 0.07).all()

    def test_severely_adverse_shifts_hpi(self, sample_features_df: pd.DataFrame) -> None:
        """Severely adverse scenario should set HPI change to -30%."""
        result = apply_scenario_overlay(sample_features_df, "severely_adverse")

        assert (result["hpi_change"] == -0.30).all()

    def test_current_ltv_increases_under_stress(self, sample_features_df: pd.DataFrame) -> None:
        """LTV should increase (worsen) under negative HPI stress."""
        baseline_result = apply_scenario_overlay(sample_features_df, "baseline")
        adverse_result = apply_scenario_overlay(sample_features_df, "severely_adverse")

        mean_ltv_baseline = baseline_result["current_ltv"].mean()
        mean_ltv_stressed = adverse_result["current_ltv"].mean()

        assert mean_ltv_stressed > mean_ltv_baseline, (
            f"Stressed LTV ({mean_ltv_stressed:.4f}) should exceed "
            f"baseline LTV ({mean_ltv_baseline:.4f})"
        )

    def test_overlay_does_not_mutate_input(self, sample_features_df: pd.DataFrame) -> None:
        """Applying an overlay should return a new DataFrame, not modify input."""
        original_gdp = sample_features_df["gdp_growth"].copy()
        _ = apply_scenario_overlay(sample_features_df, "severely_adverse")

        pd.testing.assert_series_equal(sample_features_df["gdp_growth"], original_gdp)
