import numpy as np
import pytest

from data.simulator import PipelineEcosystemSimulator
from features.drift import detect_trend_drift, trend_drift_alert
from features.engineer import FreshnessFeatureEngineer


@pytest.fixture(scope="module")
def feature_matrix():
    df, dag = PipelineEcosystemSimulator(n_days=180, seed=42).generate()
    return FreshnessFeatureEngineer(df, dag).build()


def test_flat_trend_is_not_flagged():
    rng = np.random.default_rng(0)
    res = trend_drift_alert(np.full(120, 50.0) + rng.normal(0, 0.2, 120), rng.normal(0, 1, 120))
    assert res["drift_detected"] is False
    assert res["first_alert_index"] is None


def test_ramp_is_flagged_shortly_after_onset():
    trend = np.r_[np.full(60, 50.0), 50.0 + 0.5 * np.arange(1, 61)]
    resid = np.random.default_rng(1).normal(0, 1, 120)
    res = trend_drift_alert(trend, resid)
    assert res["drift_detected"] is True
    assert 60 < res["first_alert_index"] <= 80


def test_short_spike_does_not_trigger_the_run_rule():
    trend = np.full(120, 50.0)
    trend[80:82] += 100.0
    resid = np.random.default_rng(2).normal(0, 1, 120)
    assert trend_drift_alert(trend, resid)["drift_detected"] is False


def test_length_mismatch_and_short_series_raise():
    with pytest.raises(ValueError):
        trend_drift_alert(np.zeros(100), np.zeros(99))
    with pytest.raises(ValueError):
        trend_drift_alert(np.zeros(10), np.zeros(10))


def test_only_raw_sessions_is_flagged_on_the_simulated_ecosystem(feature_matrix):
    out = detect_trend_drift(feature_matrix)
    flagged = out[out["drift_detected"]]
    assert list(flagged["table_name"]) == ["raw_sessions"]
    first_day = int(flagged["first_alert_day"].iloc[0])
    assert 60 < first_day <= 81


def test_drift_margin_over_the_next_highest_table(feature_matrix):
    out = detect_trend_drift(feature_matrix).sort_values("peak_trend_z", ascending=False)
    assert out["peak_trend_z"].iloc[0] > 5 * out["peak_trend_z"].iloc[1]


def test_missing_columns_are_reported(feature_matrix):
    with pytest.raises(KeyError):
        detect_trend_drift(feature_matrix.drop(columns=["stl_trend"]))
