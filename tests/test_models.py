import numpy as np
import pandas as pd
import pytest

from data.simulator import PipelineEcosystemSimulator
from features.engineer import FreshnessFeatureEngineer
from models.cascading_classifier import (
    CascadingFreshnessClassifier,
    OrdinalDecompositionRanker,
    expected_calibration_error,
)
from models.evaluation import compare_detectors, event_onsets
from models.honest_eval import split_failure_coverage, uncovered_failure_modes
from models.train import (
    DROP_COLS,
    leave_one_family_out,
    select_feature_cols,
    temporal_split,
)


@pytest.fixture(scope="module")
def fm():
    df, dag = PipelineEcosystemSimulator(n_days=180, seed=42).generate()
    return FreshnessFeatureEngineer(df, dag).build()


def _synthetic(n=600, max_sev=3, seed=0):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    sev = np.clip(((x1 + 0.3 * rng.normal(size=n)) * 1.2).round().astype(int), 0, max_sev)
    return pd.DataFrame({"x1": x1, "x2": x2, "severity_label": sev, "diag_state_code": 0,
                         "ewma_alert": 0, "was_retried": 0, "latency_zscore": 0.0})


def test_ece_is_zero_for_perfect_and_large_for_confident_wrong():
    y = np.array([0, 0, 1, 1])
    assert expected_calibration_error(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0.0
    assert expected_calibration_error(y, np.array([1.0, 1.0, 0.0, 0.0])) == 1.0


def test_ordinal_ranker_survives_a_missing_top_class_and_sums_to_one():
    df = _synthetic(max_sev=3)
    df = df[df["severity_label"] > 0]
    rk = OrdinalDecompositionRanker("sklearn").fit(df[["x1", "x2"]], df["severity_label"])
    p = rk.predict_proba(df[["x1", "x2"]])
    assert p.shape == (len(df), 5)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert p[:, 4].max() == pytest.approx(0.0, abs=1e-6) or rk.models_[3] is None


def test_cascade_threshold_is_calibrated_on_validation_and_floors_only_raise():
    df = _synthetic()
    tr, va = df.iloc[:400], df.iloc[400:]
    model = CascadingFreshnessClassifier(["x1", "x2"], backend="sklearn").fit(
        tr, tr["severity_label"], va, va["severity_label"])
    assert 0.1 <= model.stage1.threshold_ <= 0.9
    X = va.copy()
    X.loc[X.index[:5], "diag_state_code"] = 4
    pure = model.predict(X, floors=False)
    hyb = model.predict(X, floors=True)
    assert (hyb["severity_pred"] >= pure["severity_pred"]).all()
    assert (hyb["severity_pred"].iloc[:5] >= 3).all()
    assert (hyb["floor_applied"] == (hyb["severity_pred"] > pure["severity_pred"])).all()
    assert not pure["floor_applied"].any()


def test_compare_detectors_scores_recall_lag_and_false_alarms():
    d = pd.DataFrame({
        "table_name": ["a"] * 6 + ["b"] * 4,
        "day_idx": [0, 1, 2, 3, 4, 5, 0, 1, 2, 3],
        "event_family": ["NONE", "NONE", "GRADUAL_DRIFT", "GRADUAL_DRIFT", "GRADUAL_DRIFT", "GRADUAL_DRIFT"]
        + ["NONE"] * 4,
    })
    flag = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0])
    out = compare_detectors(d, {"det": flag}, event_onsets(d)).iloc[0]
    assert out["row_recall"] == 0.75
    assert out["median_lag_days"] == 1.0
    assert out["false_alarms_per_100_normal"] == pytest.approx(16.67, abs=0.01)


def test_feature_selection_excludes_labels_and_clocks(fm):
    cols = set(select_feature_cols(fm))
    for banned in ["severity_label", "event_family", "rule_severity", "failure_mode", "day_idx",
                   "latency_minutes", "rows_written"]:
        assert banned in DROP_COLS and banned not in cols


def test_test_window_holds_only_gradual_drift(fm):
    train, val, test = temporal_split(fm)
    assert (train["day_idx"].max(), val["day_idx"].max(), test["day_idx"].min()) == (125, 152, 153)
    assert uncovered_failure_modes(fm) == {"INFRA_CASCADE", "SILENT_BUSINESS_FAILURE", "PHANTOM_RECOVERY"}
    assert split_failure_coverage(fm).set_index("split").loc["test", "GRADUAL_DRIFT"] > 0


def test_leave_one_family_out_hybrid_generalises_where_pure_ml_does_not(fm):
    cols = select_feature_cols(fm)
    res = leave_one_family_out(fm, cols, backend="sklearn").set_index(["family", "detector"])
    for fam in ["INFRA_CASCADE", "SILENT_BUSINESS_FAILURE", "PHANTOM_RECOVERY"]:
        assert res.loc[(fam, "hybrid"), "row_recall"] == 1.0
    assert res.loc[("GRADUAL_DRIFT", "hybrid"), "row_recall"] >= 0.8
    assert res.loc[("SILENT_BUSINESS_FAILURE", "cascade_ml"), "row_recall"] < 0.5
    assert res.loc[("PHANTOM_RECOVERY", "cascade_ml"), "row_recall"] < 0.5
    assert res.loc[("GRADUAL_DRIFT", "ewma"), "false_alarms_per_100_normal"] == 0.0
    for fam in ["INFRA_CASCADE", "SILENT_BUSINESS_FAILURE", "PHANTOM_RECOVERY"]:
        assert (res.loc[(fam, "hybrid"), "false_alarms_per_100_normal"]
                <= res.loc[(fam, "rule_baseline"), "false_alarms_per_100_normal"])
