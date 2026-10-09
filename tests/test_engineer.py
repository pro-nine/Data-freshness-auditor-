import numpy as np
import pandas as pd
import pytest

from data.simulator import PipelineEcosystemSimulator
from features.engineer import (
    CausalBaseline,
    DAGFeatureExtractor,
    DriftDetectors,
    FreshnessFeatureEngineer,
    compute_diagnostic_state,
)
from features.labels import drift_stage, ground_truth_family, ground_truth_severity


@pytest.fixture(scope="module")
def sim_output():
    return PipelineEcosystemSimulator(n_days=180, seed=42).generate()


@pytest.fixture(scope="module")
def fm(sim_output):
    df, dag = sim_output
    return FreshnessFeatureEngineer(df, dag).build()


def _row(status, read_ok, write_ok):
    return pd.Series({"execution_status": status, "rows_read": 100 if read_ok else 0,
                      "rows_written": 100 if write_ok else 0})


@pytest.mark.parametrize("status, r, w, label, owner", [
    ("SUCCESS", 1, 1, "HEALTHY", "none"),
    ("SUCCESS", 1, 0, "SILENT_BUSINESS_FAILURE", "analytics_engineer"),
    ("SUCCESS", 0, 0, "EXPECTED_EMPTY", "monitor_only"),
    ("SUCCESS", 0, 1, "IMPOSSIBLE_STATE", "data_quality"),
    ("FAILED", 0, 0, "INFRASTRUCTURE_FAILURE", "platform_team"),
    ("FAILED", 1, 1, "EXEC_FAILED_DATA_OK", "orchestration_team"),
    ("FAILED", 1, 0, "PARTIAL_FAILURE", "platform_team"),
    ("FAILED", 0, 1, "IMPOSSIBLE_STATE", "data_quality"),
])
def test_state_machine_covers_all_eight_states(status, r, w, label, owner):
    _, got_label, got_owner, *_ = compute_diagnostic_state(_row(status, r, w))
    assert (got_label, got_owner) == (label, owner)


def test_causal_baseline_removes_weekday_pattern():
    dow = pd.Series(np.tile(np.arange(7), 12))
    lat = pd.Series(50 + np.where(dow >= 5, -20.0, 0.0) + np.random.default_rng(0).normal(0, 1, 84))
    out = CausalBaseline(ref_days=28).transform(lat, dow)
    assert abs(out["ds_z"].iloc[:28].mean()) < 0.5
    assert out["ds_z"][dow >= 5].mean() == pytest.approx(out["ds_z"][dow < 5].mean(), abs=0.5)


def test_baseline_never_uses_data_after_the_reference_window():
    dow = pd.Series(np.tile(np.arange(7), 12))
    rng = np.random.default_rng(1)
    a = pd.Series(50 + rng.normal(0, 1, 84))
    b = a.copy()
    b.iloc[40:] += 500
    za = CausalBaseline(28).transform(a, dow)["ds_z"].iloc[:40]
    zb = CausalBaseline(28).transform(b, dow)["ds_z"].iloc[:40]
    pd.testing.assert_series_equal(za, zb)


def test_ewma_detects_a_ramp_and_ignores_a_single_spike():
    rng = np.random.default_rng(2)
    spike = rng.normal(0, 1, 120)
    spike[70] = 200
    ramp = rng.normal(0, 1, 120)
    ramp[60:] += 0.3 * np.arange(60)
    det = DriftDetectors()
    to_df = lambda z: pd.DataFrame({"ds_z": z})
    assert det.transform(to_df(spike))["ewma_alert"].sum() == 0
    first = det.transform(to_df(ramp))["ewma_alert"].values.argmax()
    assert 60 < first < 90


def test_cusum_is_clipped_so_one_outage_cannot_latch_it():
    z = np.zeros(100)
    z[30] = 500.0
    out = DriftDetectors().transform(pd.DataFrame({"ds_z": z}))
    assert out["cusum_s_pos"].max() <= 2.0 + 1e-9
    assert out["cusum_alert_up"].sum() == 0


def test_dag_is_acyclic_and_features_match_known_tables(sim_output):
    _, dag = sim_output
    assert (dag.number_of_nodes(), dag.number_of_edges()) == (22, 31)
    ex = DAGFeatureExtractor(dag)
    assert ex.get_features("raw_orders")["dag_blast_radius"] == 13
    top = ex.get_features("mart_exec_summary")
    assert (top["dag_depth"], top["dag_ancestry_count"], top["dag_criticality"]) == (5, 17, 4)
    assert all(ex.depth[c] > ex.depth[p] for p, c in dag.edges)


def test_labels_come_from_events_not_from_features(fm):
    assert fm.shape[0] == 3960
    assert fm["event_family"].value_counts().to_dict() == {
        "NONE": 3805, "GRADUAL_DRIFT": 120, "EXPECTED_EMPTY": 16,
        "INFRA_CASCADE": 9, "SILENT_BUSINESS_FAILURE": 8, "PHANTOM_RECOVERY": 2}
    assert set(fm.loc[fm["event_family"] == "INFRA_CASCADE", "severity_label"]) == {4}
    assert set(fm.loc[fm["event_family"] == "SILENT_BUSINESS_FAILURE", "severity_label"]) == {3}
    assert (fm.loc[fm["event_family"] == "NONE", "severity_label"] == 0).all()
    assert (fm["severity_label"] != fm["rule_severity"]).mean() > 0.03


def test_weekend_rows_inside_a_drift_still_count_as_drift(fm):
    rs = fm[(fm["table_name"] == "raw_sessions") & (fm["day_idx"] >= 60)]
    assert (rs["event_family"] == "GRADUAL_DRIFT").all()
    assert len(rs) == 120


def test_drift_stage_boundaries():
    assert list(drift_stage([0, 6, 7, 20, 21, 44, 45, 90])) == [1, 1, 2, 2, 3, 3, 4, 4]


def test_criticality_adjusts_event_severity():
    df = pd.DataFrame({"failure_mode": ["PHANTOM_RECOVERY"] * 3, "table_name": list("abc"),
                       "day_idx": [1, 2, 3], "dag_criticality": [1, 3, 4]})
    assert list(ground_truth_severity(df)) == [1, 2, 3]
    assert list(ground_truth_family(df)) == ["PHANTOM_RECOVERY"] * 3


def test_infra_and_silent_rows_get_their_owners(fm):
    infra = fm[fm["failure_mode"] == "INFRA_CASCADE"]
    assert set(infra["diag_state_label"]) == {"INFRASTRUCTURE_FAILURE"}
    silent = fm[fm["failure_mode"] == "SILENT_BUSINESS_FAILURE"]
    assert set(silent["diag_owner"]) == {"analytics_engineer"}


def test_ewma_flags_only_the_drifting_table(fm):
    alerts = fm[fm["ewma_alert"] == 1]
    assert set(alerts["table_name"]) == {"raw_sessions"}
    assert 60 < alerts["day_idx"].min() <= 90


def test_cusum_is_faster_than_ewma_but_noisier(fm):
    rs = fm[fm["table_name"] == "raw_sessions"]
    assert rs.loc[rs["cusum_alert_up"] == 1, "day_idx"].min() < rs.loc[rs["ewma_alert"] == 1, "day_idx"].min()
    normal = fm[fm["event_family"].isin(["NONE", "EXPECTED_EMPTY"])]
    assert normal.loc[normal["cusum_alert_up"] == 1, "table_name"].nunique() >= 5


@pytest.mark.xfail(strict=True, reason=(
    "Known gap: a retried run with normal row counts is labelled HEALTHY by the 3-bit "
    "state machine. The hybrid floor catches it through latency instead."))
def test_state_machine_alone_does_not_call_phantom_recovery_healthy(fm):
    rows = fm[fm["failure_mode"] == "PHANTOM_RECOVERY"]
    assert (rows["diag_state_label"] != "HEALTHY").all()
