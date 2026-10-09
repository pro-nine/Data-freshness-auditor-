import numpy as np
import pandas as pd
import pytest

from data.simulator import PipelineEcosystemSimulator
from features.engineer import (
    CUSUMEngine,
    DAGFeatureExtractor,
    FreshnessFeatureEngineer,
    compute_diagnostic_state,
)


@pytest.fixture(scope="module")
def sim_output():
    return PipelineEcosystemSimulator(n_days=180, seed=42).generate()


@pytest.fixture(scope="module")
def feature_matrix(sim_output):
    df, dag = sim_output
    return FreshnessFeatureEngineer(df, dag).build()


def _row(status, read_ok, write_ok):
    return pd.Series({
        "execution_status": status,
        "rows_read": 100 if read_ok else 0,
        "rows_written": 100 if write_ok else 0,
    })


@pytest.mark.parametrize(
    "status, read_ok, write_ok, label, owner",
    [
        ("SUCCESS", 1, 1, "HEALTHY", "none"),
        ("SUCCESS", 1, 0, "SILENT_BUSINESS_FAILURE", "analytics_engineer"),
        ("SUCCESS", 0, 0, "EXPECTED_EMPTY", "monitor_only"),
        ("SUCCESS", 0, 1, "IMPOSSIBLE_STATE", "data_quality"),
        ("FAILED", 0, 0, "INFRASTRUCTURE_FAILURE", "platform_team"),
        ("FAILED", 1, 1, "EXEC_FAILED_DATA_OK", "orchestration_team"),
        ("FAILED", 1, 0, "PARTIAL_FAILURE", "platform_team"),
        ("FAILED", 0, 1, "IMPOSSIBLE_STATE", "data_quality"),
    ],
)
def test_three_bit_state_machine_covers_all_eight_states(status, read_ok, write_ok, label, owner):
    _, got_label, got_owner, *_ = compute_diagnostic_state(_row(status, read_ok, write_ok))
    assert (got_label, got_owner) == (label, owner)


def test_retried_run_with_normal_rows_is_treated_as_executed_ok():
    _, label, *_ = compute_diagnostic_state(_row("RETRIED", 1, 1))
    assert label == "HEALTHY"


def test_cusum_false_alarm_rate_on_stationary_noise_is_low():
    x = pd.Series(np.random.default_rng(0).normal(0, 1, 200))
    assert CUSUMEngine().fit_transform(x)["cusum_alert_up"].mean() < 0.05


def test_cusum_detects_a_three_sigma_step_after_the_calibration_window():
    rng = np.random.default_rng(3)
    x = pd.Series(np.r_[rng.normal(0, 1, 100), rng.normal(3, 1, 60)])
    out = CUSUMEngine().fit_transform(x)
    after = out.index[(out["cusum_alert_up"] == 1) & (out.index >= 100)]
    assert len(after) > 0 and after.min() - 100 <= 5


def test_cusum_drift_score_is_clipped():
    x = pd.Series(np.r_[np.zeros(60), np.full(40, 50.0)]) + np.random.default_rng(1).normal(0, 0.1, 100)
    assert CUSUMEngine().fit_transform(x)["cusum_drift_score"].max() == pytest.approx(5.0)


def test_dag_is_acyclic_with_expected_size(sim_output):
    _, dag = sim_output
    assert dag.number_of_nodes() == 22
    assert dag.number_of_edges() == 31


def test_dag_features_for_known_tables(sim_output):
    _, dag = sim_output
    ex = DAGFeatureExtractor(dag)
    raw_orders = ex.get_features("raw_orders")
    assert raw_orders["dag_blast_radius"] == 13
    assert raw_orders["dag_depth"] == 0
    exec_summary = ex.get_features("mart_exec_summary")
    assert exec_summary["dag_blast_radius"] == 0
    assert exec_summary["dag_depth"] == 5
    assert exec_summary["dag_ancestry_count"] == 17
    assert exec_summary["dag_criticality"] == 4


def test_every_node_is_deeper_than_its_parents(sim_output):
    _, dag = sim_output
    ex = DAGFeatureExtractor(dag)
    for parent, child in dag.edges:
        assert ex.depth[child] > ex.depth[parent]


def test_feature_matrix_shape_and_severity_scale(feature_matrix):
    assert feature_matrix.shape == (3960, 60)
    assert set(feature_matrix["severity_label"].unique()) <= {0, 1, 2, 3, 4}
    assert (feature_matrix["severity_label"] == 0).mean() > 0.5


def test_infrastructure_cascade_rows_get_the_infrastructure_state(feature_matrix):
    rows = feature_matrix[feature_matrix["failure_mode"] == "INFRA_CASCADE"]
    assert len(rows) == 9
    assert set(rows["diag_state_label"]) == {"INFRASTRUCTURE_FAILURE"}
    assert set(rows["diag_owner"]) == {"platform_team"}


def test_silent_failure_rows_are_routed_to_analytics_engineering(feature_matrix):
    rows = feature_matrix[feature_matrix["failure_mode"] == "SILENT_BUSINESS_FAILURE"]
    assert set(rows["execution_status"]) == {"SUCCESS"}
    assert set(rows["diag_state_label"]) == {"SILENT_BUSINESS_FAILURE"}
    assert set(rows["diag_owner"]) == {"analytics_engineer"}


@pytest.mark.xfail(strict=True, reason=(
    "Known gap: PHANTOM_RECOVERY rows have status RETRIED and normal row counts, "
    "so the 3-bit state machine labels them HEALTHY. Only latency can reveal them."))
def test_phantom_recovery_is_not_labelled_healthy(feature_matrix):
    rows = feature_matrix[feature_matrix["failure_mode"] == "PHANTOM_RECOVERY"]
    assert (rows["diag_state_label"] != "HEALTHY").all()


@pytest.mark.xfail(strict=True, reason=(
    "Known issue: CUSUM runs on the STL residual, and STL moves the drift into the "
    "trend. raw_sessions drifts from day 60 but its residual-CUSUM first alerts on day 143."))
def test_residual_cusum_catches_raw_sessions_drift_within_three_weeks(feature_matrix):
    rs = feature_matrix[feature_matrix["table_name"] == "raw_sessions"]
    first_alert = rs.loc[rs["cusum_alert_up"] == 1, "day_idx"].min()
    assert first_alert <= 81
