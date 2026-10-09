import pandas as pd
import pytest

from data.simulator import DAG_SCHEMA, PipelineEcosystemSimulator


@pytest.fixture(scope="module")
def runs():
    df, _ = PipelineEcosystemSimulator(n_days=180, seed=42).generate()
    return df


def test_same_seed_gives_identical_runs(runs):
    again, _ = PipelineEcosystemSimulator(n_days=180, seed=42).generate()
    pd.testing.assert_frame_equal(runs, again)


def test_shape_and_failure_mode_counts(runs):
    assert len(runs) == 22 * 180
    assert runs["failure_mode"].value_counts().to_dict() == {
        "NONE": 3805,
        "GRADUAL_DRIFT": 86,
        "EXPECTED_EMPTY": 50,
        "INFRA_CASCADE": 9,
        "SILENT_BUSINESS_FAILURE": 8,
        "PHANTOM_RECOVERY": 2,
    }


def test_infrastructure_cascade_hits_only_postgres_tables_on_three_days(runs):
    hit = runs[runs["failure_mode"] == "INFRA_CASCADE"]
    assert set(hit["day_idx"]) == {30, 95, 142}
    assert set(hit["source_system"]) == {"postgres_prod"}
    assert set(hit["execution_status"]) == {"FAILED"}
    assert (hit["rows_written"] == 0).all()
    postgres = {t for t, m in DAG_SCHEMA.items() if m["source"] == "postgres_prod"}
    assert set(hit["table_name"]) == postgres


def test_silent_failure_keeps_success_status_but_loses_rows(runs):
    hit = runs[runs["failure_mode"] == "SILENT_BUSINESS_FAILURE"]
    assert set(hit["table_name"]) == {"stg_payments"}
    assert sorted(hit["day_idx"]) == list(range(55, 63))
    assert set(hit["execution_status"]) == {"SUCCESS"}
    assert (hit["rows_written"] < 5).all()
    assert (hit["rows_read"] > 1000).all()


def test_gradual_drift_is_confined_to_raw_sessions_and_grows(runs):
    drift = runs[runs["failure_mode"] == "GRADUAL_DRIFT"].sort_values("day_idx")
    assert set(drift["table_name"]) == {"raw_sessions"}
    assert drift["day_idx"].min() == 60
    assert drift["latency_minutes"].iloc[-1] > drift["latency_minutes"].iloc[0] + 150


def test_expected_empty_is_raw_sessions_on_weekends_only(runs):
    empty = runs[runs["failure_mode"] == "EXPECTED_EMPTY"]
    assert set(empty["table_name"]) == {"raw_sessions"}
    assert set(empty["is_weekend"]) == {1}
    assert (empty["rows_read"] == 0).all()


def test_phantom_recovery_is_retried_on_two_days(runs):
    hit = runs[runs["failure_mode"] == "PHANTOM_RECOVERY"]
    assert set(hit["table_name"]) == {"mart_finance_close"}
    assert sorted(hit["day_idx"]) == [110, 111]
    assert set(hit["execution_status"]) == {"RETRIED"}
