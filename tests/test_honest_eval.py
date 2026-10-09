import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from data.simulator import PipelineEcosystemSimulator
from features.engineer import FreshnessFeatureEngineer
from models.honest_eval import (
    leakage_ablation,
    select_feature_cols,
    split_failure_coverage,
    temporal_split,
    uncovered_failure_modes,
)


@pytest.fixture(scope="module")
def feature_matrix():
    df, dag = PipelineEcosystemSimulator(n_days=180, seed=42).generate()
    return FreshnessFeatureEngineer(df, dag).build()


def _model():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)


def test_split_is_time_ordered_and_disjoint(feature_matrix):
    train, val, test = temporal_split(feature_matrix)
    assert train["day_idx"].max() <= val["day_idx"].min()
    assert val["day_idx"].max() <= test["day_idx"].min()
    assert len(train) + len(val) + len(test) == len(feature_matrix)
    assert (train["day_idx"].max(), val["day_idx"].max(), test["day_idx"].min()) == (125, 152, 153)


def test_test_window_contains_only_gradual_drift_among_injected_failures(feature_matrix):
    coverage = split_failure_coverage(feature_matrix).set_index("split")
    assert coverage.loc["test", "GRADUAL_DRIFT"] > 0
    assert uncovered_failure_modes(feature_matrix, "test") == {
        "INFRA_CASCADE", "SILENT_BUSINESS_FAILURE", "PHANTOM_RECOVERY",
    }


def test_every_failure_mode_is_seen_somewhere_before_the_test_window(feature_matrix):
    seen = set()
    for split in ("train", "val"):
        cov = split_failure_coverage(feature_matrix).set_index("split").loc[split]
        seen |= {m for m in ("INFRA_CASCADE", "SILENT_BUSINESS_FAILURE", "GRADUAL_DRIFT") if cov[m] > 0}
    assert seen == {"INFRA_CASCADE", "SILENT_BUSINESS_FAILURE", "GRADUAL_DRIFT"}


def test_feature_selection_excludes_leaky_raw_columns(feature_matrix):
    cols = select_feature_cols(feature_matrix)
    assert not {"latency_minutes", "rows_read", "rows_written", "severity_label"} & set(cols)
    assert "cusum_drift_score" in cols


def test_stage1_auc_is_mostly_rule_recovery(feature_matrix):
    out = leakage_ablation(feature_matrix, model_factory=_model).set_index("variant")
    assert out.loc["all_features", "auc"] > 0.99
    assert out.loc["rule_inputs_only", "auc"] > 0.99
    assert out.loc["without_rule_inputs", "auc"] < 0.95
    assert out.loc["without_rule_inputs", "n_features"] < out.loc["all_features", "n_features"]
