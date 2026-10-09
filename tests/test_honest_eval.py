import pandas as pd

from models.honest_eval import split_failure_coverage, uncovered_failure_modes


def test_coverage_table_and_uncovered_modes_on_a_tiny_frame():
    df = pd.DataFrame({"day_idx": [0, 130, 160],
                       "event_family": ["INFRA_CASCADE", "SILENT_BUSINESS_FAILURE", "GRADUAL_DRIFT"]})
    cov = split_failure_coverage(df).set_index("split")
    assert cov.loc["test", "GRADUAL_DRIFT"] == 1
    assert uncovered_failure_modes(df) == {"INFRA_CASCADE", "SILENT_BUSINESS_FAILURE", "PHANTOM_RECOVERY"}
