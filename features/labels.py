"""
Ground-truth severity from injected events
==========================================
`severity_label` must not be computed from the features the model sees. These
labels come from the simulator's `failure_mode` column, the table's criticality
and, for drift, the number of days since the drift began.

Base severity
  INFRA_CASCADE                           -> 4 (CRITICAL, the source is down)
  SILENT_BUSINESS_FAILURE                 -> 3 (DEGRADED)
  PHANTOM_RECOVERY                        -> 2 (WARN)
  GRADUAL_DRIFT                           -> by days since onset:
                                             <7: 1, <21: 2, <45: 3, else 4
  NONE, EXPECTED_EMPTY                    -> 0
Criticality adjustment for event rows: CRITICAL table +1, LOW table -1,
then clipped to [1, 4].
"""
import numpy as np
import pandas as pd

BASE = {"INFRA_CASCADE": 4, "SILENT_BUSINESS_FAILURE": 3, "PHANTOM_RECOVERY": 2}
CRIT_ADJUST = {1: -1, 2: 0, 3: 0, 4: 1}


def drift_stage(days_since_onset) -> np.ndarray:
    d = np.asarray(days_since_onset)
    return np.select([d < 7, d < 21, d < 45], [1, 2, 3], default=4)


def ground_truth_family(df: pd.DataFrame) -> pd.Series:
    """Event family per row. The simulator overwrites `failure_mode` with
    EXPECTED_EMPTY on weekends, but a drifting table keeps drifting through
    them, so weekend rows after a drift onset still belong to the drift."""
    fam = df["failure_mode"].copy()
    drift = fam == "GRADUAL_DRIFT"
    if drift.any():
        onset = df[drift].groupby("table_name")["day_idx"].min()
        t_onset = df["table_name"].map(onset)
        weekend_in_drift = (fam == "EXPECTED_EMPTY") & t_onset.notna() & (df["day_idx"] >= t_onset)
        fam = fam.mask(weekend_in_drift, "GRADUAL_DRIFT")
    return fam.rename("event_family")


def ground_truth_severity(df: pd.DataFrame) -> pd.Series:
    fam = ground_truth_family(df)
    mode = fam.values
    base = np.array([BASE.get(m, 0) for m in mode], dtype=int)

    drift = mode == "GRADUAL_DRIFT"
    if drift.any():
        onset = df[drift].groupby("table_name")["day_idx"].min()
        days = df.loc[drift, "day_idx"].values - df.loc[drift, "table_name"].map(onset).values
        base[drift] = drift_stage(days)

    crit = df["dag_criticality"].values if "dag_criticality" in df else np.full(len(df), 2)
    adj = np.array([CRIT_ADJUST.get(int(c), 0) for c in crit])
    sev = np.where(base > 0, np.clip(base + adj, 1, 4), 0)
    return pd.Series(sev, index=df.index, name="severity_label")
