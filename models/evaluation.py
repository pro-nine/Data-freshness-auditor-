"""
FreshnessEvaluationSuite (v2)
=============================
Metrics that match how an on-call engineer would judge the detector.

* Ordinal MAE, per-class metrics and a business-weighted false-negative cost.
* Event detection: for every injected failure event (family x table) the share
  of its rows that raise a WARN+ alert, and the lag in days from the event's
  true onset to the first alert. Onsets come from the full run history, not the
  evaluation slice, so an event that began before the slice is reported as
  "ongoing" instead of getting a lag measured from the slice's first row.
* False-alarm burden: WARN+ alerts per 100 normal table-days.
* Calibration: expected calibration error of Stage 1 probabilities.
* `compare_detectors` scores any set of boolean alert columns the same way, so
  the ML cascade can be compared with the rule baseline, EWMA, CUSUM and the
  3-bit state machine.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_auc_score

from models.cascading_classifier import SEVERITY_NAMES, expected_calibration_error

CRIT_WEIGHTS = {1: 1, 2: 2, 3: 4, 4: 8}
EVENT_MODES = ["INFRA_CASCADE", "SILENT_BUSINESS_FAILURE", "GRADUAL_DRIFT", "PHANTOM_RECOVERY"]


def event_onsets(full: pd.DataFrame) -> dict:
    """First day of each (failure_mode, table) event across the full history."""
    ev = full[full["event_family"].isin(EVENT_MODES)]
    return ev.groupby(["event_family", "table_name"])["day_idx"].min().to_dict()


def compare_detectors(df: pd.DataFrame, alerts: dict, onsets: dict) -> pd.DataFrame:
    """Row recall, events detected, median lag and false alarms per 100 normal rows."""
    d = df.reset_index(drop=True)
    normal = d["event_family"].isin(["NONE", "EXPECTED_EMPTY"]).values
    first_day = int(d["day_idx"].min())
    rows = []
    for name, flag in alerts.items():
        flag = np.asarray(flag).astype(bool)
        fp = 100.0 * flag[normal].mean() if normal.any() else np.nan
        for mode in EVENT_MODES:
            mask = d["event_family"].values == mode
            if not mask.any():
                continue
            lags, detected, n_events = [], 0, 0
            for table, grp in d[mask].groupby("table_name"):
                n_events += 1
                hit_days = grp.loc[flag[grp.index.values], "day_idx"]
                detected += int(len(hit_days) > 0)
                onset = onsets.get((mode, table), int(grp["day_idx"].min()))
                if len(hit_days) and onset >= first_day:
                    lags.append(int(hit_days.min()) - int(onset))
            rows.append({
                "detector": name, "family": mode, "rows": int(mask.sum()),
                "row_recall": round(float(flag[mask].mean()), 3),
                "events_detected": f"{detected}/{n_events}",
                "median_lag_days": float(np.median(lags)) if lags else np.nan,
                "false_alarms_per_100_normal": round(float(fp), 2),
            })
    return pd.DataFrame(rows)


class FreshnessEvaluationSuite:
    def __init__(self, df_test: pd.DataFrame, preds: pd.DataFrame, onsets: dict = None):
        self.df = df_test.reset_index(drop=True)
        self.preds = preds.reset_index(drop=True)
        self.y_true = self.df["severity_label"].values
        self.y_pred = self.preds["severity_pred"].values
        self.onsets = onsets or event_onsets(self.df)

    def standard_metrics(self) -> dict:
        return {
            "accuracy": round(float((self.y_pred == self.y_true).mean()), 4),
            "macro_f1": round(float(f1_score(self.y_true, self.y_pred, average="macro", zero_division=0)), 4),
            "weighted_f1": round(float(f1_score(self.y_true, self.y_pred, average="weighted", zero_division=0)), 4),
        }

    def ordinal_mae(self) -> float:
        return round(float(np.abs(self.y_pred - self.y_true).mean()), 4)

    def weighted_fn_cost(self) -> dict:
        d = self.df.copy()
        d["fn"] = ((self.y_pred == 0) & (self.y_true > 0)).astype(int)
        w = d["dag_criticality"].map(CRIT_WEIGHTS).fillna(1) * np.log1p(d["dag_blast_radius"].fillna(0).clip(lower=1))
        d["fn_cost"] = d["fn"] * w
        worst = float(((self.y_true > 0) * w).sum())
        return {"false_negatives": int(d["fn"].sum()), "raw_fn_cost": round(float(d["fn_cost"].sum()), 2),
                "normalised_fn_cost": round(float(d["fn_cost"].sum() / (worst + 1e-9)), 4)}

    def alert_quality(self) -> dict:
        fired, actual = self.y_pred >= 2, self.y_true >= 2
        normal = self.df["event_family"].isin(["NONE", "EXPECTED_EMPTY"]).values
        tp = int((fired & actual).sum())
        return {
            "alerts_fired": int(fired.sum()),
            "alert_precision": round(tp / max(1, int(fired.sum())), 4),
            "alert_recall": round(tp / max(1, int(actual.sum())), 4),
            "false_alarms_per_100_normal": round(100.0 * float(fired[normal].mean()), 2) if normal.any() else np.nan,
        }

    def detection(self) -> pd.DataFrame:
        return compare_detectors(self.df, {"cascade_ml": self.y_pred >= 2}, self.onsets)

    def calibration(self) -> dict:
        y = (self.y_true > 0).astype(int)
        p = self.preds["anomaly_prob"].values
        try:
            auc = round(float(roc_auc_score(y, p)), 4)
        except ValueError:
            auc = float("nan")
        return {"stage1_auc": auc, "stage1_ece": round(expected_calibration_error(y, p), 4)}

    def per_class_metrics(self) -> pd.DataFrame:
        present = sorted(set(self.y_true) | set(self.y_pred))
        rep = classification_report(self.y_true, self.y_pred, labels=present,
                                    target_names=[SEVERITY_NAMES[i] for i in present],
                                    output_dict=True, zero_division=0)
        return pd.DataFrame([{"severity": SEVERITY_NAMES[i], "precision": round(rep[SEVERITY_NAMES[i]]["precision"], 4),
                              "recall": round(rep[SEVERITY_NAMES[i]]["recall"], 4),
                              "f1": round(rep[SEVERITY_NAMES[i]]["f1-score"], 4),
                              "support": int(rep[SEVERITY_NAMES[i]]["support"])} for i in present])

    def full_report(self) -> dict:
        return {"standard": self.standard_metrics(), "ordinal_mae": self.ordinal_mae(),
                "fn_cost": self.weighted_fn_cost(), "alerts": self.alert_quality(),
                "calibration": self.calibration(), "detection": self.detection(),
                "per_class": self.per_class_metrics()}
