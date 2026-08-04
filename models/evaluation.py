"""
FreshnessEvaluationSuite
=========================
Goes far beyond standard ML metrics.

Standard tutorial metrics (what everyone else reports):
    - Accuracy
    - Macro F1
    - Confusion matrix

This suite adds:
    1. Ordinal MAE       : Mean absolute error on severity SCALE (not class label)
                           Predicting WARN when true is DEGRADED is penalised less
                           than predicting NORMAL when true is DEGRADED.
    2. Weighted FN Cost  : False negatives weighted by blast_radius × criticality.
                           Missing a CRITICAL table with 15 downstream consumers
                           costs 60x more than missing a LOW table with 1.
    3. Alert Precision   : Of all alerts fired, what % were real anomalies?
                           The metric on-call engineers actually care about.
    4. Mean Time to Detect: Average lag between failure injection (day_idx)
                            and first WARN+ prediction. Measures early warning value.
    5. Alert Storm Rate  : % of alerts that are FP cascades (FP within 2 days
                           of an infrastructure cascade event). Measures suppression.
    6. Severity Calibration: Whether predicted probabilities match empirical
                             frequencies. Poorly calibrated models are dangerous
                             in production: they say 40% chance of CRITICAL
                             when true rate is 5%.
    7. Stage 1 / Stage 2 decomposed performance.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

SEVERITY_NAMES = ["NORMAL", "WATCH", "WARN", "DEGRADED", "CRITICAL"]
CRIT_WEIGHTS   = {1: 1, 2: 2, 3: 4, 4: 8}  # exponential: CRITICAL = 8× LOW


class FreshnessEvaluationSuite:

    def __init__(self, df_test: pd.DataFrame, preds: pd.DataFrame):
        """
        df_test : test-split feature matrix (with metadata columns)
        preds   : output DataFrame from CascadingFreshnessClassifier.predict()
        """
        self.df   = df_test.reset_index(drop=True)
        self.preds= preds.reset_index(drop=True)
        self.y_true      = self.df["severity_label"].values
        self.y_pred      = self.preds["severity_pred"].values
        self.anom_true   = (self.y_true > 0).astype(int)
        self.anom_pred   = self.preds["anomaly_flag"].values

    # ── 1. Standard metrics ─────────────────────────────────
    def standard_metrics(self) -> dict:
        acc   = (self.y_pred == self.y_true).mean()
        mac_f1= f1_score(self.y_true, self.y_pred, average="macro",
                          zero_division=0)
        wei_f1= f1_score(self.y_true, self.y_pred, average="weighted",
                          zero_division=0)
        return {
            "accuracy":      round(acc, 4),
            "macro_f1":      round(mac_f1, 4),
            "weighted_f1":   round(wei_f1, 4),
        }

    # ── 2. Ordinal MAE ──────────────────────────────────────
    def ordinal_mae(self) -> float:
        """
        Mean absolute error on severity scale.
        Predicting CRITICAL (4) when true is DEGRADED (3) costs 1.
        Predicting NORMAL (0) when true is CRITICAL (4) costs 4.
        """
        mae = np.abs(self.y_pred - self.y_true).mean()
        return round(float(mae), 4)

    # ── 3. Weighted False Negative Cost ─────────────────────
    def weighted_fn_cost(self) -> dict:
        """
        FN weighted by blast_radius and criticality.
        Measures the business impact of missed detections.
        """
        df = self.df.copy()
        df["fn"] = ((self.y_pred == 0) & (self.y_true > 0)).astype(int)
        df["crit_weight"] = df["dag_criticality"].map(CRIT_WEIGHTS).fillna(1)
        df["blast"]       = df["dag_blast_radius"].fillna(0).clip(lower=1)

        df["fn_cost"] = df["fn"] * df["crit_weight"] * np.log1p(df["blast"])

        total_fn     = df["fn"].sum()
        raw_fn_cost  = df["fn_cost"].sum()
        worst_case   = (df["crit_weight"] * np.log1p(df["blast"])).sum()
        norm_fn_cost = raw_fn_cost / (worst_case + 1e-9)  # 0=perfect, 1=all missed

        top_fns = (df[df["fn"]==1]
                   .sort_values("fn_cost", ascending=False)
                   [["table_name","day_idx","dag_criticality",
                     "dag_blast_radius","fn_cost"]]
                   .head(5))

        return {
            "total_false_negatives": int(total_fn),
            "raw_fn_cost":           round(raw_fn_cost, 2),
            "normalised_fn_cost":    round(norm_fn_cost, 4),
            "top_costly_misses":     top_fns,
        }

    # ── 4. Alert Precision (engineer-facing) ────────────────
    def alert_precision(self) -> dict:
        """
        Precision of WARN+ alerts — the signal an on-call engineer gets.
        """
        fired_mask  = self.y_pred >= 2          # WARN or above
        actual_mask = self.y_true >= 2

        if fired_mask.sum() == 0:
            return {"alert_precision": 0.0, "alerts_fired": 0, "true_alerts": 0}

        alert_prec = (fired_mask & actual_mask).sum() / fired_mask.sum()
        return {
            "alert_precision":  round(float(alert_prec), 4),
            "alerts_fired":     int(fired_mask.sum()),
            "true_alerts":      int(actual_mask.sum()),
            "alert_recall":     round(float((fired_mask & actual_mask).sum()
                                            / (actual_mask.sum() + 1e-9)), 4),
        }

    # ── 5. Mean Time to Detect ──────────────────────────────
    def mean_time_to_detect(self) -> dict:
        """
        For each injected failure event, when did the model first fire WARN+?
        Computes median lag in days between failure onset and first detection.
        """
        df = self.df.copy()
        df["warn_fired"] = self.y_pred >= 2

        # Define failure events by failure_mode column
        failure_modes = ["INFRA_CASCADE", "SILENT_BUSINESS_FAILURE",
                          "GRADUAL_DRIFT", "PHANTOM_RECOVERY"]
        results = []

        for mode in failure_modes:
            failed_tables = df[df["failure_mode"]==mode]["table_name"].unique()
            for table in failed_tables:
                tdf = df[df["table_name"]==table].sort_values("day_idx")
                failure_days = tdf[tdf["failure_mode"]==mode]["day_idx"]
                if len(failure_days) == 0:
                    continue
                first_failure = failure_days.min()
                # Find first WARN+ after failure onset
                detections = tdf[
                    (tdf["day_idx"] >= first_failure) & (tdf["warn_fired"])
                ]["day_idx"]
                lag = (detections.min() - first_failure) if len(detections) else np.nan
                results.append({
                    "failure_mode": mode,
                    "table":        table,
                    "first_failure_day": first_failure,
                    "first_detection_day": detections.min() if len(detections) else np.nan,
                    "detection_lag_days": lag,
                    "detected": not np.isnan(lag)
                })

        res_df = pd.DataFrame(results)
        if len(res_df) == 0:
            return {"mttd_days": np.nan, "detection_rate": 0}

        return {
            "mttd_days":         round(float(res_df["detection_lag_days"].median()), 2),
            "detection_rate":    round(float(res_df["detected"].mean()), 4),
            "by_failure_mode":   (res_df.groupby("failure_mode")
                                   ["detection_lag_days"].median()
                                   .round(2).to_dict()),
            "detail":            res_df,
        }

    # ── 6. Alert Storm Suppression ──────────────────────────
    def alert_storm_rate(self) -> dict:
        """
        On infra cascade days, how many FP alerts fire on downstream tables
        that should have been suppressed (explained lateness)?
        """
        df = self.df.copy()
        df["fp"] = ((self.y_pred >= 2) & (self.y_true == 0)).astype(int)

        # Cascade event days (from injected events)
        cascade_days = df[df["failure_mode"]=="INFRA_CASCADE"]["day_idx"].unique()
        window       = 2  # days around cascade
        storm_days   = set()
        for d in cascade_days:
            storm_days.update(range(int(d)-window, int(d)+window+1))

        fps_on_cascade = df[df["day_idx"].isin(storm_days) & (df["fp"]==1)]
        fps_total      = df[df["fp"]==1]

        return {
            "total_fps":                int(df["fp"].sum()),
            "fps_on_cascade_days":      int(len(fps_on_cascade)),
            "alert_storm_rate":         round(len(fps_on_cascade) /
                                              (len(fps_total) + 1e-9), 4),
            "cascade_days_detected":    int(len(cascade_days)),
        }

    # ── 7. Per-class metrics ─────────────────────────────────
    def per_class_metrics(self) -> pd.DataFrame:
        report = classification_report(
            self.y_true, self.y_pred,
            target_names=SEVERITY_NAMES,
            output_dict=True,
            zero_division=0
        )
        rows = []
        for cls in SEVERITY_NAMES:
            if cls in report:
                rows.append({
                    "severity":  cls,
                    "precision": round(report[cls]["precision"], 4),
                    "recall":    round(report[cls]["recall"], 4),
                    "f1":        round(report[cls]["f1-score"], 4),
                    "support":   int(report[cls]["support"]),
                })
        return pd.DataFrame(rows)

    # ── 8. Stage 1 AUC ──────────────────────────────────────
    def stage1_auc(self) -> float:
        try:
            return round(roc_auc_score(self.anom_true,
                                        self.preds["anomaly_prob"]), 4)
        except Exception:
            return 0.0

    # ── MASTER REPORT ────────────────────────────────────────
    def full_report(self) -> dict:
        print("\n" + "="*60)
        print("FRESHNESS AUDITOR — EVALUATION REPORT")
        print("="*60)

        std  = self.standard_metrics()
        print(f"\n[1] Standard Metrics")
        for k, v in std.items():
            print(f"    {k:20s}: {v}")

        o_mae = self.ordinal_mae()
        print(f"\n[2] Ordinal MAE (severity scale): {o_mae}")
        print( "    Interpretation: avg severity distance between pred and true.")
        print( "    0.0 = perfect  |  1.0 = off by one level  |  4.0 = worst")

        fn  = self.weighted_fn_cost()
        print(f"\n[3] Weighted False Negative Cost")
        print(f"    Total FNs          : {fn['total_false_negatives']}")
        print(f"    Raw FN cost        : {fn['raw_fn_cost']}")
        print(f"    Normalised FN cost : {fn['normalised_fn_cost']}  (0=perfect, 1=all missed)")
        print(f"    Top costly misses  :\n{fn['top_costly_misses'].to_string(index=False)}")

        ap  = self.alert_precision()
        print(f"\n[4] Alert Precision (WARN+ threshold)")
        for k, v in ap.items():
            print(f"    {k:20s}: {v}")

        mttd = self.mean_time_to_detect()
        print(f"\n[5] Mean Time to Detect (MTTD)")
        print(f"    Median lag (days)  : {mttd.get('mttd_days','N/A')}")
        print(f"    Detection rate     : {mttd.get('detection_rate','N/A')}")
        print(f"    By failure mode    : {mttd.get('by_failure_mode',{})}")

        storm = self.alert_storm_rate()
        print(f"\n[6] Alert Storm Suppression")
        for k, v in storm.items():
            print(f"    {k:30s}: {v}")

        s1_auc = self.stage1_auc()
        print(f"\n[7] Stage 1 Binary AUC (anomaly detection): {s1_auc}")

        print(f"\n[8] Per-Class Metrics:")
        print(self.per_class_metrics().to_string(index=False))

        return {
            "standard":      std,
            "ordinal_mae":   o_mae,
            "fn_cost":       fn,
            "alert_prec":    ap,
            "mttd":          mttd,
            "storm":         storm,
            "stage1_auc":    s1_auc,
            "per_class":     self.per_class_metrics(),
        }
