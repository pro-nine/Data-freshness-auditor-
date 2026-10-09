"""
Evaluation integrity checks
===========================
Three questions the headline metrics in the README cannot answer on their own:

1. Which failure modes actually appear in the test window?
2. How much of Stage 1's AUC comes from re-learning the labelling rule?
3. What is left when the rule's own inputs are removed?

``severity_label`` is created in features/engineer.py as a threshold rule over
``cusum_drift_score``, ``latency_zscore``, ``rows_zscore`` and
``diag_state_code``. The models then receive those same columns as inputs, so
a high AUC partly measures how well a tree ensemble can recover a rule.

Run:  python models/honest_eval.py   (needs data/feature_matrix.csv)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

FAILURE_MODES = ["INFRA_CASCADE", "SILENT_BUSINESS_FAILURE", "GRADUAL_DRIFT", "PHANTOM_RECOVERY"]

DROP_COLS = [
    "run_id", "table_name", "pipeline_name", "source_system", "scheduled_at",
    "started_at", "completed_at", "execution_status", "failure_mode",
    "severity_label", "severity_name", "diag_state_label", "diag_owner",
    "latency_minutes", "rows_read", "rows_written",
]

# Columns that feed the severity rule in features/engineer.py, plus the
# columns derived from the same quantities.
RULE_INPUT_COLS = [
    "cusum_drift_score", "cusum_s_pos", "cusum_s_neg", "cusum_alert_up", "cusum_alert_dn",
    "latency_zscore", "rows_zscore",
    "diag_state_code", "diag_exec_ok", "diag_read_ok", "diag_write_ok", "row_write_ratio",
]


def select_feature_cols(df: pd.DataFrame) -> list:
    """Same selection as models/train.py."""
    return [
        c for c in df.columns
        if c not in DROP_COLS
        and df[c].dtype in [np.float64, np.int64, float, int]
        and df[c].notna().sum() > len(df) * 0.3
    ]


def temporal_split(df: pd.DataFrame, train: float = 0.70, val: float = 0.85):
    """Same row-based 70/15/15 time-ordered split as models/train.py."""
    df = df.sort_values("day_idx").reset_index(drop=True)
    n = len(df)
    t1, t2 = int(n * train), int(n * val)
    return df.iloc[:t1].copy(), df.iloc[t1:t2].copy(), df.iloc[t2:].copy()


def split_failure_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Rows per failure mode in each split, plus the split's day range."""
    parts = dict(zip(["train", "val", "test"], temporal_split(df)))
    rows = []
    for name, part in parts.items():
        counts = part["failure_mode"].value_counts()
        rec = {"split": name, "first_day": int(part["day_idx"].min()),
               "last_day": int(part["day_idx"].max())}
        rec.update({m: int(counts.get(m, 0)) for m in FAILURE_MODES})
        rows.append(rec)
    return pd.DataFrame(rows)


def uncovered_failure_modes(df: pd.DataFrame, split: str = "test") -> set:
    """Failure modes with zero rows in the chosen split."""
    cov = split_failure_coverage(df).set_index("split").loc[split]
    return {m for m in FAILURE_MODES if cov[m] == 0}


def _default_model():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1, random_state=42)


def leakage_ablation(df: pd.DataFrame, model_factory=None, feature_cols=None) -> pd.DataFrame:
    """Stage-1 anomaly AUC/F1 with all features, without rule inputs, and rule inputs only."""
    model_factory = model_factory or _default_model
    df = df.copy()
    df["severity_label"] = pd.to_numeric(df["severity_label"], errors="coerce").fillna(0).astype(int)
    feature_cols = feature_cols or select_feature_cols(df)
    for c in feature_cols:
        df[c] = df[c].fillna(df[c].median())

    train, _, test = temporal_split(df)
    y_tr = (train["severity_label"] > 0).astype(int)
    y_te = (test["severity_label"] > 0).astype(int)

    variants = {
        "all_features": feature_cols,
        "without_rule_inputs": [c for c in feature_cols if c not in RULE_INPUT_COLS],
        "rule_inputs_only": [c for c in feature_cols if c in RULE_INPUT_COLS],
    }
    rows = []
    for name, cols in variants.items():
        model = model_factory().fit(train[cols], y_tr)
        prob = model.predict_proba(test[cols])[:, 1]
        rows.append({
            "variant": name,
            "n_features": len(cols),
            "auc": round(float(roc_auc_score(y_te, prob)), 4),
            "f1_at_0.5": round(float(f1_score(y_te, prob >= 0.5)), 4),
        })
    return pd.DataFrame(rows)


def _md(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    df = pd.read_csv(os.path.join(root, "data", "feature_matrix.csv"))

    coverage = split_failure_coverage(df)
    ablation = leakage_ablation(df)
    missing = sorted(uncovered_failure_modes(df))

    os.makedirs(os.path.join(root, "outputs"), exist_ok=True)
    path = os.path.join(root, "outputs", "honest_eval.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Evaluation integrity report\n\n## Failure modes per split\n\n")
        fh.write(_md(coverage) + "\n\n")
        fh.write(f"Failure modes absent from the test window: {', '.join(missing) or 'none'}\n\n")
        fh.write("## Stage 1 leakage ablation\n\n" + _md(ablation) + "\n")
    print(coverage.to_string(index=False))
    print(ablation.to_string(index=False))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
