"""
train.py (v2): leak-free labels, temporal protocol, leave-one-family-out protocol
=================================================================================
Protocol A  temporal 70/15/15 split (train days 0-125, val 126-152, test 153-179).
            The test window holds only gradual drift, so it is reported but not
            used to claim generality.
Protocol B  leave-one-failure-family-out. For each family, every table that
            suffered it is removed from training. The model is then scored on
            those tables only, on data it has never seen, positives and
            negatives alike. Detectors compared on identical rows:
            cascade_ml, hybrid (ML + detector floors), rule_baseline (the v1
            labelling rule), ewma, cusum, state_machine.
Run:  python models/train.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from models.cascading_classifier import SEVERITY_NAMES, CascadingFreshnessClassifier  # noqa: E402
from models.evaluation import (EVENT_MODES, FreshnessEvaluationSuite, compare_detectors,  # noqa: E402
                               event_onsets)

DROP_COLS = [
    "run_id", "table_name", "pipeline_name", "source_system", "scheduled_at", "started_at",
    "completed_at", "execution_status", "failure_mode", "event_family", "severity_label", "severity_name",
    "rule_severity", "diag_state_label", "diag_owner",
    "latency_minutes", "rows_read", "rows_written",
    "day_idx",
]
TRAIN_END, VAL_END = 125, 152


def select_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in DROP_COLS
            and df[c].dtype in [np.float64, np.int64, float, int]
            and df[c].notna().sum() > len(df) * 0.3]


def temporal_split(df: pd.DataFrame):
    return (df[df["day_idx"] <= TRAIN_END].copy(),
            df[(df["day_idx"] > TRAIN_END) & (df["day_idx"] <= VAL_END)].copy(),
            df[df["day_idx"] > VAL_END].copy())


def fill_with_train_medians(cols, train, *others):
    med = train[cols].median()
    return [d.assign(**{c: d[c].fillna(med[c]) for c in cols}) for d in (train, *others)]


def fit_cascade(train, val, cols, backend="auto"):
    train, val = fill_with_train_medians(cols, train, val)
    model = CascadingFreshnessClassifier(cols, backend=backend).fit(
        train, train["severity_label"], val, val["severity_label"])
    return model


def detector_alerts(df: pd.DataFrame, preds: pd.DataFrame) -> dict:
    return {
        "cascade_ml": preds["severity_ml_only"].values >= 2,
        "hybrid": preds["severity_pred"].values >= 2,
        "rule_baseline": df["rule_severity"].values >= 2,
        "ewma": df["ewma_alert"].values > 0,
        "cusum": df["cusum_alert_up"].values > 0,
        "state_machine": df["diag_state_code"].isin([1, 4, 5, 6]).values,
    }


def leave_one_family_out(df: pd.DataFrame, cols: list, backend="auto") -> pd.DataFrame:
    onsets = event_onsets(df)
    out = []
    for mode in EVENT_MODES:
        tables = set(df.loc[df["event_family"] == mode, "table_name"])
        held = df[df["table_name"].isin(tables)].reset_index(drop=True)
        rest = df[~df["table_name"].isin(tables)]
        cutoff = rest["day_idx"].quantile(0.70)
        train, val = rest[rest["day_idx"] <= cutoff], rest[rest["day_idx"] > cutoff]
        model = fit_cascade(train, val, cols, backend)
        held_f = fill_with_train_medians(cols, train, held)[1]
        preds = model.predict(held_f)
        preds["severity_ml_only"] = model.predict(held_f, floors=False)["severity_pred"].values
        table = compare_detectors(held, detector_alerts(held, preds), onsets)
        table.insert(0, "held_out_tables", ",".join(sorted(tables)))
        out.append(table[table["family"] == mode])
    return pd.concat(out, ignore_index=True)


def temporal_protocol(df: pd.DataFrame, cols: list, backend="auto"):
    train, val, test = temporal_split(df)
    model = fit_cascade(train, val, cols, backend)
    test_f = fill_with_train_medians(cols, train, test)[1].reset_index(drop=True)
    preds = model.predict(test_f)
    preds["severity_ml_only"] = model.predict(test_f, floors=False)["severity_pred"].values
    report = FreshnessEvaluationSuite(test_f, preds, event_onsets(df)).full_report()
    return model, test_f, preds, report


def _md(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in frame.iterrows()]
    return "\n".join(lines)


def load_feature_matrix() -> pd.DataFrame:
    path = os.path.join(ROOT, "data", "feature_matrix.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    from data.simulator import PipelineEcosystemSimulator
    from features.engineer import FreshnessFeatureEngineer
    raw, dag = PipelineEcosystemSimulator(n_days=180, seed=42).generate()
    return FreshnessFeatureEngineer(raw, dag).build()


def save_figure(path, lofo, report, test_f, preds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    pivot = lofo.pivot(index="family", columns="detector", values="row_recall")
    pivot.plot.bar(ax=ax[0], width=0.8)
    ax[0].set_title("Held-out tables: share of event rows alerted")
    ax[0].set_ylim(0, 1.05)
    ax[0].tick_params(axis="x", rotation=25)
    fp = lofo.groupby("detector")["false_alarms_per_100_normal"].mean().sort_values()
    ax[1].barh(fp.index, fp.values, color="#E84B3A")
    ax[1].set_title("False alarms per 100 normal rows (mean over folds)")
    labels = list(range(5))
    cm = confusion_matrix(test_f["severity_label"], preds["severity_pred"], labels=labels)
    ax[2].imshow(cm, cmap="Blues")
    for i in labels:
        for j in labels:
            ax[2].text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    ax[2].set_xticks(labels, SEVERITY_NAMES, rotation=30)
    ax[2].set_yticks(labels, SEVERITY_NAMES)
    ax[2].set_title("Temporal test window (drift only)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    df = load_feature_matrix()
    df["severity_label"] = df["severity_label"].astype(int)
    cols = select_feature_cols(df)
    print(f"{len(df):,} rows, {len(cols)} features, event rows: {(df['severity_label'] > 0).sum()}")

    model, test_f, preds, report = temporal_protocol(df, cols)
    print(f"backend: {model.backend_}")
    lofo = leave_one_family_out(df, cols)

    out = os.path.join(ROOT, "outputs")
    lofo.to_csv(os.path.join(out, "lofo_results.csv"), index=False)
    pd.concat([test_f[["table_name", "day_idx", "event_family", "severity_label"]], preds], axis=1) \
        .to_csv(os.path.join(out, "test_predictions.csv"), index=False)
    with open(os.path.join(out, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# Results (backend: {model.backend_})\n\n## Protocol B: leave-one-family-out\n\n")
        fh.write(_md(lofo) + "\n\n## Protocol A: temporal test window (drift only)\n\n")
        fh.write(f"standard: {report['standard']}\n\nordinal MAE: {report['ordinal_mae']}\n\n")
        fh.write(f"alerts: {report['alerts']}\n\ncalibration: {report['calibration']}\n\n")
        fh.write(_md(report["detection"]) + "\n")
    save_figure(os.path.join(out, "phase3_results.png"), lofo, report, test_f, preds)
    print(lofo.to_string(index=False))
    print(report["standard"], report["alerts"], report["calibration"])


if __name__ == "__main__":
    main()
