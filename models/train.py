"""
train.py — Full Phase 3 Training Runner
Runs: data load → feature selection → temporal split →
      cascading model training → evaluation → visualisations
"""
import sys, os, warnings
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap

from models.cascading_classifier import CascadingFreshnessClassifier, SEVERITY_NAMES
from models.evaluation import FreshnessEvaluationSuite

os.makedirs(os.path.join(_ROOT, "outputs"), exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. LOAD FEATURE MATRIX
# ─────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(_ROOT, "data", "feature_matrix.csv"))
df["severity_label"] = pd.to_numeric(df["severity_label"], errors="coerce").fillna(0).astype(int)
print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
print(f"Severity distribution:\n{df['severity_label'].value_counts().sort_index().to_string()}\n")

# ─────────────────────────────────────────────────────────────
# 2. FEATURE SELECTION
# Drop metadata/leakage columns, keep engineered features
# ─────────────────────────────────────────────────────────────
DROP_COLS = [
    "run_id","table_name","pipeline_name","source_system","scheduled_at",
    "started_at","completed_at","execution_status","failure_mode",
    "severity_label","severity_name","diag_state_label","diag_owner",
    # raw targets that would leak
    "latency_minutes","rows_read","rows_written",
]
FEATURE_COLS = [c for c in df.columns
                if c not in DROP_COLS
                and df[c].dtype in [np.float64, np.int64, float, int]
                and df[c].notna().sum() > len(df) * 0.3]

print(f"Feature columns selected: {len(FEATURE_COLS)}")

# Fill remaining NaNs with column medians (safe for tree models)
for col in FEATURE_COLS:
    med = df[col].median()
    df[col] = df[col].fillna(med)

# ─────────────────────────────────────────────────────────────
# 3. TEMPORAL TRAIN / VAL / TEST SPLIT (no leakage)
# 70% train → 15% val → 15% test, time-ordered
# ─────────────────────────────────────────────────────────────
df = df.sort_values("day_idx").reset_index(drop=True)
n  = len(df)
t1 = int(n * 0.70)
t2 = int(n * 0.85)

train = df.iloc[:t1].copy()
val   = df.iloc[t1:t2].copy()
test  = df.iloc[t2:].copy()

print(f"Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
print(f"Train day range: {train['day_idx'].min()}–{train['day_idx'].max()}")
print(f"Test  day range: {test['day_idx'].min()}–{test['day_idx'].max()}\n")

# ─────────────────────────────────────────────────────────────
# 4. TRAIN CASCADING MODEL
# ─────────────────────────────────────────────────────────────
model = CascadingFreshnessClassifier(feature_cols=FEATURE_COLS)
model.fit(
    X_train=train, y_train=train["severity_label"],
    X_val=val,     y_val=val["severity_label"]
)

# ─────────────────────────────────────────────────────────────
# 5. PREDICT ON TEST SET
# ─────────────────────────────────────────────────────────────
preds = model.predict(test, apply_cusum_override=True)

# Attach metadata back for evaluation
test_eval = test.copy().reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# 6. EVALUATION SUITE
# ─────────────────────────────────────────────────────────────
evaluator = FreshnessEvaluationSuite(test_eval, preds)
report    = evaluator.full_report()

# ─────────────────────────────────────────────────────────────
# 7. SHAP — Stage 1 binary model
# ─────────────────────────────────────────────────────────────
print("\n── SHAP Analysis (Stage 1) ──")
explainer   = shap.TreeExplainer(model.stage1.model)
sample_X    = test[FEATURE_COLS].sample(min(300, len(test)), random_state=42)
shap_values = explainer.shap_values(sample_X)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

shap_df = pd.DataFrame({
    "feature":       FEATURE_COLS,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False)
print(f"\nTop 10 SHAP features (Stage 1):")
print(shap_df.head(10).to_string(index=False))

# Stage 2 ordinal feature importance
s2_imp = model.stage2.get_feature_importance(FEATURE_COLS)
print(f"\nTop 10 ordinal importance features (Stage 2):")
print(s2_imp.head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 8. VISUALISATIONS
# ─────────────────────────────────────────────────────────────
C = {"bg":"#F8F7F3","dark":"#1A1A1A","grid":"#E5E3DE","blue":"#3A7BD5",
     "green":"#1D9E75","red":"#E84B3A","orange":"#F5A623","grey":"#888780",
     "critical":"#D0021B","purple":"#7B2D8B"}

plt.rcParams.update({
    "font.family":"DejaVu Sans","axes.facecolor":C["bg"],
    "figure.facecolor":C["bg"],"axes.edgecolor":C["grid"],
    "axes.grid":True,"grid.color":C["grid"],"grid.linewidth":0.5,
    "text.color":C["dark"],"axes.labelcolor":C["dark"],
    "xtick.color":C["dark"],"ytick.color":C["dark"],
    "axes.spines.top":False,"axes.spines.right":False,
})

fig = plt.figure(figsize=(20, 22))
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.38)

def T(ax, t, sub=""):
    ax.set_title(f"{t}\n{sub}" if sub else t,
                 fontsize=10, fontweight="bold", loc="left", pad=6)

# P1 — Confusion matrix heatmap
ax1 = fig.add_subplot(gs[0, :2])
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_eval["severity_label"], preds["severity_pred"],
                       labels=list(range(5)))
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
im = ax1.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(5):
    for j in range(5):
        ax1.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})",
                 ha="center", va="center", fontsize=8,
                 color="white" if cm_norm[i,j] > 0.5 else C["dark"])
ax1.set_xticks(range(5)); ax1.set_xticklabels(SEVERITY_NAMES, rotation=30, ha="right")
ax1.set_yticks(range(5)); ax1.set_yticklabels(SEVERITY_NAMES)
ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")
plt.colorbar(im, ax=ax1, fraction=0.025, label="Row-normalised rate")
ax1.grid(False)
T(ax1, "Confusion Matrix — Two-Stage Cascading Classifier",
  "Row-normalised. Strong diagonal = correct. Off-diagonal = severity misclassification.")

# P2 — Per-class F1 bar
ax2 = fig.add_subplot(gs[0, 2])
pc = report["per_class"]
colors = [C["green"],C["blue"],C["orange"],C["red"],C["critical"]]
bars = ax2.bar(pc["severity"], pc["f1"], color=colors, width=0.55, zorder=3)
for b, v in zip(bars, pc["f1"]):
    ax2.text(b.get_x()+b.get_width()/2, v+0.01,
             f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
ax2.set_ylim(0, 1.15); ax2.set_ylabel("F1 Score")
T(ax2, "Per-Class F1 Score")

# P3 — SHAP importance Stage 1
ax3 = fig.add_subplot(gs[1, :2])
top10 = shap_df.head(10).sort_values("mean_abs_shap")
label_map = {
    "cusum_drift_score":   "CUSUM drift score",
    "cusum_s_pos":         "CUSUM S⁺ (cumulative)",
    "latency_zscore":      "Latency Z-score",
    "stl_residual":        "STL residual",
    "rows_zscore":         "Row count Z-score",
    "roll7_entropy":       "Rolling entropy (7d)",
    "latency_velocity":    "Latency velocity (Δ/day)",
    "cusum_alert_up":      "CUSUM alert flag",
    "dag_blast_radius":    "DAG blast radius",
    "dag_criticality":     "DAG criticality",
    "hist_mean_21d":       "Historical mean (21d)",
    "diag_state_code":     "Diagnostic state code",
    "row_write_ratio":     "Row write ratio",
    "roll7_kurtosis":      "Rolling kurtosis (7d)",
}
top10["label"] = top10["feature"].map(label_map).fillna(top10["feature"])
ax3.barh(top10["label"], top10["mean_abs_shap"],
          color=C["blue"], height=0.6, zorder=3)
T(ax3, "SHAP Feature Importance — Stage 1 Binary Anomaly Detector",
  "What signals drive the anomaly/no-anomaly decision.")
ax3.set_xlabel("Mean |SHAP| value")
ax3.tick_params(axis="y", labelsize=8.5)

# P4 — Ordinal importance Stage 2
ax4 = fig.add_subplot(gs[1, 2])
s2_top8 = s2_imp.head(8).sort_values("mean_importance")
ax4.barh(s2_top8["feature"].map(label_map).fillna(s2_top8["feature"]),
          s2_top8["mean_importance"], color=C["purple"], height=0.6, zorder=3)
T(ax4, "Ordinal Importance — Stage 2",
  "Aggregated across 4 binary classifiers.")
ax4.tick_params(axis="y", labelsize=8)

# P5 — Weighted FN cost by table
ax5 = fig.add_subplot(gs[2, :2])
tdf = test_eval.copy()
tdf["fn"]         = ((preds["severity_pred"].values == 0) & (tdf["severity_label"] > 0)).astype(int)
tdf["crit_weight"]= tdf["dag_criticality"].map({1:1,2:2,3:4,4:8}).fillna(1)
tdf["fn_cost"]    = tdf["fn"] * tdf["crit_weight"] * np.log1p(tdf["dag_blast_radius"].fillna(0))
fn_by_table = tdf.groupby("table_name")["fn_cost"].sum().sort_values(ascending=False).head(10)
ax5.barh(fn_by_table.index[::-1], fn_by_table.values[::-1],
          color=C["red"], height=0.65, zorder=3)
T(ax5, "Weighted False Negative Cost by Table",
  "Cost = FN count × criticality_weight × log(1 + blast_radius). Higher = more dangerous miss.")
ax5.set_xlabel("Weighted FN Cost")

# P6 — Severity prediction timeline
ax6 = fig.add_subplot(gs[2, 2])
pred_sev_time = test_eval.copy()
pred_sev_time["sev_pred"] = preds["severity_pred"].values
st = pred_sev_time.groupby(["day_idx","sev_pred"]).size().unstack(fill_value=0)
sev_pal = [C["green"],C["blue"],C["orange"],C["red"],C["critical"]]
bot = np.zeros(len(st))
for k, clr in enumerate(sev_pal):
    if k in st.columns:
        v = st[k].values
        ax6.fill_between(st.index, bot, bot+v, color=clr, alpha=0.82,
                          label=SEVERITY_NAMES[k])
        bot += v
ax6.legend(fontsize=7, loc="upper left", framealpha=0.9)
T(ax6, "Predicted Severity Over Time (Test)",
  "Model output across test window.")
ax6.set_xlabel("Day"); ax6.set_ylabel("Count")

# P7 — MTTD by failure mode
ax7 = fig.add_subplot(gs[3, :2])
mttd_detail = report["mttd"].get("detail", pd.DataFrame())
if len(mttd_detail) > 0:
    mode_colors = {"INFRA_CASCADE":C["red"],"SILENT_BUSINESS_FAILURE":C["critical"],
                    "GRADUAL_DRIFT":C["orange"],"PHANTOM_RECOVERY":C["purple"]}
    for _, row in mttd_detail.iterrows():
        if pd.notna(row.get("detection_lag_days")):
            ax7.scatter(row["first_failure_day"], row["detection_lag_days"],
                        color=mode_colors.get(row["failure_mode"], C["grey"]),
                        s=80, zorder=4, alpha=0.85)
    ax7.axhline(report["mttd"].get("mttd_days",0), color=C["dark"],
                ls="--", lw=1.3, label=f"Median MTTD={report['mttd'].get('mttd_days','N/A')}d")
    from matplotlib.lines import Line2D
    legend_els = [Line2D([0],[0],marker='o',color='w',
                          markerfacecolor=c,markersize=9,label=m)
                  for m,c in mode_colors.items()]
    ax7.legend(handles=legend_els, fontsize=7.5, framealpha=0.9)
T(ax7, "Mean Time to Detect — By Failure Mode & Day",
  "Each dot = one detected failure event. Y-axis = lag in days between injection and first WARN+ alert.")
ax7.set_xlabel("Failure onset day (test window)"); ax7.set_ylabel("Detection lag (days)")

# P8 — Business metrics scorecard
ax8 = fig.add_subplot(gs[3, 2])
ax8.axis("off")
metrics_text = [
    ("Stage 1 AUC",        f"{report['stage1_auc']:.4f}"),
    ("Ordinal MAE",         f"{report['ordinal_mae']:.4f}"),
    ("Macro F1",            f"{report['standard']['macro_f1']:.4f}"),
    ("Alert Precision",     f"{report['alert_prec']['alert_precision']:.4f}"),
    ("Alert Recall",        f"{report['alert_prec']['alert_recall']:.4f}"),
    ("FN Cost (norm.)",     f"{report['fn_cost']['normalised_fn_cost']:.4f}"),
    ("MTTD (days)",         f"{report['mttd'].get('mttd_days','N/A')}"),
    ("Detection Rate",      f"{report['mttd'].get('detection_rate','N/A')}"),
    ("Storm Rate",          f"{report['storm']['alert_storm_rate']:.4f}"),
]
y = 0.95
ax8.text(0.05, y, "Business Metrics Scorecard",
          fontsize=11, fontweight="bold", transform=ax8.transAxes)
y -= 0.10
for label, val in metrics_text:
    ax8.text(0.05, y, label, fontsize=9, transform=ax8.transAxes, color=C["grey"])
    ax8.text(0.72, y, val,   fontsize=9, transform=ax8.transAxes,
              fontweight="bold", color=C["dark"])
    y -= 0.09
ax8.set_xlim(0,1); ax8.set_ylim(0,1)

fig.suptitle(
    "Data Freshness Auditor — Phase 3: Modeling & Evaluation\n"
    "Two-Stage Cascading Classifier  ·  Ordinal Decomposition  ·  CUSUM Override",
    fontsize=13, fontweight="bold", y=0.999, color=C["dark"]
)
plt.savefig(os.path.join(_ROOT, "outputs", "phase3_results.png"),
            dpi=150, bbox_inches="tight", facecolor=C["bg"])
print("\n✓ Phase 3 visualisation saved.")

# ── Save predictions CSV ──────────────────────────────────
out = test_eval[["table_name","day_idx","failure_mode",
                  "severity_label","dag_criticality","dag_blast_radius"]].copy()
out = pd.concat([out.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
out.to_csv(os.path.join(_ROOT, "outputs", "test_predictions.csv"), index=False)
print("✓ Predictions saved.")
print("\n=== Phase 3 Complete ===")
