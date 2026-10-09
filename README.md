# Data Freshness Auditor — Project README

![ci](https://github.com/pro-nine/Data-freshness-auditor-/actions/workflows/ci.yml/badge.svg)

## Architecture
Self-calibrating, DAG-aware Data Freshness Intelligence System.

## Run Order
```
1. python3 data/simulator.py          # Generate 3960 pipeline run records
2. python3 features/engineer.py       # Build 60-feature matrix
3. python3 models/train.py            # Train + evaluate + visualise
4. python3 models/honest_eval.py      # Evaluation-integrity report (outputs/honest_eval.md)
```

## Key Results (original run)
- Stage 1 AUC (anomaly detection):  0.9984
- Alert Precision (WARN+):          96.4%
- Ordinal MAE (severity scale):     0.42
- Weighted FN cost (normalised):    0.24
- Alert Storm Rate:                 0.0 (see caveat 2 below)

## Evaluation Integrity
The numbers above are reproducible, but they need context. These caveats were
found by running the pipeline, and each one is pinned by a test in `tests/`.

1. **The label is a rule over the model's own inputs.** `severity_label` is a
   threshold rule over `cusum_drift_score`, `latency_zscore`, `rows_zscore` and
   `diag_state_code`, and those columns are also model features. In a
   reproduction with scikit-learn's HistGradientBoosting, 12 rule-input columns
   alone reach Stage 1 AUC ~1.00, and removing them leaves ~0.88 (F1 0.69 at a
   0.5 threshold). The headline AUC mostly measures rule recovery.
2. **The test window has one injected failure type.** The 70/15/15 split puts
   days 153-179 in the test set. It contains gradual drift and expected-empty
   weekends only. Infrastructure cascades (days 30, 95, 142), the silent
   business failure (days 55-62) and phantom recovery (days 110-111) are all
   earlier. An alert-storm rate of 0.0 is therefore not evidence of suppression,
   because the test window has no cascade to suppress.
3. **Residual-CUSUM misses the injected drift.** CUSUM runs on the STL residual.
   For `raw_sessions` (drift from day 60) the STL trend rises from about 60 to
   about 347 minutes while the residual mean stays near zero, and the first
   residual-CUSUM alert is on day 143. `features/drift.py` flags the same drift
   from the trend on day 71, with no other table flagged.
4. **The state machine cannot see phantom recovery.** Retried runs with normal
   row counts are labelled HEALTHY.

Caveats 3 and 4 are strict expected-failure tests, so CI will flag the moment a
fix makes them pass. `python3 models/honest_eval.py` regenerates the ablation and
split-coverage tables with LightGBM.

## Project Structure
```
freshness_auditor/
├── data/
│   ├── simulator.py          # 5-failure-mode pipeline ecosystem generator
│   └── feature_matrix.csv    # Generated feature matrix (3960 x 60)
├── features/
│   ├── engineer.py           # STL + CUSUM + 3-bit state + DAG features
│   └── drift.py              # Trend-based drift detection (offline audit rule)
├── models/
│   ├── cascading_classifier.py  # Two-stage: binary anomaly + ordinal severity
│   ├── evaluation.py            # Business metrics suite
│   ├── honest_eval.py           # Split coverage + rule-leakage ablation
│   └── train.py                 # Full training runner
├── tests/                    # Simulator, feature, drift and evaluation tests
├── eda/
│   └── eda_dashboard.png     # 8-panel EDA
└── outputs/
    ├── phase3_results.png    # Model results dashboard
    └── test_predictions.csv  # Per-row predictions
```

## The Architecture in One Sentence
CUSUM control charts on STL residuals detect sudden shifts, a trend-based rule
detects gradual drift, a two-stage cascading classifier (binary anomaly -> ordinal
severity) quantifies impact, and DAG traversal resolves root cause to a single
upstream failure point.

## Known Limits
- All data is simulated: 22 tables, 180 days, 3,960 runs, one drift event.
- The drift rule's threshold has only been checked on that one event.
- STL's trend is a centred smoother, so the drift rule is an offline audit, not a real-time alert.
