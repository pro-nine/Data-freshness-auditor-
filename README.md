# Data Freshness Auditor — Project README

## Architecture
Self-calibrating, DAG-aware Data Freshness Intelligence System.

## Run Order
```
1. python3 data/simulator.py          # Generate 3960 pipeline run records
2. python3 features/engineer.py       # Build 60-feature matrix
3. python3 models/train.py            # Train + evaluate + visualise
```

## Key Results
- Stage 1 AUC (anomaly detection):  0.9984
- Alert Precision (WARN+):          96.4%
- Ordinal MAE (severity scale):     0.42
- Weighted FN cost (normalised):    0.24
- Alert Storm Rate:                 0.0 (zero false cascade alerts)

## Project Structure
```
freshness_auditor/
├── data/
│   ├── simulator.py          # 5-failure-mode pipeline ecosystem generator
│   └── feature_matrix.csv    # Generated feature matrix (3960 × 60)
├── features/
│   └── engineer.py           # STL + CUSUM + 3-bit state + DAG features
├── models/
│   ├── cascading_classifier.py  # Two-stage: binary anomaly + ordinal severity
│   ├── evaluation.py            # Business metrics suite
│   └── train.py                 # Full training runner
├── eda/
│   └── eda_dashboard.png     # 8-panel EDA
└── outputs/
    ├── phase3_results.png    # Model results dashboard
    └── test_predictions.csv  # Per-row predictions

```

## The Architecture in One Sentence
CUSUM control charts on STL residuals detect gradual drift,
a two-stage cascading classifier (binary anomaly → ordinal severity)
quantifies impact, and DAG traversal resolves root cause to
a single upstream failure point.
