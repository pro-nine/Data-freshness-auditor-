# Data Freshness Auditor

![ci](https://github.com/pro-nine/Data-freshness-auditor-/actions/workflows/ci.yml/badge.svg)

Self-calibrating, DAG-aware data freshness auditing on a simulated 22-table, 180-day pipeline ecosystem with five injected failure modes.

## Run Order
```
python3 data/simulator.py        # 3,960 pipeline run records
python3 features/engineer.py     # causal feature matrix + ground-truth labels
python3 models/train.py          # temporal and leave-one-family-out evaluation
python3 models/honest_eval.py    # which failure families each split contains
```

## What v2 changed, and why
v1 reported Stage 1 AUC 0.998 and alert precision 96.4%. Running it showed that those numbers mostly measured how well a model recovers a rule:

| Finding in v1 | Evidence | v2 change |
|---|---|---|
| `severity_label` was a threshold rule over columns the model also received | the 12 rule-input columns alone gave AUC ~1.00; removing them left ~0.88 | labels come from the injected events (`features/labels.py`); the old rule is kept as `rule_severity`, a baseline to beat |
| CUSUM ran on the STL residual, which hides gradual drift | on `raw_sessions` the trend rose 60 to 347 min, the residual mean stayed near 0, first alert on day 143 for a drift that began on day 60 | causal EWMA chart and clipped CUSUM on a weekday-adjusted series, calibrated on a fixed reference window |
| STL and CUSUM calibration used future data | STL fit the whole series; CUSUM calibrated on the first 60% of it | all detectors are causal |
| The test window held one failure family | days 153-179 contain gradual drift only, so "alert storm rate 0.0" had no cascade to suppress | leave-one-family-out evaluation, plus `models/honest_eval.py` |
| The Stage 1 threshold was tuned on training data | `BinaryAnomalyDetector.fit` used training probabilities | threshold and isotonic calibration use validation data |
| Missing severity classes crashed Stage 2 | the ordinal ranker needs a positive for every threshold | thresholds with no positives become constants |

## Evaluation protocol
Protocol A is the temporal split (train days 0-125, validation 126-152, test 153-179). Its test window holds only drift, so it is reported but not used to claim generality.

Protocol B is leave-one-failure-family-out. For each family, every table that suffered it is removed from training, and the model is scored on those tables only. Detectors are compared on identical rows: the pure learned cascade, the hybrid cascade with detector floors, the v1 rule, EWMA, CUSUM and the 3-bit state machine.

Results from the scikit-learn backend (`python models/train.py` regenerates them with LightGBM and writes `outputs/results.md`):

| Held-out family | Detector | Row recall | Lag (days) | False alarms / 100 normal rows |
|---|---|---|---|---|
| Infrastructure cascade | pure ML cascade | 1.00 | 0 | 0.0 |
| Infrastructure cascade | v1 rule | 1.00 | 0 | 3.95 |
| Silent business failure | pure ML cascade | 0.00 | - | 0.0 |
| Silent business failure | hybrid | 1.00 | 0 | 0.0 |
| Silent business failure | v1 rule | 1.00 | 0 | 3.49 |
| Gradual drift | pure ML cascade | 0.28 | 1 | 15.0 |
| Gradual drift | hybrid | 0.90 | 1 | 15.0 |
| Gradual drift | EWMA | 0.87 | 16 | 0.0 |
| Gradual drift | CUSUM | 0.94 | 7 | 0.0 |
| Gradual drift | v1 rule | 0.95 | 4 | 3.33 |
| Phantom recovery | pure ML cascade | 0.00 | - | 0.0 |
| Phantom recovery | hybrid | 1.00 | 0 | 0.0 |

Reading the table:
- A learned model cannot detect a failure family it never saw. The pure cascade misses silent failures and phantom recoveries on unseen tables.
- Detector floors fix that: a failed or silent 3-bit state raises severity to DEGRADED, an EWMA alert to WARN, and a retried run with abnormal latency to WARN.
- On a held-out table the learned model also raises false alarms (15 per 100 normal rows on `raw_sessions`), where the plain v1 rule raises 3.3. The hybrid inherits them.
- On full data, EWMA alerts on `raw_sessions` only (day 76, 16 days after onset) with no alert on the other 21 tables. CUSUM alerts on day 67 but also fires on normal rows of 8 tables.

## Known limits
- All data is simulated: 22 tables, 180 days, 3,960 runs, one drift event and four short events. Event counts are small (3 cascade tables, 1 silent, 1 drift, 1 phantom), so treat recall figures as illustrations.
- The first 28 days of each table are assumed healthy and used as the reference window.
- The scikit-learn backend produced the numbers above. The LightGBM and SHAP code paths were not run when these results were written.
- The simulator does not propagate cascades to downstream tables, although its docstring says it does, so downstream alert suppression is not evaluated.
- Table identity features (`dag_*`, `layer`) let a model memorise tables in the temporal split; Protocol B removes that shortcut.

## Structure
```
data/simulator.py            pipeline ecosystem generator (5 failure modes)
features/engineer.py         causal baseline, EWMA/CUSUM, 3-bit state machine, DAG and upstream features
features/labels.py           ground-truth severity from injected events
features/drift.py            offline STL-trend drift audit (include_stl_audit=True)
models/cascading_classifier.py   two-stage cascade, validation calibration, detector floors
models/evaluation.py         event detection, lag from true onset, false-alarm burden, calibration
models/train.py              Protocol A and B, figure and tables in outputs/
models/honest_eval.py        failure-family coverage of each split
tests/                       simulator, features, labels, models and evaluation tests
```
