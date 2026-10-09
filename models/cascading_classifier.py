"""
CascadingFreshnessClassifier (v2)
=================================
Stage 1  binary anomaly detector: "is this run wrong?"
Stage 2  ordinal severity ranker (Frank & Hall decomposition): "how bad?"
Floors   detectors that need no failure examples set a minimum severity:
         3-bit state (failed or silent) -> DEGRADED, EWMA alert -> WARN,
         retried run with abnormal latency -> WARN. `predict(floors=False)`
         returns the pure learned cascade.

Changes from v1
* The Stage 1 threshold is chosen on the validation set. v1 chose it on the
  training set, where the model has already seen the labels.
* Stage 1 probabilities are calibrated on validation data with isotonic
  regression, so `anomaly_prob` can be read as a frequency.
* Class imbalance uses sample weights, which both backends support.
* Stage 2 skips thresholds that have no positive examples (v1 crashed when a
  severity level was missing from the training rows).
* v1's CUSUM override flagged rows that were already at WARN as overridden;
  `floor_applied` marks only rows whose severity was actually raised.
* LightGBM is the default backend. If it is not installed the classifier falls
  back to scikit-learn's HistGradientBoosting, and reports which one it used.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

SEVERITY_NAMES = ["NORMAL", "WATCH", "WARN", "DEGRADED", "CRITICAL"]
N_CLASSES = len(SEVERITY_NAMES)


def make_booster(n_estimators=400, learning_rate=0.04, num_leaves=47, min_child_samples=20,
                 reg_lambda=0.2, random_state=42, backend="auto"):
    """Return (model, backend_name)."""
    if backend in ("auto", "lightgbm"):
        try:
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                objective="binary", n_estimators=n_estimators, learning_rate=learning_rate,
                num_leaves=num_leaves, min_child_samples=min_child_samples, feature_fraction=0.75,
                bagging_fraction=0.75, bagging_freq=5, reg_alpha=0.1, reg_lambda=reg_lambda,
                verbose=-1, n_jobs=-1, random_state=random_state), "lightgbm"
        except ImportError:
            if backend == "lightgbm":
                raise
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=n_estimators, learning_rate=learning_rate, max_leaf_nodes=num_leaves,
        min_samples_leaf=min_child_samples, l2_regularization=reg_lambda,
        random_state=random_state), "sklearn"


def fit_booster(model, kind, X, y, X_val=None, y_val=None, pos_weight=1.0):
    w = np.where(np.asarray(y) == 1, pos_weight, 1.0)
    if kind == "lightgbm" and X_val is not None and len(np.unique(y_val)) > 1:
        import lightgbm as lgb
        model.fit(X, y, sample_weight=w, eval_set=[(X_val, y_val)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(period=-1)])
    else:
        model.fit(X, y, sample_weight=w)
    return model


def expected_calibration_error(y_true, prob, bins=10) -> float:
    y_true, prob = np.asarray(y_true, float), np.asarray(prob, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(prob[m].mean() - y_true[m].mean())
    return float(ece)


class BinaryAnomalyDetector:
    def __init__(self, backend="auto"):
        self.backend = backend
        self.model, self.kind = make_booster(500, 0.03, 63, 25, 0.25, backend=backend)
        self.calibrator_ = None
        self.threshold_ = 0.5

    def _raw(self, X):
        return self.model.predict_proba(X)[:, 1]

    def fit(self, X, y, X_val, y_val, beta=0.5):
        pos = max(float(np.mean(y)), 1e-6)
        fit_booster(self.model, self.kind, X, y, X_val, y_val, pos_weight=min((1 - pos) / pos, 50.0))
        raw_val = self._raw(X_val)
        if len(np.unique(y_val)) > 1:
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip").fit(raw_val, y_val)
            prob = self.calibrator_.predict(raw_val)
            best_f, best_t = -1.0, 0.5
            for t in np.arange(0.1, 0.91, 0.02):
                p = prob >= t
                tp, fp, fn = (p & (y_val == 1)).sum(), (p & (y_val == 0)).sum(), (~p & (y_val == 1)).sum()
                prec, rec = tp / (tp + fp + 1e-9), tp / (tp + fn + 1e-9)
                f = (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec + 1e-9)
                if f > best_f:
                    best_f, best_t = f, t
            self.threshold_ = float(best_t)
        return self

    def predict_proba(self, X):
        raw = self._raw(X)
        return self.calibrator_.predict(raw) if self.calibrator_ is not None else raw

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold_).astype(int)


class OrdinalDecompositionRanker:
    """K-1 binary models for P(Y > k). Thresholds without positives become constants."""

    def __init__(self, backend="auto"):
        self.backend = backend
        self.models_, self.kinds_, self.const_ = [], [], []

    def fit(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y)
        self.models_, self.kinds_, self.const_ = [], [], []
        for k in range(N_CLASSES - 1):
            yb = (y > k).astype(int)
            if yb.sum() == 0 or yb.sum() == len(yb):
                self.models_.append(None)
                self.kinds_.append(None)
                self.const_.append(float(yb.mean()) if len(yb) else 0.0)
                continue
            model, kind = make_booster(backend=self.backend)
            yv = None if y_val is None else (np.asarray(y_val) > k).astype(int)
            pos = yb.mean()
            fit_booster(model, kind, X, yb, X_val, yv, pos_weight=min((1 - pos) / (pos + 1e-9), 20.0))
            self.models_.append(model)
            self.kinds_.append(kind)
            self.const_.append(None)
        return self

    def predict_proba(self, X):
        exceed = np.zeros((len(X), N_CLASSES - 1))
        for k, m in enumerate(self.models_):
            exceed[:, k] = self.const_[k] if m is None else m.predict_proba(X)[:, 1]
        exceed = np.minimum.accumulate(exceed, axis=1)
        p = np.zeros((len(X), N_CLASSES))
        p[:, 0] = 1 - exceed[:, 0]
        for k in range(1, N_CLASSES - 1):
            p[:, k] = exceed[:, k - 1] - exceed[:, k]
        p[:, -1] = exceed[:, -1]
        p = np.clip(p, 0, 1)
        return p / (p.sum(axis=1, keepdims=True) + 1e-9)

    def feature_importance(self, names):
        parts = [pd.DataFrame({"feature": names, "importance": m.feature_importances_})
                 for m in self.models_ if m is not None and hasattr(m, "feature_importances_")]
        if not parts:
            return pd.DataFrame({"feature": names, "mean_importance": 0.0})
        return (pd.concat(parts).groupby("feature")["importance"].mean().sort_values(ascending=False)
                .reset_index().rename(columns={"importance": "mean_importance"}))


class CascadingFreshnessClassifier:
    def __init__(self, feature_cols, backend="auto"):
        self.feature_cols = list(feature_cols)
        self.stage1 = BinaryAnomalyDetector(backend)
        self.stage2 = OrdinalDecompositionRanker(backend)
        self.backend_ = self.stage1.kind
        self.is_fitted_ = False

    def fit(self, X_train, y_train, X_val, y_val):
        yt, yv = np.asarray(y_train), np.asarray(y_val)
        self.stage1.fit(X_train[self.feature_cols], (yt > 0).astype(int),
                        X_val[self.feature_cols], (yv > 0).astype(int))
        m_tr, m_va = yt > 0, yv > 0
        self.stage2.fit(X_train[self.feature_cols][m_tr], yt[m_tr],
                        X_val[self.feature_cols][m_va], yv[m_va])
        self.is_fitted_ = True
        return self

    @staticmethod
    def detector_floor(X: pd.DataFrame) -> np.ndarray:
        """Severity floors from detectors that need no failure examples to work."""
        n = len(X)
        floor = np.zeros(n, dtype=int)
        state = X["diag_state_code"].values if "diag_state_code" in X else np.zeros(n)
        floor = np.where(np.isin(state, [1, 4, 5, 6]), 3, floor)
        if "ewma_alert" in X:
            floor = np.maximum(floor, np.where(X["ewma_alert"].fillna(0).values > 0, 2, 0))
        if {"was_retried", "latency_zscore"} <= set(X.columns):
            retry = (X["was_retried"].values > 0) & (X["latency_zscore"].fillna(0).values > 3.0)
            floor = np.maximum(floor, np.where(retry, 2, 0))
        return floor

    def predict(self, X, floors=True) -> pd.DataFrame:
        """floors=False gives the pure learned cascade."""
        assert self.is_fitted_, "Call fit() first."
        feats = X[self.feature_cols]
        prob = self.stage1.predict_proba(feats)
        flag = self.stage1.predict(feats)
        sev_p = self.stage2.predict_proba(feats)
        sev = np.where(flag == 0, 0, np.maximum(sev_p.argmax(axis=1), 1))
        raised = np.zeros(len(X), dtype=bool)
        if floors:
            fl = self.detector_floor(X)
            raised = fl > sev
            sev = np.maximum(sev, fl)
        return pd.DataFrame({
            "anomaly_prob": prob, "anomaly_flag": flag, "severity_pred": sev,
            "severity_name": [SEVERITY_NAMES[s] for s in sev], "floor_applied": raised,
            **{f"p_{n.lower()}": sev_p[:, i] for i, n in enumerate(SEVERITY_NAMES)},
        }, index=X.index)
