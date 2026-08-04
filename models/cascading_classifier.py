"""
CascadingFreshnessClassifier
=============================
Two-stage architecture:

Stage 1 — Binary Anomaly Detector
    "Is something wrong with this pipeline run?"
    LightGBM binary classifier on all 60 features.
    Outputs: anomaly_prob, anomaly_flag

Stage 2 — Ordinal Severity Ranker
    "How bad is it?" — only runs when Stage 1 flags anomaly.
    Uses Frank et al. (2001) ordinal decomposition:
    Decomposes 5-class ordinal problem into 4 binary problems.
    P(Y > k) for k in {NORMAL, WATCH, WARN, DEGRADED}
    Combines binary outputs to produce calibrated ordinal probabilities.

Why NOT a single 5-class classifier (the common tutorial approach):
    A flat multiclass model treats NORMAL→CRITICAL as unordered.
    It can output "CRITICAL" for what should be "WATCH" and the loss
    function penalises it identically to predicting "NORMAL" → complete
    misuse of ordinal information.
    The ordinal decomposition preserves rank information in the loss
    function: predicting WARN when true is DEGRADED costs less than
    predicting NORMAL when true is DEGRADED.

CUSUM Hard Override Rule:
    If cusum_alert_up == 1 AND dag_criticality >= 3:
        severity_floor = WARN (severity index >= 2)
    This is a domain-knowledge rule that cannot be learned from data alone
    in a short training window. CUSUM is a sequential test with provable
    statistical properties; letting the ML model override it on CRITICAL
    tables would be epistemically wrong.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

SEVERITY_NAMES  = ["NORMAL", "WATCH", "WARN", "DEGRADED", "CRITICAL"]
SEVERITY_INT    = {v: i for i, v in enumerate(SEVERITY_NAMES)}
N_CLASSES       = len(SEVERITY_NAMES)


# ─────────────────────────────────────────────────────────────
# ORDINAL BINARY DECOMPOSITION
# Frank & Hall (2001): "A Simple Approach to Ordinal Classification"
# ─────────────────────────────────────────────────────────────
class OrdinalDecompositionRanker(BaseEstimator, ClassifierMixin):
    """
    Trains K-1 binary classifiers for K ordinal classes.
    Binary classifier k asks: P(Y > k)?

    For 5 severity levels: trains 4 classifiers:
        Clf0: P(severity > NORMAL)
        Clf1: P(severity > WATCH)
        Clf2: P(severity > WARN)
        Clf3: P(severity > DEGRADED)

    Final class probabilities:
        P(NORMAL)   = 1 - P(Y>0)
        P(WATCH)    = P(Y>0) - P(Y>1)
        P(WARN)     = P(Y>1) - P(Y>2)
        P(DEGRADED) = P(Y>2) - P(Y>3)
        P(CRITICAL) = P(Y>3)
    """

    def __init__(self, lgb_params: dict = None):
        self.lgb_params = lgb_params or {
            "objective":        "binary",
            "metric":           "auc",
            "n_estimators":     400,
            "learning_rate":    0.04,
            "num_leaves":       47,
            "min_child_samples": 20,
            "feature_fraction": 0.75,
            "bagging_fraction": 0.75,
            "bagging_freq":     5,
            "reg_alpha":        0.1,
            "reg_lambda":       0.2,
            "verbose":          -1,
            "n_jobs":           -1,
            "random_state":     42,
        }
        self.classifiers_ = []
        self.classes_      = np.arange(N_CLASSES)

    def fit(self, X: pd.DataFrame, y: pd.Series,
            eval_set=None) -> "OrdinalDecompositionRanker":
        y_int = y.values if hasattr(y, "values") else np.array(y)
        self.classifiers_ = []

        for k in range(N_CLASSES - 1):
            # Binary target: 1 if severity > k, else 0
            y_binary = (y_int > k).astype(int)
            pos_rate  = y_binary.mean()

            params = self.lgb_params.copy()
            # Class weight to handle imbalance at each threshold
            if 0.05 < pos_rate < 0.95:
                params["scale_pos_weight"] = (1 - pos_rate) / (pos_rate + 1e-9)

            clf = lgb.LGBMClassifier(**params)
            clf.fit(X, y_binary,
                    eval_set=[(eval_set[0], (eval_set[1].values > k).astype(int))]
                              if eval_set else None,
                    callbacks=[lgb.early_stopping(40, verbose=False),
                                lgb.log_evaluation(period=-1)]
                               if eval_set else None)
            self.classifiers_.append(clf)
            auc = roc_auc_score(y_binary,
                                clf.predict_proba(X)[:,1])
            print(f"    Ordinal clf {k} (P(Y>{k})={SEVERITY_NAMES[k]}): "
                  f"AUC={auc:.4f}  pos_rate={pos_rate:.3f}")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Collect P(Y > k) for each threshold k
        exceed_probs = np.stack(
            [clf.predict_proba(X)[:, 1] for clf in self.classifiers_],
            axis=1
        )  # shape (n, K-1)

        # Ensure monotonically non-increasing: P(Y>0) >= P(Y>1) >= ...
        for k in range(1, exceed_probs.shape[1]):
            exceed_probs[:, k] = np.minimum(exceed_probs[:, k],
                                             exceed_probs[:, k-1])

        # Convert to class probabilities
        n = len(X)
        class_probs = np.zeros((n, N_CLASSES))
        class_probs[:, 0] = 1 - exceed_probs[:, 0]
        for k in range(1, N_CLASSES - 1):
            class_probs[:, k] = exceed_probs[:, k-1] - exceed_probs[:, k]
        class_probs[:, -1] = exceed_probs[:, -1]

        # Clip and renormalise
        class_probs = np.clip(class_probs, 0, 1)
        class_probs /= class_probs.sum(axis=1, keepdims=True) + 1e-9
        return class_probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Aggregate importance across all K-1 binary classifiers."""
        imp_dfs = []
        for k, clf in enumerate(self.classifiers_):
            imp_dfs.append(pd.DataFrame({
                "feature":    feature_names,
                "importance": clf.feature_importances_,
                "threshold":  k,
            }))
        combined = pd.concat(imp_dfs)
        return (combined.groupby("feature")["importance"]
                        .mean()
                        .sort_values(ascending=False)
                        .reset_index()
                        .rename(columns={"importance": "mean_importance"}))


# ─────────────────────────────────────────────────────────────
# STAGE 1 — BINARY ANOMALY DETECTOR
# ─────────────────────────────────────────────────────────────
class BinaryAnomalyDetector:

    def __init__(self):
        self.model = lgb.LGBMClassifier(
            objective        = "binary",
            metric           = "auc",
            n_estimators     = 500,
            learning_rate    = 0.03,
            num_leaves       = 63,
            min_child_samples= 25,
            feature_fraction = 0.70,
            bagging_fraction = 0.70,
            bagging_freq     = 5,
            reg_alpha        = 0.15,
            reg_lambda       = 0.25,
            verbose          = -1,
            n_jobs           = -1,
            random_state     = 42,
        )
        self.threshold_ = 0.5

    def fit(self, X, y_binary, eval_set=None):
        self.model.fit(
            X, y_binary,
            eval_set    = eval_set,
            callbacks   = [lgb.early_stopping(50, verbose=False),
                            lgb.log_evaluation(period=-1)]
        )
        # Calibrate threshold on training set using F-beta (beta=0.5)
        # We prefer precision over recall: false alarms cost engineer attention
        probs = self.model.predict_proba(X)[:, 1]
        best_f, best_t = 0, 0.5
        for t in np.arange(0.3, 0.85, 0.02):
            preds = (probs >= t).astype(int)
            tp = ((preds==1) & (y_binary==1)).sum()
            fp = ((preds==1) & (y_binary==0)).sum()
            fn = ((preds==0) & (y_binary==1)).sum()
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            # F0.5: weights precision twice as much as recall
            fb = (1 + 0.25) * prec * rec / (0.25 * prec + rec + 1e-9)
            if fb > best_f:
                best_f, best_t = fb, t
        self.threshold_ = best_t
        print(f"    Stage 1 threshold calibrated: {best_t:.2f}  (F0.5={best_f:.4f})")
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return (self.model.predict_proba(X)[:, 1] >= self.threshold_).astype(int)


# ─────────────────────────────────────────────────────────────
# CASCADING PIPELINE — Stage 1 → Stage 2 → CUSUM Override
# ─────────────────────────────────────────────────────────────
class CascadingFreshnessClassifier:

    def __init__(self, feature_cols: list):
        self.feature_cols    = feature_cols
        self.stage1          = BinaryAnomalyDetector()
        self.stage2          = OrdinalDecompositionRanker()
        self.is_fitted_      = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
             X_val: pd.DataFrame, y_val: pd.Series):
        """
        y_train / y_val : integer severity (0=NORMAL … 4=CRITICAL)
        """
        print("\n── Stage 1: Binary Anomaly Detector ──")
        y_bin_train = (y_train > 0).astype(int)
        y_bin_val   = (y_val   > 0).astype(int)
        self.stage1.fit(
            X_train[self.feature_cols], y_bin_train,
            eval_set=[(X_val[self.feature_cols], y_bin_val)]
        )
        s1_auc = roc_auc_score(
            y_bin_val,
            self.stage1.predict_proba(X_val[self.feature_cols])[:, 1]
        )
        print(f"    Stage 1 Val AUC: {s1_auc:.4f}")

        print("\n── Stage 2: Ordinal Severity Ranker ──")
        # Train Stage 2 only on anomalous training rows (severity > 0)
        # This prevents NORMAL examples from diluting the severity signal
        anom_mask_train = y_train > 0
        anom_mask_val   = y_val   > 0
        print(f"    Stage 2 training rows (anomalous only): "
              f"{anom_mask_train.sum():,} / {len(y_train):,}")
        self.stage2.fit(
            X_train[self.feature_cols][anom_mask_train],
            y_train[anom_mask_train],
            eval_set=(X_val[self.feature_cols][anom_mask_val],
                       y_val[anom_mask_val])
        )
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame,
                 apply_cusum_override: bool = True) -> pd.DataFrame:
        """
        Returns DataFrame with:
            anomaly_prob, anomaly_flag,
            severity_pred (int), severity_name,
            severity_proba (array),
            cusum_overridden (bool)
        """
        assert self.is_fitted_, "Call fit() first."
        feats  = X[self.feature_cols]
        n      = len(X)

        # Stage 1
        anom_proba = self.stage1.predict_proba(feats)[:, 1]
        anom_flag  = self.stage1.predict(feats)

        # Stage 2 — run on all rows, gate output by Stage 1
        sev_proba  = self.stage2.predict_proba(feats)   # (n, 5)
        sev_pred   = sev_proba.argmax(axis=1)

        # Gate: if Stage 1 says NORMAL, force severity = NORMAL
        sev_pred   = np.where(anom_flag == 0, 0, sev_pred)

        # CUSUM hard override
        cusum_overridden = np.zeros(n, dtype=bool)
        if apply_cusum_override and "cusum_alert_up" in X.columns:
            cusum_fire   = X["cusum_alert_up"].fillna(0).values.astype(bool)
            high_crit    = X["dag_criticality"].fillna(0).values >= 3
            override_mask= cusum_fire & high_crit
            # Floor severity to WARN (2) for CUSUM-triggered high-criticality tables
            sev_pred     = np.where(override_mask & (sev_pred < 2), 2, sev_pred)
            cusum_overridden = override_mask & (sev_pred == 2)

        return pd.DataFrame({
            "anomaly_prob":      anom_proba,
            "anomaly_flag":      anom_flag,
            "severity_pred":     sev_pred,
            "severity_name":     [SEVERITY_NAMES[s] for s in sev_pred],
            "cusum_overridden":  cusum_overridden,
            "p_normal":          sev_proba[:, 0],
            "p_watch":           sev_proba[:, 1],
            "p_warn":            sev_proba[:, 2],
            "p_degraded":        sev_proba[:, 3],
            "p_critical":        sev_proba[:, 4],
        }, index=X.index)
