"""
FreshnessFeatureEngineer
========================
Transforms raw pipeline run records into a rich feature matrix.

Feature families:
  F1 — Temporal baseline features (learned per-table arrival distribution)
  F2 — STL decomposition features (trend + seasonality + residual on latency)
  F3 — CUSUM control chart signals (S+, S-, alert flag, drift score)
  F4 — 3-bit diagnostic state machine (execution x rows_read x rows_written)
  F5 — DAG-derived graph features (centrality, blast radius, depth, ancestry load)
  F6 — Rolling window statistics (not just mean/std — entropy, IQR, kurtosis)
  F7 — Row count anomaly signals (Z-score + deviation from historical quantile)

Why STL before CUSUM (the critical design decision):
  CUSUM operates on the assumption that the baseline process mean μ₀ is stable.
  Raw latency violates this — it has weekly seasonality (weekends are faster),
  hourly patterns, and long-term trend. If you run CUSUM on raw latency,
  Monday's natural high will look like a drift signal on Sunday's baseline.
  Solution: STL decomposes latency → trend + seasonal + residual.
  CUSUM runs on the RESIDUAL only — pure unexplained deviation.
"""

import numpy as np
import pandas as pd
import networkx as nx
from statsmodels.tsa.seasonal import STL
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# CUSUM ENGINE — per-table stateful control chart
# ─────────────────────────────────────────────────────────────
class CUSUMEngine:
    """
    One-sided upper CUSUM for detecting positive latency drift.
    Two-sided variant also computed for anomalous speed-up detection.

    Parameters
    ----------
    k : float
        Allowance parameter. Typically k = 0.5σ (half the shift to detect).
        We auto-calibrate k from the table's historical residual std.
    h : float
        Decision interval (control limit). Alert when S+ > h.
        h = 4σ catches shifts > 1σ with ARL ≈ 168 observations.
    """

    def __init__(self, k_factor: float = 0.5, h_factor: float = 4.0):
        self.k_factor = k_factor
        self.h_factor = h_factor

    def fit_transform(self, residuals: pd.Series) -> pd.DataFrame:
        """
        residuals : STL residual of latency time series (mean ~0)
        Returns DataFrame with S+, S-, alert flags, drift score.
        """
        # Auto-calibrate from in-sample variance (first 60% of data)
        train_len  = max(10, int(len(residuals) * 0.6))
        sigma      = residuals.iloc[:train_len].std()
        mu0        = residuals.iloc[:train_len].mean()

        k = self.k_factor * sigma   # allowance
        h = self.h_factor * sigma   # control limit

        S_pos  = np.zeros(len(residuals))
        S_neg  = np.zeros(len(residuals))
        x      = residuals.values

        for t in range(1, len(x)):
            S_pos[t] = max(0, S_pos[t-1] + (x[t] - mu0 - k))
            S_neg[t] = max(0, S_neg[t-1] - (x[t] - mu0 - k))

        alert_up   = (S_pos > h).astype(int)
        alert_down = (S_neg > h).astype(int)

        # Drift score: normalised cumulative signal relative to control limit
        drift_score = np.clip(S_pos / (h + 1e-9), 0, 5.0)

        return pd.DataFrame({
            "cusum_s_pos":    S_pos,
            "cusum_s_neg":    S_neg,
            "cusum_alert_up": alert_up,
            "cusum_alert_dn": alert_down,
            "cusum_drift_score": drift_score,
            "cusum_k":        k,
            "cusum_h":        h,
            "cusum_sigma":    sigma,
        }, index=residuals.index)


# ─────────────────────────────────────────────────────────────
# 3-BIT DIAGNOSTIC STATE MACHINE
# ─────────────────────────────────────────────────────────────
DIAGNOSTIC_STATE_MAP = {
    # (exec_ok, rows_read_ok, rows_written_ok) → (state_code, label, owner)
    (1, 1, 1): (0, "HEALTHY",                 "none"),
    (1, 1, 0): (1, "SILENT_BUSINESS_FAILURE", "analytics_engineer"),
    (1, 0, 0): (2, "EXPECTED_EMPTY",          "monitor_only"),
    (1, 0, 1): (3, "IMPOSSIBLE_STATE",        "data_quality"),
    (0, 0, 0): (4, "INFRASTRUCTURE_FAILURE",  "platform_team"),
    (0, 1, 1): (5, "EXEC_FAILED_DATA_OK",     "orchestration_team"),
    (0, 1, 0): (6, "PARTIAL_FAILURE",         "platform_team"),
    (0, 0, 1): (7, "IMPOSSIBLE_STATE",        "data_quality"),
}

def compute_diagnostic_state(row: pd.Series,
                               rows_read_floor: int = 10,
                               rows_written_floor: int = 10) -> tuple:
    exec_ok        = int(row["execution_status"] in ["SUCCESS", "RETRIED"])
    rows_read_ok   = int(row["rows_read"] >= rows_read_floor)
    rows_written_ok= int(row["rows_written"] >= rows_written_floor)
    key = (exec_ok, rows_read_ok, rows_written_ok)
    state_code, label, owner = DIAGNOSTIC_STATE_MAP.get(key, (8, "UNKNOWN", "unknown"))
    return state_code, label, owner, exec_ok, rows_read_ok, rows_written_ok


# ─────────────────────────────────────────────────────────────
# DAG FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────
class DAGFeatureExtractor:

    def __init__(self, dag: nx.DiGraph):
        self.dag = dag
        self._precompute()

    def _precompute(self):
        G = self.dag
        # Betweenness centrality: high-centrality nodes are single points of failure
        self.centrality     = nx.betweenness_centrality(G)
        # In-degree: number of dependencies (more deps = more failure exposure)
        self.in_degree      = dict(G.in_degree())
        # Out-degree: blast radius at immediate next level
        self.out_degree     = dict(G.out_degree())
        # Topological depth (level in DAG)
        self.depth          = {n: 0 for n in G.nodes}
        for n in nx.topological_sort(G):
            for successor in G.successors(n):
                self.depth[successor] = max(
                    self.depth[successor], self.depth[n] + 1)
        # Total downstream impact: all nodes reachable from this one
        self.blast_radius   = {
            n: len(nx.descendants(G, n)) for n in G.nodes
        }
        # Total upstream ancestry: how many tables must succeed before this one
        self.ancestry_count = {
            n: len(nx.ancestors(G, n)) for n in G.nodes
        }
        # Criticality score: map categorical to numeric
        crit_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        from data.simulator import DAG_SCHEMA
        self.crit_numeric   = {
            n: crit_map.get(DAG_SCHEMA[n]["criticality"], 1)
            for n in G.nodes
        }

    def get_features(self, table: str) -> dict:
        return {
            "dag_centrality":     round(self.centrality.get(table, 0), 6),
            "dag_in_degree":      self.in_degree.get(table, 0),
            "dag_out_degree":     self.out_degree.get(table, 0),
            "dag_depth":          self.depth.get(table, 0),
            "dag_blast_radius":   self.blast_radius.get(table, 0),
            "dag_ancestry_count": self.ancestry_count.get(table, 0),
            "dag_criticality":    self.crit_numeric.get(table, 1),
        }


# ─────────────────────────────────────────────────────────────
# MAIN FEATURE ENGINEER
# ─────────────────────────────────────────────────────────────
class FreshnessFeatureEngineer:

    def __init__(self, df: pd.DataFrame, dag: nx.DiGraph):
        self.df           = df.copy().sort_values(["table_name", "day_idx"])
        self.dag          = dag
        self.dag_extractor= DAGFeatureExtractor(dag)
        self.cusum_engine = CUSUMEngine(k_factor=0.5, h_factor=4.0)

    # ── F1: Temporal baseline features ──────────────────────
    def _f1_temporal(self, grp: pd.DataFrame) -> pd.DataFrame:
        # Rolling 21-day (3-week) historical stats per day-of-week
        # Critical: group by day-of-week before rolling to avoid
        # mixing Mon baseline with Sun baseline
        grp = grp.copy()
        grp["hist_mean_21d"] = (
            grp.groupby("day_of_week")["latency_minutes"]
               .transform(lambda x: x.shift(1).rolling(21, min_periods=5).mean())
        )
        grp["hist_std_21d"] = (
            grp.groupby("day_of_week")["latency_minutes"]
               .transform(lambda x: x.shift(1).rolling(21, min_periods=5).std())
        )
        grp["hist_p90_21d"] = (
            grp.groupby("day_of_week")["latency_minutes"]
               .transform(lambda x: x.shift(1).rolling(21, min_periods=5)
                          .quantile(0.90))
        )
        # Z-score of current latency vs historical distribution
        grp["latency_zscore"] = (
            (grp["latency_minutes"] - grp["hist_mean_21d"])
            / (grp["hist_std_21d"].replace(0, 1e-9))
        )
        # Percentile rank within window (non-parametric)
        grp["latency_pct_rank"] = (
            grp.groupby("day_of_week")["latency_minutes"]
               .transform(lambda x: x.shift(1).rolling(21, min_periods=5)
                          .apply(lambda w: stats.percentileofscore(w, w[-1])
                                 / 100 if len(w) > 0 else 0.5, raw=True))
        )
        return grp

    # ── F2: STL decomposition + residual extraction ─────────
    def _f2_stl(self, grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.copy().reset_index(drop=True)
        lat = grp["latency_minutes"].ffill().fillna(0)

        # STL requires at least 2 full periods
        # Period = 7 (weekly seasonality in daily data)
        if len(lat) < 14:
            grp["stl_trend"]    = lat
            grp["stl_seasonal"] = 0.0
            grp["stl_residual"] = 0.0
            return grp

        try:
            stl    = STL(lat, period=7, robust=True)
            result = stl.fit()
            grp["stl_trend"]    = result.trend
            grp["stl_seasonal"] = result.seasonal
            grp["stl_residual"] = result.resid   # ← CUSUM will run on this
        except Exception:
            grp["stl_trend"]    = lat
            grp["stl_seasonal"] = 0.0
            grp["stl_residual"] = lat - lat.mean()

        return grp

    # ── F3: CUSUM signals on STL residual ───────────────────
    def _f3_cusum(self, grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.copy().reset_index(drop=True)
        residual = grp["stl_residual"].fillna(0)
        cusum_df = self.cusum_engine.fit_transform(residual)
        return pd.concat([grp, cusum_df], axis=1)

    # ── F4: 3-bit diagnostic state machine ──────────────────
    def _f4_state_machine(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Compute per-table dynamic row floor (5th percentile of historical writes)
        row_floors = (
            df[df["rows_written"] > 0]
              .groupby("table_name")["rows_written"]
              .quantile(0.05)
              .to_dict()
        )
        results = []
        for _, row in df.iterrows():
            floor = max(10, int(row_floors.get(row["table_name"], 10)))
            sc, label, owner, e, rr, rw = compute_diagnostic_state(row, floor, floor)
            results.append({
                "diag_state_code":  sc,
                "diag_state_label": label,
                "diag_owner":       owner,
                "diag_exec_ok":     e,
                "diag_read_ok":     rr,
                "diag_write_ok":    rw,
                "row_write_ratio":  (row["rows_written"] /
                                     max(1, row["rows_read"])),
            })
        return pd.concat([df, pd.DataFrame(results, index=df.index)], axis=1)

    # ── F5: DAG graph features ───────────────────────────────
    def _f5_dag(self, df: pd.DataFrame) -> pd.DataFrame:
        dag_rows = [self.dag_extractor.get_features(t)
                    for t in df["table_name"]]
        return pd.concat([df, pd.DataFrame(dag_rows, index=df.index)], axis=1)

    # ── F6: Rolling entropy + higher-order stats ─────────────
    def _f6_rolling(self, grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.copy()
        lat = grp["latency_minutes"]

        def safe_entropy(x):
            x = np.array(x)
            x = x[x > 0]
            if len(x) < 3:
                return 0.0
            hist, _ = np.histogram(x, bins=min(10, len(x)//2+1), density=True)
            hist    = hist[hist > 0]
            return float(-np.sum(hist * np.log(hist + 1e-9)))

        grp["roll7_mean"]     = lat.shift(1).rolling(7,  min_periods=3).mean()
        grp["roll7_std"]      = lat.shift(1).rolling(7,  min_periods=3).std()
        grp["roll7_iqr"]      = lat.shift(1).rolling(7,  min_periods=3).apply(
                                    lambda x: np.percentile(x,75)-np.percentile(x,25), raw=True)
        grp["roll7_kurtosis"] = lat.shift(1).rolling(7,  min_periods=4).apply(
                                    lambda x: stats.kurtosis(x), raw=True)
        grp["roll7_entropy"]  = lat.shift(1).rolling(14, min_periods=7).apply(
                                    safe_entropy, raw=True)
        # Velocity: day-over-day change in latency
        grp["latency_velocity"]     = lat.diff()
        grp["latency_acceleration"] = grp["latency_velocity"].diff()
        return grp

    # ── F7: Row count anomaly signals ───────────────────────
    def _f7_row_anomaly(self, grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.copy()
        rows = grp["rows_written"].astype(float)
        grp["rows_roll7_mean"] = rows.shift(1).rolling(7, min_periods=3).mean()
        grp["rows_roll7_std"]  = rows.shift(1).rolling(7, min_periods=3).std()
        grp["rows_zscore"]     = (
            (rows - grp["rows_roll7_mean"])
            / (grp["rows_roll7_std"].replace(0, 1e-9))
        )
        grp["rows_pct_drop"]   = (
            (grp["rows_roll7_mean"] - rows)
            / (grp["rows_roll7_mean"].replace(0, 1e-9))
        ).clip(lower=0)
        return grp

    # ── ORCHESTRATOR ─────────────────────────────────────────
    def build(self) -> pd.DataFrame:
        print("Building feature matrix...")
        tables  = self.df["table_name"].unique()
        chunks  = []

        for table in tables:
            grp = self.df[self.df["table_name"] == table].copy()
            grp = self._f1_temporal(grp)
            grp = self._f2_stl(grp)
            grp = self._f3_cusum(grp)
            grp = self._f6_rolling(grp)
            grp = self._f7_row_anomaly(grp)
            chunks.append(grp)
            print(f"  ✓ {table}")

        full = pd.concat(chunks).sort_values(["day_idx", "table_name"])
        full = self._f4_state_machine(full)
        full = self._f5_dag(full)

        # ── Composite severity target (for Phase 3 model) ──
        # NOT a simple binary — 5-class ordinal severity
        # 0=NORMAL, 1=WATCH, 2=WARN, 3=DEGRADED, 4=CRITICAL
        conditions = [
            (full["diag_state_code"].isin([4, 5, 6])) |
            (full["cusum_drift_score"] > 4.0),
            (full["cusum_drift_score"] > 2.5) |
            (full["latency_zscore"] > 3.0) |
            (full["diag_state_code"] == 1),
            (full["cusum_drift_score"] > 1.5) |
            (full["latency_zscore"] > 2.0) |
            (full["rows_zscore"] < -2.5),
            (full["cusum_drift_score"] > 0.5) |
            (full["latency_zscore"] > 1.0),
        ]
        severity = np.zeros(len(full), dtype=int)
        for level, cond in enumerate(reversed(conditions), start=1):
            severity = np.where(cond & (severity < level), level, severity)

        full["severity_label"] = severity
        full["severity_name"]  = pd.Categorical(
            full["severity_label"].map(
                {0:"NORMAL",1:"WATCH",2:"WARN",3:"DEGRADED",4:"CRITICAL"}),
            categories=["NORMAL","WATCH","WARN","DEGRADED","CRITICAL"],
            ordered=True
        )
        print(f"\nFeature matrix: {full.shape}")
        print(f"\nSeverity distribution:")
        print(full["severity_name"].value_counts().sort_index().to_string())
        return full


if __name__ == "__main__":
    import sys, os
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _ROOT)
    from data.simulator import PipelineEcosystemSimulator

    sim = PipelineEcosystemSimulator(n_days=180, seed=42)
    df, dag = sim.generate()

    engineer = FreshnessFeatureEngineer(df, dag)
    features = engineer.build()
    features.to_csv(
        os.path.join(_ROOT, "data", "feature_matrix.csv"), index=False)
    print(f"\n✓ Saved feature matrix — {features.shape[1]} features, {len(features):,} rows")
