"""
FreshnessFeatureEngineer (v2: causal features, ground-truth labels)
===================================================================
Feature families
F1  Temporal baseline   per-table, per-weekday rolling history (shifted, causal)
F2  Causal baseline     weekday offsets and robust level/scale from a fixed
                        reference window (first `ref_days` days), no future data
F3  Drift detectors     EWMA control chart + clipped CUSUM on the deseasonalised
                        latency z-score
F4  3-bit state machine execution x rows_read x rows_written
F5  DAG features        centrality, blast radius, depth, criticality
F6  Rolling statistics  entropy, IQR, kurtosis, velocity
F7  Row anomalies       rolling z-score and drop ratio
F8  Upstream context    failed or alerting ancestors on the same day

What changed from v1, and why
-----------------------------
* v1 ran STL over the whole series and calibrated CUSUM on the first 60% of the
  full series, so both used future data. STL also moved a gradual drift into
  its trend, leaving the residual-CUSUM blind to it. v2 detectors are causal.
* v1 built `severity_label` as a threshold rule over the model's own inputs. v2
  takes `severity_label` from the injected events (features/labels.py) and keeps
  the old rule as `rule_severity`, a baseline detector to beat.
* STL is still available through ``include_stl_audit=True`` for offline audits.

Assumption: the first `ref_days` days of each table are healthy (a Phase-I
reference window). Cascades in the simulation start on day 30, after day 28.
"""
import os
import sys
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.labels import ground_truth_family, ground_truth_severity  # noqa: E402

warnings.filterwarnings("ignore")


class CausalBaseline:
    """Weekday offsets plus robust level/scale from a reference window."""

    def __init__(self, ref_days: int = 28):
        self.ref_days = ref_days

    def transform(self, latency: pd.Series, dow: pd.Series) -> pd.DataFrame:
        lat = latency.astype(float).ffill().fillna(0).values
        dw = dow.values
        r = self.ref_days
        profile = pd.Series(lat[:r]).groupby(dw[:r]).mean()
        offset = np.array([profile.get(d, profile.mean()) for d in dw]) - profile.mean()
        ds = lat - offset
        mu = float(np.median(ds[:r]))
        sd = max(1.4826 * float(np.median(np.abs(ds[:r] - mu))), 1e-6)
        return pd.DataFrame({"lat_ds": ds, "ds_z": (ds - mu) / sd, "ref_mu": mu, "ref_sd": sd},
                            index=latency.index)


class DriftDetectors:
    """EWMA chart (level shifts, drift) and clipped CUSUM (small persistent shifts)."""

    def __init__(self, ewma_span=10, ewma_limit=3.0, ewma_run=3, cusum_k=1.0, cusum_h=5.0, clip=3.0):
        self.span, self.limit, self.run = ewma_span, ewma_limit, ewma_run
        self.k, self.h, self.clip = cusum_k, cusum_h, clip

    def transform(self, base: pd.DataFrame) -> pd.DataFrame:
        z = base["ds_z"].values
        ewma = pd.Series(np.clip(z, -4.0, 4.0)).ewm(span=self.span, adjust=False).mean().values
        over = (ewma > self.limit).astype(int)
        run = np.zeros(len(z), dtype=int)
        for t in range(len(z)):
            run[t] = run[t - 1] + 1 if (t and over[t]) else int(over[t])
        zc = np.clip(z, -self.clip, self.clip)
        s_pos, s_neg = np.zeros(len(z)), np.zeros(len(z))
        for t in range(1, len(z)):
            s_pos[t] = max(0.0, s_pos[t - 1] + zc[t] - self.k)
            s_neg[t] = max(0.0, s_neg[t - 1] - zc[t] - self.k)
        return pd.DataFrame({
            "ewma_z": ewma,
            "ewma_run": run,
            "ewma_alert": (run >= self.run).astype(int),
            "cusum_s_pos": s_pos,
            "cusum_s_neg": s_neg,
            "cusum_alert_up": (s_pos > self.h).astype(int),
            "cusum_alert_dn": (s_neg > self.h).astype(int),
            "cusum_drift_score": np.clip(s_pos / self.h, 0, 5.0),
        }, index=base.index)


DIAGNOSTIC_STATE_MAP = {
    (1, 1, 1): (0, "HEALTHY", "none"),
    (1, 1, 0): (1, "SILENT_BUSINESS_FAILURE", "analytics_engineer"),
    (1, 0, 0): (2, "EXPECTED_EMPTY", "monitor_only"),
    (1, 0, 1): (3, "IMPOSSIBLE_STATE", "data_quality"),
    (0, 0, 0): (4, "INFRASTRUCTURE_FAILURE", "platform_team"),
    (0, 1, 1): (5, "EXEC_FAILED_DATA_OK", "orchestration_team"),
    (0, 1, 0): (6, "PARTIAL_FAILURE", "platform_team"),
    (0, 0, 1): (7, "IMPOSSIBLE_STATE", "data_quality"),
}


def compute_diagnostic_state(row, rows_read_floor: int = 10, rows_written_floor: int = 10) -> tuple:
    exec_ok = int(row["execution_status"] in ["SUCCESS", "RETRIED"])
    read_ok = int(row["rows_read"] >= rows_read_floor)
    write_ok = int(row["rows_written"] >= rows_written_floor)
    code, label, owner = DIAGNOSTIC_STATE_MAP.get((exec_ok, read_ok, write_ok), (8, "UNKNOWN", "unknown"))
    return code, label, owner, exec_ok, read_ok, write_ok


class DAGFeatureExtractor:
    CRIT = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

    def __init__(self, dag: nx.DiGraph):
        self.dag = dag
        G = dag
        self.centrality = nx.betweenness_centrality(G)
        self.in_degree = dict(G.in_degree())
        self.out_degree = dict(G.out_degree())
        self.depth = {n: 0 for n in G.nodes}
        for n in nx.topological_sort(G):
            for s in G.successors(n):
                self.depth[s] = max(self.depth[s], self.depth[n] + 1)
        self.blast_radius = {n: len(nx.descendants(G, n)) for n in G.nodes}
        self.ancestors = {n: nx.ancestors(G, n) for n in G.nodes}
        self.ancestry_count = {n: len(a) for n, a in self.ancestors.items()}
        self.crit_numeric = {n: self.CRIT.get(G.nodes[n].get("criticality"), 1) for n in G.nodes}

    def get_features(self, table: str) -> dict:
        return {
            "dag_centrality": round(self.centrality.get(table, 0), 6),
            "dag_in_degree": self.in_degree.get(table, 0),
            "dag_out_degree": self.out_degree.get(table, 0),
            "dag_depth": self.depth.get(table, 0),
            "dag_blast_radius": self.blast_radius.get(table, 0),
            "dag_ancestry_count": self.ancestry_count.get(table, 0),
            "dag_criticality": self.crit_numeric.get(table, 1),
        }


class FreshnessFeatureEngineer:
    def __init__(self, df: pd.DataFrame, dag: nx.DiGraph, ref_days: int = 28,
                 include_stl_audit: bool = False):
        self.df = df.copy().sort_values(["table_name", "day_idx"])
        self.dag = dag
        self.dag_extractor = DAGFeatureExtractor(dag)
        self.baseline = CausalBaseline(ref_days)
        self.detectors = DriftDetectors()
        self.include_stl_audit = include_stl_audit

    def _f1_temporal(self, grp):
        grp = grp.copy()
        by = grp.groupby("day_of_week")["latency_minutes"]
        roll = lambda x: x.shift(1).rolling(21, min_periods=5)
        grp["hist_mean_21d"] = by.transform(lambda x: roll(x).mean())
        grp["hist_std_21d"] = by.transform(lambda x: roll(x).std())
        grp["hist_p90_21d"] = by.transform(lambda x: roll(x).quantile(0.90))
        grp["latency_zscore"] = (grp["latency_minutes"] - grp["hist_mean_21d"]) / grp["hist_std_21d"].replace(0, 1e-9)
        grp["latency_pct_rank"] = by.transform(
            lambda x: roll(x).apply(lambda w: stats.percentileofscore(w, w[-1]) / 100 if len(w) else 0.5, raw=True))
        return grp

    def _f2_f3_causal(self, grp):
        grp = grp.copy().reset_index(drop=True)
        base = self.baseline.transform(grp["latency_minutes"], grp["day_of_week"])
        return pd.concat([grp, base, self.detectors.transform(base)], axis=1)

    def _stl_audit(self, grp):
        grp = grp.copy().reset_index(drop=True)
        lat = grp["latency_minutes"].ffill().fillna(0)
        try:
            res = STL(lat, period=7, robust=True).fit()
            grp["stl_trend"], grp["stl_seasonal"], grp["stl_residual"] = res.trend, res.seasonal, res.resid
        except Exception:
            grp["stl_trend"], grp["stl_seasonal"], grp["stl_residual"] = lat, 0.0, lat - lat.mean()
        return grp

    def _f4_state_machine(self, df):
        df = df.copy()
        floors = df[df["rows_written"] > 0].groupby("table_name")["rows_written"].quantile(0.05).to_dict()
        out = []
        for _, row in df.iterrows():
            floor = max(10, int(floors.get(row["table_name"], 10)))
            sc, label, owner, e, rr, rw = compute_diagnostic_state(row, floor, floor)
            out.append({"diag_state_code": sc, "diag_state_label": label, "diag_owner": owner,
                        "diag_exec_ok": e, "diag_read_ok": rr, "diag_write_ok": rw,
                        "row_write_ratio": row["rows_written"] / max(1, row["rows_read"]),
                        "was_retried": int(row["execution_status"] == "RETRIED")})
        return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)

    def _f5_dag(self, df):
        rows = [self.dag_extractor.get_features(t) for t in df["table_name"]]
        return pd.concat([df, pd.DataFrame(rows, index=df.index)], axis=1)

    def _f6_rolling(self, grp):
        grp = grp.copy()
        lat = grp["latency_minutes"]

        def entropy(x):
            x = np.asarray(x)
            x = x[x > 0]
            if len(x) < 3:
                return 0.0
            h, _ = np.histogram(x, bins=min(10, len(x) // 2 + 1), density=True)
            h = h[h > 0]
            return float(-np.sum(h * np.log(h + 1e-9)))

        r7 = lat.shift(1).rolling(7, min_periods=3)
        grp["roll7_mean"], grp["roll7_std"] = r7.mean(), r7.std()
        grp["roll7_iqr"] = r7.apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25), raw=True)
        grp["roll7_kurtosis"] = lat.shift(1).rolling(7, min_periods=4).apply(lambda x: stats.kurtosis(x), raw=True)
        grp["roll7_entropy"] = lat.shift(1).rolling(14, min_periods=7).apply(entropy, raw=True)
        grp["latency_velocity"] = lat.diff()
        grp["latency_acceleration"] = grp["latency_velocity"].diff()
        return grp

    def _f7_row_anomaly(self, grp):
        grp = grp.copy()
        rows = grp["rows_written"].astype(float)
        r7 = rows.shift(1).rolling(7, min_periods=3)
        grp["rows_roll7_mean"], grp["rows_roll7_std"] = r7.mean(), r7.std()
        grp["rows_zscore"] = (rows - grp["rows_roll7_mean"]) / grp["rows_roll7_std"].replace(0, 1e-9)
        grp["rows_pct_drop"] = ((grp["rows_roll7_mean"] - rows) / grp["rows_roll7_mean"].replace(0, 1e-9)).clip(lower=0)
        return grp

    def _f8_upstream(self, df):
        failed = df[df["diag_state_code"].isin([4, 5, 6])].groupby("day_idx")["table_name"].agg(set).to_dict()
        alert = df[df["ewma_alert"] == 1].groupby("day_idx")["table_name"].agg(set).to_dict()
        anc = self.dag_extractor.ancestors
        df = df.copy()
        df["upstream_failed_count"] = [len(anc[t] & failed.get(d, set())) for t, d in zip(df["table_name"], df["day_idx"])]
        df["upstream_alert_count"] = [len(anc[t] & alert.get(d, set())) for t, d in zip(df["table_name"], df["day_idx"])]
        return df

    @staticmethod
    def rule_baseline(full: pd.DataFrame) -> np.ndarray:
        """v1 threshold rule, kept as a baseline detector (never a training target)."""
        c = [
            (full["diag_state_code"].isin([4, 5, 6])) | (full["cusum_drift_score"] > 4.0),
            (full["cusum_drift_score"] > 2.5) | (full["latency_zscore"] > 3.0) | (full["diag_state_code"] == 1),
            (full["cusum_drift_score"] > 1.5) | (full["latency_zscore"] > 2.0) | (full["rows_zscore"] < -2.5),
            (full["cusum_drift_score"] > 0.5) | (full["latency_zscore"] > 1.0),
        ]
        sev = np.zeros(len(full), dtype=int)
        for level, cond in enumerate(reversed(c), start=1):
            sev = np.where(cond & (sev < level), level, sev)
        return sev

    def build(self, verbose: bool = False) -> pd.DataFrame:
        chunks = []
        for table in self.df["table_name"].unique():
            grp = self.df[self.df["table_name"] == table].copy()
            grp = self._f1_temporal(grp)
            grp = self._f2_f3_causal(grp)
            if self.include_stl_audit:
                grp = self._stl_audit(grp)
            grp = self._f6_rolling(grp)
            grp = self._f7_row_anomaly(grp)
            chunks.append(grp)
            if verbose:
                print(f"  built {table}")
        full = pd.concat(chunks).sort_values(["day_idx", "table_name"])
        full = self._f4_state_machine(full)
        full = self._f5_dag(full)
        full = self._f8_upstream(full)
        full = full.reset_index(drop=True)

        names = ["NORMAL", "WATCH", "WARN", "DEGRADED", "CRITICAL"]
        full["rule_severity"] = self.rule_baseline(full)
        full["event_family"] = ground_truth_family(full)
        full["severity_label"] = ground_truth_severity(full)
        full["severity_name"] = pd.Categorical(full["severity_label"].map(dict(enumerate(names))),
                                               categories=names, ordered=True)
        return full


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from data.simulator import PipelineEcosystemSimulator

    raw, graph = PipelineEcosystemSimulator(n_days=180, seed=42).generate()
    features = FreshnessFeatureEngineer(raw, graph).build(verbose=True)
    features.to_csv(os.path.join(root, "data", "feature_matrix.csv"), index=False)
    print(f"Saved feature matrix: {features.shape[1]} columns, {len(features):,} rows")
    print(features["severity_name"].value_counts().sort_index().to_string())
