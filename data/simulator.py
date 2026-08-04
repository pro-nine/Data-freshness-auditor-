"""
PipelineEcosystemSimulator
==========================
Generates realistic pipeline run metadata for a 3-layer DAG ecosystem.

Layer 0 — Source tables    : Raw ingestion from external systems (Postgres, APIs, S3)
Layer 1 — Intermediate     : Cleaned, joined, partially modelled tables
Layer 2 — Mart tables      : Business-facing aggregations and reporting tables

Failure modes injected:
  A. Infrastructure Cascade  : One source system outage → correlated downstream failures
  B. Silent Business Failure : Pipeline runs, but rows_written << rows_read (filter bug)
  C. Gradual Latency Drift   : A pipeline gets 3 min slower every week (debt accumulation)
  D. Phantom Recovery        : Pipeline auto-retries but at irregular time (false "ok")
  E. Expected Empty          : Weekend batch has zero rows legitimately
"""

import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# DAG SCHEMA — 22 tables across 3 layers
# ─────────────────────────────────────────────────────────────
DAG_SCHEMA = {
    # Layer 0: Raw sources (4 source systems)
    "raw_orders":          {"layer": 0, "source": "postgres_prod",  "schedule_hour": 1,  "criticality": "HIGH",   "depends_on": []},
    "raw_customers":       {"layer": 0, "source": "postgres_prod",  "schedule_hour": 1,  "criticality": "HIGH",   "depends_on": []},
    "raw_products":        {"layer": 0, "source": "postgres_prod",  "schedule_hour": 1,  "criticality": "MEDIUM", "depends_on": []},
    "raw_inventory":       {"layer": 0, "source": "erp_api",        "schedule_hour": 2,  "criticality": "HIGH",   "depends_on": []},
    "raw_payments":        {"layer": 0, "source": "payment_gateway","schedule_hour": 1,  "criticality": "HIGH",   "depends_on": []},
    "raw_sessions":        {"layer": 0, "source": "clickstream_s3", "schedule_hour": 3,  "criticality": "LOW",    "depends_on": []},
    "raw_support_tickets": {"layer": 0, "source": "zendesk_api",    "schedule_hour": 4,  "criticality": "LOW",    "depends_on": []},

    # Layer 1: Intermediate (processed, joined)
    "stg_orders":          {"layer": 1, "source": "dbt",            "schedule_hour": 3,  "criticality": "HIGH",   "depends_on": ["raw_orders", "raw_customers"]},
    "stg_payments":        {"layer": 1, "source": "dbt",            "schedule_hour": 3,  "criticality": "HIGH",   "depends_on": ["raw_payments", "raw_orders"]},
    "stg_inventory":       {"layer": 1, "source": "dbt",            "schedule_hour": 4,  "criticality": "MEDIUM", "depends_on": ["raw_inventory", "raw_products"]},
    "stg_sessions":        {"layer": 1, "source": "dbt",            "schedule_hour": 5,  "criticality": "LOW",    "depends_on": ["raw_sessions", "raw_customers"]},
    "int_order_payments":  {"layer": 1, "source": "dbt",            "schedule_hour": 5,  "criticality": "HIGH",   "depends_on": ["stg_orders", "stg_payments"]},
    "int_fulfillment":     {"layer": 1, "source": "dbt",            "schedule_hour": 5,  "criticality": "HIGH",   "depends_on": ["stg_orders", "stg_inventory"]},
    "int_customer_ltv":    {"layer": 1, "source": "dbt",            "schedule_hour": 6,  "criticality": "MEDIUM", "depends_on": ["stg_orders", "stg_sessions", "stg_payments"]},

    # Layer 2: Mart (business-facing)
    "mart_revenue":        {"layer": 2, "source": "dbt",            "schedule_hour": 7,  "criticality": "CRITICAL","depends_on": ["int_order_payments"]},
    "mart_inventory_health":{"layer":2, "source": "dbt",            "schedule_hour": 7,  "criticality": "HIGH",   "depends_on": ["int_fulfillment", "stg_inventory"]},
    "mart_customer_360":   {"layer": 2, "source": "dbt",            "schedule_hour": 8,  "criticality": "HIGH",   "depends_on": ["int_customer_ltv", "int_order_payments"]},
    "mart_ops_dashboard":  {"layer": 2, "source": "dbt",            "schedule_hour": 8,  "criticality": "CRITICAL","depends_on": ["mart_revenue", "mart_inventory_health"]},
    "mart_support_kpis":   {"layer": 2, "source": "dbt",            "schedule_hour": 8,  "criticality": "MEDIUM", "depends_on": ["int_customer_ltv", "raw_support_tickets"]},
    "mart_exec_summary":   {"layer": 2, "source": "dbt",            "schedule_hour": 9,  "criticality": "CRITICAL","depends_on": ["mart_revenue", "mart_customer_360", "mart_ops_dashboard"]},
    "mart_finance_close":  {"layer": 2, "source": "spark_job",      "schedule_hour": 6,  "criticality": "CRITICAL","depends_on": ["stg_payments", "int_order_payments"]},
    "mart_churn_signals":  {"layer": 2, "source": "ml_pipeline",    "schedule_hour": 7,  "criticality": "MEDIUM", "depends_on": ["int_customer_ltv", "stg_sessions"]},
}

# ─────────────────────────────────────────────────────────────
# BASELINE LATENCY PROFILES (minutes)
# Each table has a realistic mean and std dev,
# plus weekday/weekend multipliers and hour-of-day sensitivity
# ─────────────────────────────────────────────────────────────
LATENCY_PROFILES = {
    "raw_orders":           {"mean": 18, "std": 3.2, "weekend_mult": 0.6},
    "raw_customers":        {"mean": 12, "std": 2.1, "weekend_mult": 0.5},
    "raw_products":         {"mean": 8,  "std": 1.5, "weekend_mult": 1.0},
    "raw_inventory":        {"mean": 35, "std": 6.8, "weekend_mult": 0.3},
    "raw_payments":         {"mean": 22, "std": 4.1, "weekend_mult": 0.7},
    "raw_sessions":         {"mean": 55, "std": 9.2, "weekend_mult": 1.3},
    "raw_support_tickets":  {"mean": 28, "std": 5.5, "weekend_mult": 0.4},
    "stg_orders":           {"mean": 14, "std": 2.8, "weekend_mult": 0.6},
    "stg_payments":         {"mean": 16, "std": 3.1, "weekend_mult": 0.7},
    "stg_inventory":        {"mean": 20, "std": 4.0, "weekend_mult": 0.4},
    "stg_sessions":         {"mean": 38, "std": 7.0, "weekend_mult": 1.2},
    "int_order_payments":   {"mean": 25, "std": 4.5, "weekend_mult": 0.6},
    "int_fulfillment":      {"mean": 22, "std": 4.2, "weekend_mult": 0.4},
    "int_customer_ltv":     {"mean": 45, "std": 8.5, "weekend_mult": 0.8},
    "mart_revenue":         {"mean": 18, "std": 3.5, "weekend_mult": 0.5},
    "mart_inventory_health":{"mean": 15, "std": 2.9, "weekend_mult": 0.4},
    "mart_customer_360":    {"mean": 30, "std": 5.5, "weekend_mult": 0.7},
    "mart_ops_dashboard":   {"mean": 12, "std": 2.2, "weekend_mult": 0.5},
    "mart_support_kpis":    {"mean": 20, "std": 3.8, "weekend_mult": 0.3},
    "mart_exec_summary":    {"mean": 25, "std": 4.5, "weekend_mult": 0.4},
    "mart_finance_close":   {"mean": 48, "std": 9.1, "weekend_mult": 0.2},
    "mart_churn_signals":   {"mean": 65, "std": 12.0,"weekend_mult": 0.9},
}

ROW_PROFILES = {
    t: {
        "base_rows": int(np.random.default_rng(i).integers(5000, 500000)),
        "weekend_mult": DAG_SCHEMA[t].get("weekend_mult", 0.6) if "weekend_mult" in DAG_SCHEMA.get(t,{}) else 0.7
    }
    for i, t in enumerate(DAG_SCHEMA)
}

class PipelineEcosystemSimulator:

    def __init__(self, n_days: int = 180, seed: int = 42):
        self.n_days   = n_days
        self.rng      = np.random.default_rng(seed)
        self.tables   = list(DAG_SCHEMA.keys())
        self.dag      = self._build_dag()
        self.start_dt = datetime(2024, 1, 1)

    # ── DAG construction ────────────────────────────────────
    def _build_dag(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for table, meta in DAG_SCHEMA.items():
            G.add_node(table, **meta)
            for dep in meta["depends_on"]:
                G.add_edge(dep, table)
        assert nx.is_directed_acyclic_graph(G), "DAG has cycles — check schema"
        return G

    # ── Baseline latency with seasonal pattern ───────────────
    def _baseline_latency(self, table: str, day_of_week: int,
                           day_idx: int) -> float:
        p   = LATENCY_PROFILES[table]
        mu  = p["mean"]
        sig = p["std"]
        # Weekend reduction
        if day_of_week >= 5:
            mu *= p["weekend_mult"]
            sig *= 0.7
        # Monthly trend: slight growth over time (pipeline debt accumulation)
        mu += (day_idx / self.n_days) * 0.05 * p["mean"]
        return max(1.0, self.rng.normal(mu, sig))

    # ── Row count simulation ─────────────────────────────────
    def _simulate_rows(self, table: str, day_of_week: int) -> tuple:
        base = ROW_PROFILES[table]["base_rows"]
        wk_m = ROW_PROFILES[table]["weekend_mult"]
        mult = wk_m if day_of_week >= 5 else 1.0
        noise = self.rng.normal(1.0, 0.08)
        rows_read    = max(0, int(base * mult * noise))
        # Small loss rate: 0.001–0.01%
        loss_rate    = self.rng.uniform(0.0001, 0.0001)
        rows_written = max(0, int(rows_read * (1 - loss_rate)))
        return rows_read, rows_written

    # ── Core run record builder ──────────────────────────────
    def _build_run(self, table: str, run_date: datetime,
                    day_idx: int) -> dict:
        meta      = DAG_SCHEMA[table]
        sched_dt  = run_date.replace(hour=meta["schedule_hour"], minute=0, second=0)
        dow       = run_date.weekday()
        latency   = self._baseline_latency(table, dow, day_idx)
        start_lag = self.rng.uniform(0.5, 2.0)  # startup overhead
        started   = sched_dt + timedelta(minutes=start_lag)
        completed = started  + timedelta(minutes=latency)
        rows_r, rows_w = self._simulate_rows(table, dow)

        return {
            "run_id":          f"{table}_{run_date.strftime('%Y%m%d')}",
            "table_name":      table,
            "pipeline_name":   f"{table}_pipeline",
            "source_system":   meta["source"],
            "layer":           meta["layer"],
            "criticality":     meta["criticality"],
            "scheduled_at":    sched_dt,
            "started_at":      started,
            "completed_at":    completed,
            "latency_minutes": round(latency, 3),
            "rows_read":       rows_r,
            "rows_written":    rows_w,
            "execution_status":"SUCCESS",
            "failure_mode":    "NONE",
            "day_of_week":     dow,
            "is_weekend":      int(dow >= 5),
            "day_idx":         day_idx,
        }

    # ── FAILURE MODE INJECTORS ──────────────────────────────

    def _inject_infrastructure_cascade(self, runs: list,
                                        event_days: list) -> list:
        """
        Failure Mode A: postgres_prod goes down for 3–5 hours.
        All tables sourced from postgres_prod fail.
        Downstream tables get delayed (not failed) by expected propagation.
        """
        affected_sources = {"postgres_prod"}
        affected_tables  = [t for t, m in DAG_SCHEMA.items()
                             if m["source"] in affected_sources]

        for record in runs:
            if record["day_idx"] in event_days:
                if record["table_name"] in affected_tables:
                    record["execution_status"] = "FAILED"
                    record["rows_written"]      = 0
                    record["rows_read"]         = 0
                    record["failure_mode"]      = "INFRA_CASCADE"
                    record["latency_minutes"]  += self.rng.uniform(180, 300)
        return runs

    def _inject_silent_business_failure(self, runs: list,
                                         table: str, event_days: list) -> list:
        """
        Failure Mode B: Pipeline runs successfully (status=SUCCESS)
        but rows_written drops to near-zero due to a filter bug.
        This is the hardest failure to detect with row-count alone.
        """
        for record in runs:
            if record["table_name"] == table and record["day_idx"] in event_days:
                record["rows_written"]  = self.rng.integers(0, 5)
                record["failure_mode"]  = "SILENT_BUSINESS_FAILURE"
                # execution_status stays SUCCESS — this is the trap
        return runs

    def _inject_gradual_drift(self, runs: list, table: str,
                               drift_rate_per_day: float = 3.0) -> list:
        """
        Failure Mode C: Latency grows linearly over time.
        Starts imperceptibly, becomes catastrophic by end of window.
        Classic CUSUM detection case — Z-score misses this entirely.
        """
        drift_start_day = self.n_days // 3  # drift starts at day 60
        for record in runs:
            if (record["table_name"] == table
                    and record["day_idx"] >= drift_start_day):
                days_drifting = record["day_idx"] - drift_start_day
                record["latency_minutes"] += days_drifting * drift_rate_per_day
                record["failure_mode"]     = "GRADUAL_DRIFT"
                record["completed_at"]    += timedelta(
                    minutes=days_drifting * drift_rate_per_day)
        return runs

    def _inject_phantom_recovery(self, runs: list,
                                   table: str, event_days: list) -> list:
        """
        Failure Mode D: Pipeline fails, auto-retries at irregular time.
        Row counts are correct but timing window is violated.
        System must not flag this as recovered if retry window is abnormal.
        """
        for record in runs:
            if record["table_name"] == table and record["day_idx"] in event_days:
                retry_delay = self.rng.uniform(90, 240)
                record["latency_minutes"] += retry_delay
                record["completed_at"]   += timedelta(minutes=retry_delay)
                record["failure_mode"]    = "PHANTOM_RECOVERY"
                record["execution_status"]= "RETRIED"
        return runs

    def _inject_expected_empty(self, runs: list) -> list:
        """
        Failure Mode E: Weekend sessions data is legitimately empty.
        System must distinguish from silent failure.
        """
        for record in runs:
            if (record["table_name"] == "raw_sessions"
                    and record["is_weekend"] == 1):
                record["rows_read"]    = 0
                record["rows_written"] = 0
                record["failure_mode"] = "EXPECTED_EMPTY"
        return runs

    # ── MAIN GENERATE METHOD ─────────────────────────────────
    def generate(self) -> tuple:
        runs = []
        for day_idx in range(self.n_days):
            run_date = self.start_dt + timedelta(days=day_idx)
            for table in self.tables:
                run = self._build_run(table, run_date, day_idx)
                runs.append(run)

        # ── Inject all failure modes ──
        # A: Infrastructure cascade on days 30, 95, 142
        runs = self._inject_infrastructure_cascade(runs, [30, 95, 142])
        # B: Silent business failure on stg_payments days 55–62
        runs = self._inject_silent_business_failure(runs, "stg_payments",
                                                     list(range(55, 63)))
        # C: Gradual drift on raw_sessions (from day 60 onward)
        runs = self._inject_gradual_drift(runs, "raw_sessions",
                                           drift_rate_per_day=2.5)
        # D: Phantom recovery on mart_finance_close days 110, 111
        runs = self._inject_phantom_recovery(runs, "mart_finance_close",
                                              [110, 111])
        # E: Expected empty weekends on raw_sessions
        runs = self._inject_expected_empty(runs)

        df  = pd.DataFrame(runs)
        dag = self.dag
        return df, dag


if __name__ == "__main__":
    import os
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sim = PipelineEcosystemSimulator(n_days=180, seed=42)
    df, dag = sim.generate()
    df.to_csv(os.path.join(_ROOT, "data", "pipeline_runs.csv"),
              index=False)
    print(f"Generated {len(df):,} pipeline run records")
    print(f"Tables: {df['table_name'].nunique()} | Days: {df['day_idx'].nunique()}")
    print(f"\nFailure mode distribution:")
    print(df["failure_mode"].value_counts().to_string())
    print(f"\nSchema:\n{df.dtypes}")
