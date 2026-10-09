"""
Offline trend-based drift audit
===============================
v1 of engineer.py ran CUSUM on the STL *residual*. STL absorbs slow movements
into its trend, so on the simulated ``raw_sessions`` drift the trend rose from
about 60 to 347 minutes while the residual mean stayed near zero, and the
residual-CUSUM first alerted on day 143 for a drift that began on day 60.

engineer.py v2 detects drift causally with an EWMA chart. This module remains as
an offline cross-check on the STL trend and needs
``FreshnessFeatureEngineer(..., include_stl_audit=True)``. A table is flagged
when the trend sits more than ``z_threshold`` reference-residual standard
deviations above its reference level for ``min_run`` consecutive days.

Limits: STL's trend is a centred smoother, so this uses data on both sides of
each day and is not a real-time alert. ``z_threshold`` was set with a margin
above the largest trend excursion on non-drifting simulated tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trend_drift_alert(trend, residual, ref_len: int = 45, z_threshold: float = 5.0, min_run: int = 3) -> dict:
    """Return drift status for one table's STL trend and residual."""
    trend = np.asarray(trend, dtype=float)
    residual = np.asarray(residual, dtype=float)
    if len(trend) != len(residual):
        raise ValueError("trend and residual must have the same length")
    if len(trend) < ref_len + min_run:
        raise ValueError("series is shorter than the reference window plus min_run")

    sd = float(np.std(residual[:ref_len], ddof=1))
    sd = sd if sd > 0 else 1e-9
    z = (trend - trend[:ref_len].mean()) / sd

    above = (z > z_threshold).astype(int)
    run_hits = np.convolve(above, np.ones(min_run, dtype=int), mode="valid") >= min_run
    hits = np.flatnonzero(run_hits)
    return {
        "drift_detected": bool(len(hits)),
        "first_alert_index": int(hits.min()) if len(hits) else None,
        "peak_trend_z": float(z.max()),
    }


def detect_trend_drift(features: pd.DataFrame, ref_len: int = 45, z_threshold: float = 5.0,
                       min_run: int = 3) -> pd.DataFrame:
    """Apply ``trend_drift_alert`` to every table in the feature matrix."""
    needed = {"table_name", "day_idx", "stl_trend", "stl_residual"}
    missing = needed - set(features.columns)
    if missing:
        raise KeyError(f"feature matrix is missing columns: {sorted(missing)}")

    rows = []
    for table, grp in features.groupby("table_name"):
        grp = grp.sort_values("day_idx").reset_index(drop=True)
        res = trend_drift_alert(grp["stl_trend"], grp["stl_residual"], ref_len, z_threshold, min_run)
        first = res["first_alert_index"]
        rows.append({
            "table_name": table,
            "drift_detected": res["drift_detected"],
            "first_alert_day": int(grp.loc[first, "day_idx"]) if first is not None else None,
            "peak_trend_z": round(res["peak_trend_z"], 2),
        })
    return pd.DataFrame(rows).sort_values("peak_trend_z", ascending=False).reset_index(drop=True)
