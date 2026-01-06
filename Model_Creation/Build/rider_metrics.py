# rider_metrics.py
"""
Rider-centric feature engineering utilities.

Public function:
    • calculate_rider_data(df, short_rides=25, long_days=1095, w1=0.15, id_col="rider_internal_id")

Output:
    One row per input ride (keyed by master_id) with rider-side features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable

__all__ = ["calculate_rider_data"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mean(x: pd.Series) -> float | np.floating | float:
    x = x.dropna()
    return float(x.mean()) if not x.empty else np.nan

def _require_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"rider_metrics.calculate_rider_data: df missing required columns: {missing}")

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def calculate_rider_data(
    df: pd.DataFrame,
    *,
    short_rides: int = 25,
    long_days: int = 1095,
    w1: float = 0.15,
    id_col: str = "rider_internal_id",
    new_flag_n: int = 20,   # how many first rides to flag as "new"
) -> pd.DataFrame:
    """Compute rider rolling/weighted metrics (QRP, ARS, PBR-rate, ROE flags).

    Parameters
    ----------
    df : DataFrame from build_dataset.py (must include at least:
         master_id, ride_id, event_start_date, qr, rider_score, pbr, and id_col)
    short_rides : window size for short-term form (default 25 rides)
    long_days : trailing days considered long-term (≈ 3 years)
    w1 : weight for short-term QRP in weighted QRP (long-term weight = 1 - w1)
    id_col : which rider identifier to group by (default "rider_internal_id")
    new_flag_n : mark first N rides for a rider as new (default 20)

    Returns
    -------
    DataFrame with one row per master_id and columns:
      master_id, id_col, ride_id, event_start_date, new_rider_flag,
      r_qrp_long, r_ars_long, r_qrp_short, r_ars_short,
      r_qrp_weighted, r_pbr_rate
    """
    # Required columns
    _require_cols(df, [id_col, "master_id", "ride_id", "event_start_date", "qr", "rider_score", "pbr"])

    # Ensure date dtype
    work = df.copy()
    work["event_start_date"] = pd.to_datetime(work["event_start_date"], errors="coerce")

    # Sort for causal windows
    work = work.sort_values([id_col, "event_start_date", "ride_id"]).reset_index(drop=True)

    out_blocks: list[pd.DataFrame] = []

    # Group per rider
    for rid, sub in work.groupby(id_col, sort=False):
        sub = sub.reset_index(drop=True)
        n = len(sub)

        # Pre-allocate
        r_qrp_short = np.full(n, np.nan, dtype=float)
        r_qrp_long  = np.full(n, np.nan, dtype=float)
        r_ars_short = np.full(n, np.nan, dtype=float)
        r_ars_long  = np.full(n, np.nan, dtype=float)
        r_weighted  = np.full(n, np.nan, dtype=float)
        r_pbr_rate  = np.full(n, np.nan, dtype=float)
        new_flag    = np.zeros(n, dtype=int)

        dates = sub["event_start_date"]
        qr    = sub["qr"].astype(float)
        pbr   = sub["pbr"].astype(float)
        rsc   = sub["rider_score"].astype(float)

        for i in range(n):
            curr_date = dates.iat[i]

            # Prior mask (strictly before current ride)
            prior_mask = dates < curr_date

            # Long window within trailing 'long_days'
            if pd.notna(curr_date):
                long_mask = prior_mask & (dates >= curr_date - pd.Timedelta(days=long_days))
            else:
                long_mask = pd.Series(False, index=sub.index)

            # Long-term QRP/ARS/PBR over long window; if empty, fall back to "all previous"
            if long_mask.any():
                idx = np.where(long_mask)[0]
            elif prior_mask.any():
                idx = np.where(prior_mask)[0]
            else:
                idx = np.array([], dtype=int)

            if idx.size > 0:
                r_qrp_long[i] = float(qr.iloc[idx].mean())
                r_ars_long[i] = _safe_mean(rsc.iloc[idx][qr.iloc[idx] == 1])
                r_pbr_rate[i] = float(pbr.iloc[idx].mean())

            # Short-term over last `short_rides` previous rides
            if i > 0:
                start_idx = max(0, i - short_rides)
                short_slice = slice(start_idx, i)  # [start, i)
                qr_short  = qr.iloc[short_slice]
                rsc_short = rsc.iloc[short_slice]
                if not qr_short.empty:
                    r_qrp_short[i] = float(qr_short.mean())
                    r_ars_short[i] = _safe_mean(rsc_short[qr_short == 1])

            # Weighted QRP: prefer both; else fall back to whichever exists
            short_val = r_qrp_short[i]
            long_val  = r_qrp_long[i]
            if not np.isnan(short_val) and not np.isnan(long_val):
                r_weighted[i] = w1 * short_val + (1 - w1) * long_val
            elif not np.isnan(long_val):
                r_weighted[i] = long_val
            elif not np.isnan(short_val):
                r_weighted[i] = short_val
            # else remains NaN for very first ride(s)

            new_flag[i] = 1 if i < new_flag_n else 0

        out = pd.DataFrame({
            "master_id": sub["master_id"].values,
            id_col: rid,
            "ride_id": sub["ride_id"].values,
            "event_start_date": dates.values,
            "new_rider_flag": new_flag,
            "r_qrp_long": r_qrp_long,
            "r_ars_long": r_ars_long,
            "r_qrp_short": r_qrp_short,
            "r_ars_short": r_ars_short,
            "r_qrp_weighted": r_weighted,
            "r_pbr_rate": r_pbr_rate,
        })
        out_blocks.append(out)

    result = pd.concat(out_blocks, ignore_index=True)

    # Keep exact row count parity with input (1:1 on master_id)
    assert result["master_id"].nunique() == df["master_id"].nunique(), \
        "Output is not one row per input ride (master_id)."

    return result
