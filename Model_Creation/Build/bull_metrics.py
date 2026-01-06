# bull_metrics.py
"""
Feature‑engineering utilities that replicate the R metrics functions:
  * calculate_bull_data
  * calculate_bull_data_by_hand
  * calculate_rider_data

All functions accept a **pandas.DataFrame** (the `rides_data` we produced in
`build_dataset.py`) and return a new DataFrame with engineered features.

These utilities are intentionally written with **pure‑pandas / numpy** so they
run anywhere.  If you have a huge dataset you may eventually port these to
polars or PySpark – but pandas keeps the translation simple for now.

USAGE EXAMPLE
-------------
>>> import pandas as pd
>>> from bull_metrics import calculate_bull_data, calculate_bull_data_by_hand, calculate_rider_data
>>> # assuming ROOT points to the repo root folder
>>> # rides = pd.read_csv(ROOT/"Data/Processed/rides_data.csv")
>>> # bull_overall = calculate_bull_data(rides, range=10)
>>> # bull_hand    = calculate_bull_data_by_hand(rides, range=10)
>>> # rider_df     = calculate_rider_data(rides)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

__all__ = [
    "calculate_bull_data",
    "calculate_bull_data_by_hand",
    "calculate_rider_data",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K_PRIOR = 10  # Strength of league-wide prior for smoothing

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    """Safe division that handles zero denominators."""
    denom = denom.replace(0, np.nan)
    return num / denom

def _calculate_smooth_ars(df: pd.DataFrame, grp, mask_score_ok: pd.Series, 
                         range: int = 10) -> tuple[pd.Series, pd.Series]:
    """
    Calculate smooth ARS stats using league priors and K_PRIOR weighting.
    
    This addresses the issue where unridden bulls get NaN ARS values,
    which can be misleading for bulls with small sample sizes.
    
    Parameters
    ----------
    df : DataFrame with ride data
    grp : GroupBy object for bulls
    mask_score_ok : Boolean mask for valid scores
    range : Rolling window size for short-term stats
    
    Returns
    -------
    tuple : (smooth_ars_cum, smooth_ars_short)
    """
    # Calculate league-wide average rider score on qualified rides
    league_ars_mean = (
        df.loc[mask_score_ok & (df["qr"] == 1), "rider_score"].mean()
        if (mask_score_ok & (df["qr"] == 1)).any() else 85.0  # fallback
    )
    
    # Calculate cumulative smooth ARS
    rider_score_ok = df["rider_score"].where(mask_score_ok, 0)
    cum_rs = grp["rider_score"].apply(lambda s: rider_score_ok.loc[s.index].cumsum()).shift(1).fillna(0)
    cum_qr = grp["qr"].cumsum().shift(1).fillna(0)
    
    # Smooth cumulative ARS using league prior
    smooth_ars_cum = (
        (cum_rs + K_PRIOR * league_ars_mean) / 
        (cum_qr + K_PRIOR)
    )
    
    # Calculate rolling smooth ARS for short-term stats
    def _rolling_smooth_ars(x: pd.Series) -> pd.Series:
        mask = mask_score_ok.loc[x.index]
        routed = x.where(mask, 0)
        # Rolling sum of rider_score*qr and qr counts within window
        rolling_sum = (routed * df.loc[x.index, "qr"]).rolling(range, min_periods=1).sum().shift(1)
        rolling_cnt = df.loc[x.index, "qr"].rolling(range, min_periods=1).sum().shift(1)
        
        # Apply smoothing to rolling ARS
        smooth_rolling_ars = (
            (rolling_sum + K_PRIOR * league_ars_mean) / 
            (rolling_cnt + K_PRIOR)
        )
        return smooth_rolling_ars
    
    smooth_ars_short = grp["rider_score"].apply(_rolling_smooth_ars)
    
    return smooth_ars_cum, smooth_ars_short

# ---------------------------------------------------------------------------
# Bull‑centric metrics
# ---------------------------------------------------------------------------

def calculate_bull_data(df: pd.DataFrame, range: int = 10) -> pd.DataFrame:  # noqa: A002
    """Compute cumulative & recent metrics for *every* bull (all hands).

    Parameters
    ----------
    df : rides_data after `build_dataset.py`
    range : size of the recent window (default 10 outs)  [= Bull_Short]

    Returns
    -------
    DataFrame with one row per *ride* (same number of rows as `df`).
    """
    print(f"[bull_metrics] Computing bull data for {len(df):,} rides with range={range}")
    
    work = (
        df
        .sort_values(["bull_id", "out"], ascending=[True, True])
        .copy()
    )

    # groupby so we can use expanding window ops
    grp = work.groupby("bull_id", group_keys=False)
    
    # cumulative counts & sums *BEFORE* current ride (shift(1))
    cum_n   = grp.cumcount()           # 0‑based index of each ride within bull history
    cum_qr  = grp["qr"].cumsum().shift(1).fillna(0)
    cum_pbr = grp["pbr"].cumsum().shift(1).fillna(0)

    work["new_bull_flag"]   = (work["out"] == 1).astype(int)
    work["few_rides_flag"] = (work["out"] <= range) & (work["out"] != 1)

    # cumulative rates
    work["b_qrp"]      = _safe_div(cum_qr, cum_n)
    work["b_pbr_rate"] = _safe_div(cum_pbr, cum_n)

    # cumulative averages (only if score_flag == 0)
    mask_score_ok = (work["score_flag"] == 0)
    rider_score_ok = work["rider_score"].where(mask_score_ok, 0)
    bull_score_ok  = work["bull_score"].where(mask_score_ok, 0)

    cum_rs = grp["rider_score"].apply(lambda s: rider_score_ok.loc[s.index].cumsum()).shift(1).fillna(0)
    cum_bs = grp["bull_score"].apply(lambda s: bull_score_ok.loc[s.index].cumsum()).shift(1).fillna(0)

    # Calculate smooth ARS stats using league priors
    smooth_ars_cum, smooth_ars_short = _calculate_smooth_ars(work, grp, mask_score_ok, range)

    work["b_ars"] = smooth_ars_cum  # smooth avg rider score on qualified rides
    work["b_abs"] = _safe_div(cum_bs, cum_n)   # avg bull score overall

    # rolling window – last `range` outs
    roll_qr = grp["qr"].rolling(range, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
    roll_bs = grp["bull_score"].rolling(range, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)

    work["b_qrp_short"] = roll_qr
    work["b_abs_short"] = roll_bs
    work["b_ars_short"] = smooth_ars_short

    # replace short‑term NAs for brand‑new bulls
    for col in ["b_qrp_short", "b_abs_short", "b_ars_short"]:
        work.loc[work["few_rides_flag"], col] = work.loc[work["few_rides_flag"], col.replace("_short", "")]

    print(f"[bull_metrics] Completed bull data: {len(work):,} rows with smooth ARS stats")
    
    return work[[
        "master_id", "bull_id", "out", "new_bull_flag", "few_rides_flag",
        "b_ars", "b_abs", "b_qrp", "b_qrp_short", "b_abs_short", "b_ars_short", "pbr", "b_pbr_rate"
    ]]


def calculate_bull_data_by_hand(df: pd.DataFrame, range: int = 10) -> pd.DataFrame:  # noqa: A002
    """Same as `calculate_bull_data` but separate stats per *hand*."""
    print(f"[bull_metrics] Computing hand-based bull data for {len(df):,} rides with range={range}")
    
    work = (
        df
        .sort_values(["bull_id", "out"], ascending=[True, True])
        .copy()
    )
    grp = work.groupby(["bull_id", "hand"], group_keys=False)

    cum_n   = grp.cumcount()
    cum_qr  = grp["qr"].cumsum().shift(1).fillna(0)

    mask_score_ok = (work["score_flag"] == 0)
    rider_score_ok = work["rider_score"].where(mask_score_ok, 0)
    bull_score_ok  = work["bull_score"].where(mask_score_ok, 0)

    cum_rs = grp["rider_score"].apply(lambda s: rider_score_ok.loc[s.index].cumsum()).shift(1).fillna(0)
    cum_bs = grp["bull_score"].apply(lambda s: bull_score_ok.loc[s.index].cumsum()).shift(1).fillna(0)

    work["few_rides_flag_hand"] = (cum_n < (range // 2))
    work["h_qrp"] = _safe_div(cum_qr, cum_n)
    work["h_abs"] = _safe_div(cum_bs, cum_n)
    
    # Calculate smooth hand-based ARS stats using league priors
    smooth_h_ars_cum, smooth_h_ars_short = _calculate_smooth_ars(work, grp, mask_score_ok, range // 2)
    work["h_ars"] = smooth_h_ars_cum

    roll_qr = grp["qr"].rolling(range // 2, min_periods=1).mean().shift(1).reset_index(level=[0,1], drop=True)
    roll_bs = grp["bull_score"].rolling(range // 2, min_periods=1).mean().shift(1).reset_index(level=[0,1], drop=True)

    work["h_qrp_short"] = roll_qr
    work["h_abs_short"] = roll_bs
    work["h_ars_short"] = smooth_h_ars_short

    # fix for few rides
    for col in ["h_qrp_short", "h_abs_short", "h_ars_short"]:
        work.loc[work["few_rides_flag_hand"], col] = work.loc[work["few_rides_flag_hand"], col.replace("_short", "")]

    print(f"[bull_metrics] Completed hand-based bull data: {len(work):,} rows with smooth ARS stats")
    
    return work[[
        "master_id", "bull_id", "out", "hand", "few_rides_flag_hand",
        "h_ars", "h_abs", "h_qrp", "h_qrp_short", "h_abs_short", "h_ars_short"
    ]]
