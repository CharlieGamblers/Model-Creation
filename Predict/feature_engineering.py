# feature_engineering.py

import pandas as pd
import numpy as np
from .config import (
    RIDER_LONG_DAYS,
    RIDER_SHORT_RIDES,
    BULL_SHORT,
    W1,
    W2,
    K_PRIOR
)
from pathlib import Path

def _safe_mean(x: pd.Series) -> float:
    """Return the mean of a Series, ignoring NaNs. If empty, return NaN."""
    return x.mean(skipna=True) if x.size else np.nan


def solo_data_pull(
    df: pd.DataFrame,
    rider_id: int | None,
    bull_id: int,
    *,
    rider_internal_id: int | None = None,
    long_range_days: int = RIDER_LONG_DAYS,
    short_rides: int = RIDER_SHORT_RIDES,
    bull_short_range: int = BULL_SHORT,
    start_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Return a single-row DataFrame of engineered features for a given rider/bull matchup,
    using the same logic as training.

    Backward-compatible:
        - Prefer `rider_internal_id` (new pipeline).
        - Falls back to legacy `rider_id` if internal id is not provided.
    """
    # Filter to strictly prior data
    df_hist = df[df["event_start_date"] < pd.to_datetime(start_date)]

    # ---------------------------
    # Rider and bull histories
    # ---------------------------
    use_internal = (rider_internal_id is not None) and ("rider_internal_id" in df_hist.columns)
    if use_internal:
        df_rider_all = df_hist[df_hist["rider_internal_id"] == rider_internal_id].sort_values("event_start_date")
    else:
        df_rider_all = df_hist[df_hist["rider_id"] == rider_id].sort_values("event_start_date")

    df_bull_all = df_hist[df_hist["bull_id"] == bull_id].sort_values("out")

    # League average QR for smoothing
    league_qr_mean = df_hist["qr"].mean() if not df_hist.empty else 0.26
    b_qrp_smooth = (
        (df_bull_all["qr"].sum() + K_PRIOR * league_qr_mean) /
        (len(df_bull_all) + K_PRIOR)
    ) if (len(df_bull_all) + K_PRIOR) > 0 else league_qr_mean

    # ---------------------------
    # Rider hand (robust inference)
    # ---------------------------
    if "hand" in df_rider_all.columns and not df_rider_all.empty:
        hand_series = df_rider_all["hand"].dropna().astype(str).str.upper().str.strip()
        rider_hand = hand_series.mode().iat[0] if not hand_series.empty else np.nan
        if rider_hand not in ("L", "R"):
            rider_hand = np.nan
    else:
        rider_hand = np.nan

    # Hand-specific bull subsets
    df_bull_short = df_bull_all.nlargest(bull_short_range, "out")
    if rider_hand in ("L", "R"):
        df_bull_hand = df_bull_all[df_bull_all["hand"].astype(str).str.upper() == rider_hand]
        df_bull_hand_short = df_bull_hand.nlargest(bull_short_range, "out")
    else:
        df_bull_hand = pd.DataFrame(columns=df_bull_all.columns)
        df_bull_hand_short = pd.DataFrame(columns=df_bull_all.columns)

    # Recent rider subset
    df_rider_short = df_rider_all.tail(short_rides)

    # ---------------------------
    # ROE temp (robust fallbacks)
    # ---------------------------
    if "r_ROE_temp" in df_rider_all.columns:
        new_rider_roe_temp = pd.to_numeric(df_rider_all["r_ROE_temp"], errors="coerce")
    else:
        qr_r = pd.to_numeric(df_rider_all.get("qr", pd.Series(dtype=float)), errors="coerce")
        b_qrp_r = pd.to_numeric(df_rider_all.get("b_qrp", pd.Series(dtype=float)), errors="coerce")
        new_rider_roe_temp = qr_r - b_qrp_r

    if "b_ROE_temp" in df_bull_all.columns:
        new_bull_roe_temp = pd.to_numeric(df_bull_all["b_ROE_temp"], errors="coerce")
    else:
        r_qrp_long_b = pd.to_numeric(df_bull_all.get("r_qrp_long", pd.Series(dtype=float)), errors="coerce")
        qr_b = pd.to_numeric(df_bull_all.get("qr", pd.Series(dtype=float)), errors="coerce")
        new_bull_roe_temp = r_qrp_long_b - qr_b

    # ---------------------------
    # Ride & out sequence values
    # ---------------------------
    out_value = (df_bull_all["out"].max() if not df_bull_all.empty else 0) + 1
    ride_value = (df_rider_all["ride_id"].max() if "ride_id" in df.columns and not df_rider_all.empty else 0) + 1

    # ---------------------------
    # Rider weighted QRP (smoothed with 30 rides at league average)
    # ---------------------------
    RIDER_QRP_PRIOR = 30  # Number of virtual rides at league average for smoothing
    def _case_weighted_qrp() -> float:
        # Smooth short-term QRP
        rider_short_qr_sum = df_rider_short["qr"].sum() if not df_rider_short.empty else 0
        rider_short_count = len(df_rider_short)
        smooth_short = (
            (rider_short_qr_sum + RIDER_QRP_PRIOR * league_qr_mean) /
            (rider_short_count + RIDER_QRP_PRIOR)
        ) if (rider_short_count + RIDER_QRP_PRIOR) > 0 else league_qr_mean
        
        # Smooth long-term QRP
        rider_all_qr_sum = df_rider_all["qr"].sum() if not df_rider_all.empty else 0
        rider_all_count = len(df_rider_all)
        smooth_all = (
            (rider_all_qr_sum + RIDER_QRP_PRIOR * league_qr_mean) /
            (rider_all_count + RIDER_QRP_PRIOR)
        ) if (rider_all_count + RIDER_QRP_PRIOR) > 0 else league_qr_mean
        
        # Weighted combination
        return W1 * smooth_short + W2 * smooth_all

    # ---------------------------
    # Smooth ARS using league priors
    # ---------------------------
    league_ars_mean = df_hist.loc[df_hist["qr"] == 1, "rider_score"].mean() if not df_hist.empty else 85.0

    # Smooth bull ARS (all hands)
    bull_qr_sum = df_bull_all["qr"].sum()
    b_ars_smooth = (
        (df_bull_all.loc[df_bull_all["qr"] == 1, "rider_score"].sum() + K_PRIOR * league_ars_mean) /
        (bull_qr_sum + K_PRIOR)
    ) if bull_qr_sum > 0 else league_ars_mean

    # Smooth short-term bull ARS
    bull_short_qr_sum = df_bull_short["qr"].sum()
    b_ars_short_smooth = (
        (df_bull_short.loc[df_bull_short["qr"] == 1, "rider_score"].sum() + K_PRIOR * league_ars_mean) /
        (bull_short_qr_sum + K_PRIOR)
    ) if bull_short_qr_sum > 0 else league_ars_mean

    # Smooth hand-based ARS
    if not df_bull_hand.empty:
        hand_qr_sum = df_bull_hand["qr"].sum()
        h_ars_smooth = (
            (df_bull_hand.loc[df_bull_hand["qr"] == 1, "rider_score"].sum() + K_PRIOR * league_ars_mean) /
            (hand_qr_sum + K_PRIOR)
        ) if hand_qr_sum > 0 else league_ars_mean
    else:
        h_ars_smooth = np.nan

    # Smooth short-term hand-based ARS
    if not df_bull_hand_short.empty:
        hand_short_qr_sum = df_bull_hand_short["qr"].sum()
        h_ars_short_smooth = (
            (df_bull_hand_short.loc[df_bull_hand_short["qr"] == 1, "rider_score"].sum() + K_PRIOR * league_ars_mean) /
            (hand_short_qr_sum + K_PRIOR)
        ) if hand_short_qr_sum > 0 else league_ars_mean
    else:
        h_ars_short_smooth = np.nan

    # ---------------------------
    # Previous matchups between this rider and bull
    # ---------------------------
    if use_internal:
        prev_matchups = df_hist[
            (df_hist["rider_internal_id"] == rider_internal_id) & 
            (df_hist["bull_id"] == bull_id)
        ]
    else:
        prev_matchups = df_hist[
            (df_hist["rider_id"] == rider_id) & 
            (df_hist["bull_id"] == bull_id)
        ]
    prev_matchups_count = len(prev_matchups)
    prev_rides_sum = prev_matchups["qr"].sum() if not prev_matchups.empty and "qr" in prev_matchups.columns else 0.0

    # ---------------------------
    # Return single-row DataFrame
    # ---------------------------
    return pd.DataFrame([{
        # Include IDs so trainer can exclude them from features
        "rider_internal_id": rider_internal_id if use_internal else np.nan,
        "rider_id": rider_id if not use_internal else np.nan,
        "bull_id": bull_id,

        # Flags
        "new_rider_flag"      : int(ride_value <= 20),
        "new_bull_flag"       : int(out_value  <= 2),
        "few_rides_flag"      : int(out_value  <= BULL_SHORT),
        "few_rides_flag_hand" : int(out_value  <= BULL_SHORT / 2),

        # Rider stats
        "r_ars_long"  : _safe_mean(df_rider_all.loc[df_rider_all["qr"] == 1, "rider_score"]),
        "r_ars_short" : _safe_mean(df_rider_short.loc[df_rider_short["qr"] == 1, "rider_score"]),
        "r_qrp_weighted": _case_weighted_qrp(),
        "r_ROE_positive": np.nan if ride_value <= 5 else _safe_mean(new_rider_roe_temp[new_rider_roe_temp > 0]),
        "r_ROE_negative": np.nan if ride_value <= 5 else _safe_mean(new_rider_roe_temp[new_rider_roe_temp < 0]),
        "r_pbr_rate"  : _safe_mean(df_rider_all["pbr"]),

        # Bull stats (all hands)
        "b_abs"       : _safe_mean(df_bull_all["bull_score"]),
        "b_ars"       : b_ars_smooth,
        "b_qrp_smooth": b_qrp_smooth,
        "b_abs_short" : _safe_mean(df_bull_short["bull_score"]),
        "b_ars_short" : b_ars_short_smooth,

        # Hand-based bull stats
        "h_qrp"       : _safe_mean(df_bull_hand["qr"]) if not df_bull_hand.empty else np.nan,
        "h_abs"       : _safe_mean(df_bull_hand["bull_score"]) if not df_bull_hand.empty else np.nan,
        "h_ars"       : h_ars_smooth,
        "h_qrp_short" : _safe_mean(df_bull_hand_short["qr"]) if not df_bull_hand_short.empty else np.nan,
        "h_abs_short" : _safe_mean(df_bull_hand_short["bull_score"]) if not df_bull_hand_short.empty else np.nan,
        "h_ars_short" : h_ars_short_smooth,

        # *** Add bull ROE flags so they flow into FEATURE_COLS ***
        "b_ROE_positive": np.nan if out_value <= 5 else _safe_mean(new_bull_roe_temp[new_bull_roe_temp > 0]),
        "b_ROE_negative": np.nan if out_value <= 5 else _safe_mean(new_bull_roe_temp[new_bull_roe_temp < 0]),

        # Previous matchup features
        "Prev_Matchups": prev_matchups_count,
        "Previous_Rides": prev_rides_sum,

        # misc
        "out": out_value,
        "pbr": 1,
    }])


def rider_data_pull(
    df: pd.DataFrame,
    rider_id: int,
    *,
    long_range_days: int = RIDER_LONG_DAYS,
    short_rides: int = RIDER_SHORT_RIDES,
    start_date: pd.Timestamp,
) -> pd.DataFrame:
    """Return single-row DataFrame with rider-only engineered features prior to start_date."""
    df_hist = df[df["event_start_date"] < start_date]

    df_rider_all = df_hist.query("rider_id == @rider_id").sort_values("event_start_date")
    df_rider_short = df_rider_all.tail(short_rides)

    # League average QR for smoothing
    league_qr_mean = df_hist["qr"].mean() if not df_hist.empty else 0.26

    # ROE temp (robust)
    if "r_ROE_temp" in df_rider_all.columns:
        new_rider_roe_temp = pd.to_numeric(df_rider_all["r_ROE_temp"], errors="coerce")
    else:
        qr_r = pd.to_numeric(df_rider_all.get("qr", pd.Series(dtype=float)), errors="coerce")
        b_qrp_r = pd.to_numeric(df_rider_all.get("b_qrp", pd.Series(dtype=float)), errors="coerce")
        new_rider_roe_temp = qr_r - b_qrp_r

    ride_value = (df_rider_all["ride_id"].max() if "ride_id" in df.columns and not df_rider_all.empty else 0) + 1

    # Rider weighted QRP (smoothed with 30 rides at league average)
    RIDER_QRP_PRIOR = 30  # Number of virtual rides at league average for smoothing
    def _case_weighted_qrp() -> float:
        # Smooth short-term QRP
        rider_short_qr_sum = df_rider_short["qr"].sum() if not df_rider_short.empty else 0
        rider_short_count = len(df_rider_short)
        smooth_short = (
            (rider_short_qr_sum + RIDER_QRP_PRIOR * league_qr_mean) /
            (rider_short_count + RIDER_QRP_PRIOR)
        ) if (rider_short_count + RIDER_QRP_PRIOR) > 0 else league_qr_mean
        
        # Smooth long-term QRP
        rider_all_qr_sum = df_rider_all["qr"].sum() if not df_rider_all.empty else 0
        rider_all_count = len(df_rider_all)
        smooth_all = (
            (rider_all_qr_sum + RIDER_QRP_PRIOR * league_qr_mean) /
            (rider_all_count + RIDER_QRP_PRIOR)
        ) if (rider_all_count + RIDER_QRP_PRIOR) > 0 else league_qr_mean
        
        # Weighted combination
        return W1 * smooth_short + W2 * smooth_all

    return pd.DataFrame([{
        "rider_id": rider_id,
        "new_rider_flag": int(ride_value <= 20),
        "r_ars_long": _safe_mean(df_rider_all.loc[df_rider_all["qr"] == 1, "rider_score"]),
        "r_ars_short": _safe_mean(df_rider_short.loc[df_rider_short["qr"] == 1, "rider_score"]),
        "r_qrp_weighted": _case_weighted_qrp(),
        "r_ROE_positive": np.nan if ride_value <= 5 else _safe_mean(new_rider_roe_temp[new_rider_roe_temp > 0]),
        "r_ROE_negative": np.nan if ride_value <= 5 else _safe_mean(new_rider_roe_temp[new_rider_roe_temp < 0]),
        "r_pbr_rate": _safe_mean(df_rider_all["pbr"]),
    }])


def bull_data_pull(
    df: pd.DataFrame,
    bull_id: int,
    *,
    bull_short_range: int = BULL_SHORT,
    start_date: pd.Timestamp,
    rider_hand: str | float | None = None,
) -> pd.DataFrame:
    """Return single-row DataFrame with bull-only engineered features prior to start_date.

    If rider_hand is provided ("L" or "R"), hand-based metrics are computed; otherwise set to NaN.
    """
    df_hist = df[df["event_start_date"] < start_date]

    df_bull_all = df_hist.query("bull_id == @bull_id").sort_values("out")
    df_bull_short = df_bull_all.nlargest(bull_short_range, "out")

    # League average for smoothing
    league_qr_mean = df_hist["qr"].mean() if not df_hist.empty else 0.26
    b_qrp_smooth = (
        (df_bull_all.get("qr", pd.Series(dtype=float)).sum() + K_PRIOR * league_qr_mean) /
        ((len(df_bull_all)) + K_PRIOR if (len(df_bull_all) + K_PRIOR) != 0 else np.nan)
    )

    # Hand-based subsets
    if rider_hand in ("L", "R"):
        df_bull_hand = df_bull_all.query("hand == @rider_hand")
        df_bull_hand_short = df_bull_hand.nlargest(bull_short_range, "out")
    else:
        df_bull_hand = pd.DataFrame(columns=df_bull_all.columns)
        df_bull_hand_short = pd.DataFrame(columns=df_bull_all.columns)

    # ROE temp (robust)
    if "b_ROE_temp" in df_bull_all.columns:
        new_bull_roe_temp = pd.to_numeric(df_bull_all["b_ROE_temp"], errors="coerce")
    else:
        r_qrp_long_b = pd.to_numeric(df_bull_all.get("r_qrp_long", pd.Series(dtype=float)), errors="coerce")
        qr_b = pd.to_numeric(df_bull_all.get("qr", pd.Series(dtype=float)), errors="coerce")
        new_bull_roe_temp = r_qrp_long_b - qr_b

    out_value = (df_bull_all["out"].max() if not df_bull_all.empty else 0) + 1

    # Calculate smooth ARS using league priors
    league_ars_mean = df_hist.loc[df_hist["qr"] == 1, "rider_score"].mean() if not df_hist.empty else 85.0

    # Smooth bull ARS
    bull_qr_sum = df_bull_all["qr"].sum()
    b_ars_smooth = (
        (df_bull_all.loc[df_bull_all["qr"] == 1, "rider_score"].sum() + K_PRIOR * league_ars_mean) /
        (bull_qr_sum + K_PRIOR)
    ) if bull_qr_sum > 0 else league_ars_mean

    # Smooth short-term bull ARS
    bull_short_qr_sum = df_bull_short["qr"].sum()
    b_ars_short_smooth = (
        (df_bull_short.loc[df_bull_short["qr"] == 1, "rider_score"].sum() + K_PRIOR * league_ars_mean) /
        (bull_short_qr_sum + K_PRIOR)
    ) if bull_short_qr_sum > 0 else league_ars_mean

    # Smooth hand-based ARS
    if not df_bull_hand.empty:
        hand_qr_sum = df_bull_hand["qr"].sum()
        h_ars_smooth = (
            (df_bull_hand.loc[df_bull_hand["qr"] == 1, "rider_score"].sum() + K_PRIOR * league_ars_mean) /
            (hand_qr_sum + K_PRIOR)
        ) if hand_qr_sum > 0 else league_ars_mean
    else:
        h_ars_smooth = np.nan

    # Smooth short-term hand-based ARS
    if not df_bull_hand_short.empty:
        hand_short_qr_sum = df_bull_hand_short["qr"].sum()
        h_ars_short_smooth = (
            (df_bull_hand_short.loc[df_bull_hand_short["qr"] == 1, "rider_score"].sum() + K_PRIOR * league_ars_mean) /
            (hand_short_qr_sum + K_PRIOR)
        ) if hand_short_qr_sum > 0 else league_ars_mean
    else:
        h_ars_short_smooth = np.nan

    return pd.DataFrame([{
        "bull_id": bull_id,
        "new_bull_flag": int(out_value <= 2),
        "few_rides_flag": int(out_value <= BULL_SHORT),
        "few_rides_flag_hand": int(out_value <= BULL_SHORT / 2),
        "b_abs": _safe_mean(df_bull_all["bull_score"]),
        "b_ars": b_ars_smooth,
        "b_qrp_smooth": b_qrp_smooth,
        "b_abs_short": _safe_mean(df_bull_short["bull_score"]),
        "b_ars_short": b_ars_short_smooth,
        "h_qrp": _safe_mean(df_bull_hand["qr"]) if not df_bull_hand.empty else np.nan,
        "h_abs": _safe_mean(df_bull_hand["bull_score"]) if not df_bull_hand.empty else np.nan,
        "h_ars": h_ars_smooth,
        "h_qrp_short": _safe_mean(df_bull_hand_short["qr"]) if not df_bull_hand_short.empty else np.nan,
        "h_abs_short": _safe_mean(df_bull_hand_short["bull_score"]) if not df_bull_hand_short.empty else np.nan,
        "h_ars_short": h_ars_short_smooth,
        "b_ROE_positive": np.nan if out_value <= 5 else _safe_mean(new_bull_roe_temp[new_bull_roe_temp > 0]),
        "b_ROE_negative": np.nan if out_value <= 5 else _safe_mean(new_bull_roe_temp[new_bull_roe_temp < 0]),
        "b_pbr_rate": _safe_mean(df_bull_all["pbr"]),
        "out": out_value,
    }])


def load_average_bull_features(path: str | Path = None) -> pd.Series:
    """
    Load the average bull features from CSV as a pandas Series.
    Default path is Predict/average_bull_features.csv.
    """
    if path is None:
        from .config import ROOT
        path = ROOT / "Predict" / "average_bull_features.csv"
    df = pd.read_csv(path)
    if df.shape[0] != 1:
        raise ValueError(f"Expected one row in average bull features, got {df.shape[0]}")
    return df.iloc[0]
