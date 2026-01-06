from __future__ import annotations

import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Iterable, Literal

from .predictions import predict_multiple, predict_one
from . import predictions as _pred
from .config import DEFAULT_DATE, FINAL_DATA, FEATURE_LIST, MODEL_FILE
from .feature_engineering import solo_data_pull


def predict_cartesian(
    rider_names: Iterable[str],
    bull_names: Iterable[str],
    *,
    event_date: str = DEFAULT_DATE,
    mode: Literal["auto", "multiple", "one", "fast"] = "auto",
) -> pd.DataFrame:
    """
    Generate predictions for all rider × bull combinations.

    - rider_names: iterable of rider display names (strings)
    - bull_names: iterable of bull display names (strings)
    - event_date: yyyy-mm-dd used for feature cutoff
    - mode:
        "fast"     → optimized batch mode: pull rider features once per rider,
                     bull features once per bull, cartesian join, single batch predict
                     (FASTEST for large combinations, ~100x faster than "one")
        "multiple" → use Predict.predict_multiple for all pairs
        "one"      → call Predict.predict_one per pair (slower, more robust)
        "auto"     → try "fast" if >50 combinations, else "multiple", fallback to "one"

    Returns a DataFrame with columns: rider, bull, probability, probability_pct.
    """
    riders = [str(r).strip() for r in rider_names if str(r).strip()]
    bulls = [str(b).strip() for b in bull_names if str(b).strip()]

    def _to_df(records: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame.from_records(records)
        if "probability" not in df.columns:
            return pd.DataFrame(columns=["rider", "bull", "probability", "probability_pct"])
        df["probability_pct"] = (df["probability"] * 100).round(2)
        df = df.sort_values("probability", ascending=False).reset_index(drop=True)
        # keep commonly used columns in front
        cols = [c for c in ["rider", "bull", "probability", "probability_pct", "event_date"] if c in df.columns]
        other = [c for c in df.columns if c not in cols]
        return df[cols + other]

    # Preflight: report unmatched rider/bull names against current ID maps
    try:
        rider_map, bull_map = _pred._load_id_maps()
        norm = lambda s: _pred._normalize_rider_name(str(s))
        normb = lambda s: _pred._normalize_bull_name(str(s))
        unmatched_riders = [r for r in riders if rider_map.get(norm(r)) is None]
        unmatched_bulls  = [b for b in bulls  if bull_map.get(normb(b)) is None]
        if unmatched_riders:
            print(f"[warn] Unmatched riders (by name → internal id): {unmatched_riders}")
        if unmatched_bulls:
            print(f"[warn] Unmatched bulls (by name → id): {unmatched_bulls}")
    except Exception:
        pass

    if mode == "multiple":
        recs = predict_multiple(riders, bulls, event_date=event_date)
        return _to_df(recs)

    if mode == "one":
        out: list[dict] = []
        for r in riders:
            for b in bulls:
                rec = predict_one(r, b, event_date=event_date)
                out.append(rec)
        return _to_df(out)

    if mode == "fast":
        return _predict_cartesian_fast(riders, bulls, event_date=event_date)

    # auto: use fast for large batches, else try multiple, fallback to one
    total_combos = len(riders) * len(bulls)
    if total_combos > 50:
        try:
            return _predict_cartesian_fast(riders, bulls, event_date=event_date)
        except Exception as e:
            print(f"[fast mode failed, falling back] {e}")
    
    try:
        recs = predict_multiple(riders, bulls, event_date=event_date)
        return _to_df(recs)
    except Exception:
        out = []
        for r in riders:
            for b in bulls:
                rec = predict_one(r, b, event_date=event_date)
                out.append(rec)
        return _to_df(out)


def _predict_cartesian_fast(rider_names: list[str], bull_names: list[str], event_date: str) -> pd.DataFrame:
    """
    Fast batch prediction: pull rider features once per rider, bull features once per bull,
    then cartesian join and batch predict.
    """
    # Load data once
    final_data = _pred._load_final_data()
    features = list(_pred._load_features())
    bst = _pred._load_model()
    rider_dict, bull_dict = _pred._load_id_maps()
    rid_internal_map, rid_legacy_map = _pred._load_rider_id_maps()
    start_date = pd.to_datetime(event_date)

    # Partition features
    rider_feature_cols = [c for c in features if c.startswith("r_") or c in ("new_rider_flag", "few_rides_flag", "few_rides_flag_hand")]
    bull_feature_cols  = [c for c in features if c.startswith("b_") or c.startswith("h_") or c == "new_bull_flag"]
    
    # Fixed bull for rider feature extraction (any bull works)
    fixed_bull_name = bull_names[0] if bull_names else "19H Man Hater"
    fixed_bid = bull_dict.get(_pred._normalize_bull_name(fixed_bull_name))
    if fixed_bid is None and bull_names:
        # try any bull from the list
        for b in bull_names:
            fixed_bid = bull_dict.get(_pred._normalize_bull_name(b))
            if fixed_bid is not None:
                fixed_bull_name = b
                break
    if fixed_bid is None:
        raise ValueError(f"No valid bull found for feature extraction. Bulls: {bull_names}")

    # Fixed riders for bull feature extraction (left/right)
    fixed_left_rider = "Dalton Kasel"
    fixed_right_rider = "Kaique Pacheco"
    left_key = _pred._normalize_rider_name(fixed_left_rider)
    right_key = _pred._normalize_rider_name(fixed_right_rider)
    left_rid_internal = rid_internal_map.get(left_key)
    left_rid_legacy = rid_legacy_map.get(left_key) if left_rid_internal is None else None
    right_rid_internal = rid_internal_map.get(right_key)
    right_rid_legacy = rid_legacy_map.get(right_key) if right_rid_internal is None else None
    
    if left_rid_internal is None and left_rid_legacy is None:
        fixed_left_rider = rider_names[0] if rider_names else None
        if fixed_left_rider:
            left_key = _pred._normalize_rider_name(fixed_left_rider)
            left_rid_internal = rid_internal_map.get(left_key)
            left_rid_legacy = rid_legacy_map.get(left_key) if left_rid_internal is None else None
    if right_rid_internal is None and right_rid_legacy is None:
        fixed_right_rider = rider_names[1] if len(rider_names) > 1 else rider_names[0] if rider_names else None
        if fixed_right_rider:
            right_key = _pred._normalize_rider_name(fixed_right_rider)
            right_rid_internal = rid_internal_map.get(right_key)
            right_rid_legacy = rid_legacy_map.get(right_key) if right_rid_internal is None else None

    # Step 1: Pull rider features once per rider
    rider_rows = {}
    rider_hand_map = {}
    for rider in rider_names:
        key = _pred._normalize_rider_name(rider)
        rid_internal = rid_internal_map.get(key)
        rid_legacy = rid_legacy_map.get(key) if rid_internal is None else None
        if rid_internal is None and rid_legacy is None:
            continue
        try:
            if rid_internal is not None:
                row = solo_data_pull(final_data, rider_id=None, bull_id=fixed_bid, rider_internal_id=rid_internal, start_date=start_date)
            else:
                row = solo_data_pull(final_data, rider_id=rid_legacy, bull_id=fixed_bid, rider_internal_id=None, start_date=start_date)
            for col in rider_feature_cols:
                if col not in row.columns:
                    row[col] = pd.NA
            rider_rows[rider] = row[rider_feature_cols].iloc[0]
            # Infer hand
            df_rider = final_data[final_data["rider_id"] == rid_legacy] if rid_legacy else final_data[final_data["rider_internal_id"] == rid_internal]
            if not df_rider.empty and "hand" in df_rider.columns and df_rider["hand"].notna().any():
                try:
                    rider_hand_map[rider] = str(df_rider["hand"].value_counts().idxmax())
                except Exception:
                    rider_hand_map[rider] = "Unknown"
            else:
                rider_hand_map[rider] = "Unknown"
        except Exception:
            continue

    # Step 2: Pull bull features once per bull (left/right variants)
    bull_rows_left = {}
    bull_rows_right = {}
    for bull in set(bull_names):
        bid = bull_dict.get(_pred._normalize_bull_name(bull))
        if bid is None:
            continue
        try:
            if left_rid_internal is not None:
                row_l = solo_data_pull(final_data, rider_id=None, bull_id=bid, rider_internal_id=left_rid_internal, start_date=start_date)
            else:
                row_l = solo_data_pull(final_data, rider_id=left_rid_legacy, bull_id=bid, rider_internal_id=None, start_date=start_date)
            for col in bull_feature_cols:
                if col not in row_l.columns:
                    row_l[col] = pd.NA
            bull_rows_left[bull] = row_l[bull_feature_cols].iloc[0]
        except Exception:
            pass
        try:
            if right_rid_internal is not None:
                row_r = solo_data_pull(final_data, rider_id=None, bull_id=bid, rider_internal_id=right_rid_internal, start_date=start_date)
            else:
                row_r = solo_data_pull(final_data, rider_id=right_rid_legacy, bull_id=bid, rider_internal_id=None, start_date=start_date)
            for col in bull_feature_cols:
                if col not in row_r.columns:
                    row_r[col] = pd.NA
            bull_rows_right[bull] = row_r[bull_feature_cols].iloc[0]
        except Exception:
            pass

    # Step 3: Cartesian join
    combined_rows = []
    meta = []
    for rider, rider_series in rider_rows.items():
        hand = str(rider_hand_map.get(rider, "Unknown")).lower()
        for bull in bull_names:
            bull_series = None
            if hand.startswith("l") and bull in bull_rows_left:
                bull_series = bull_rows_left[bull]
            elif hand.startswith("r") and bull in bull_rows_right:
                bull_series = bull_rows_right[bull]
            else:
                if bull in bull_rows_right:
                    bull_series = bull_rows_right[bull]
                elif bull in bull_rows_left:
                    bull_series = bull_rows_left[bull]
                else:
                    continue
            full = pd.Series(index=features, dtype=float)
            full[rider_feature_cols] = pd.to_numeric(rider_series.reindex(rider_feature_cols), errors="coerce")
            full[bull_feature_cols]  = pd.to_numeric(bull_series.reindex(bull_feature_cols), errors="coerce")
            combined_rows.append(full)
            meta.append((rider, bull, rider_hand_map.get(rider, "Unknown")))

    if not combined_rows:
        return pd.DataFrame(columns=["rider", "bull", "probability", "probability_pct"])

    # Step 4: Batch predict
    X = pd.DataFrame(combined_rows)
    dm = xgb.DMatrix(X.values, feature_names=features)
    preds = bst.predict(dm)

    df = pd.DataFrame(meta, columns=["rider", "bull", "rider_hand"])
    df["probability"] = np.round(preds.astype(float), 4)
    df["probability_pct"] = (df["probability"] * 100).round(2)
    df = df.sort_values("probability", ascending=False).reset_index(drop=True)
    return df[["rider", "bull", "probability", "probability_pct", "rider_hand"]]


__all__ = ["predict_cartesian"]


