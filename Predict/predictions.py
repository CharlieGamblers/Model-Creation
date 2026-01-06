# predictions.py
from __future__ import annotations

import pandas as pd
import xgboost as xgb
import numpy as np
import shap
from functools import lru_cache
from .config import (
    FINAL_DATA,
    FEATURE_LIST,
    MODEL_FILE,
    RIDER_XLSX,
    BULL_XLSX,
    DEFAULT_DATE,
    replacements,            # NEW
    rider_replacements,      # NEW
)
from .feature_engineering import solo_data_pull, load_average_bull_features


# ---------- Normalization helpers ----------
def _norm(s: str) -> str:
    return str(s).strip().lower()

@lru_cache(maxsize=1)
def _bull_synonyms() -> dict[str, str]:
    """
    Map known 'bad' bull strings -> canonical replacements (both lowercased).
    Built from config.replacements.
    """
    df = replacements.copy()
    a = df["Known_Discrepancies"].astype(str).str.strip().str.lower()
    b = df["Replacements"].astype(str).str.strip().str.lower()
    return dict(zip(a, b))

def _normalize_rider_name(name: str) -> str:
    key = _norm(name)
    return rider_replacements.get(key, key)

def _normalize_bull_name(name: str) -> str:
    key = _norm(name)
    return _bull_synonyms().get(key, key)


# -------------------- Cached loaders --------------------
@lru_cache(maxsize=1)
def _load_final_data() -> pd.DataFrame:
    return pd.read_csv(FINAL_DATA, parse_dates=["event_start_date"], low_memory=False)

@lru_cache(maxsize=1)
def _load_features() -> tuple[str, ...]:
    with open(FEATURE_LIST, encoding="utf-8") as fh:
        features_list = tuple(ln.strip() for ln in fh if ln.strip())
    return features_list

@lru_cache(maxsize=1)
def _load_model() -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(MODEL_FILE))
    return booster

@lru_cache(maxsize=1)
def _load_id_maps():
    """
    Rider map (union): name(lower) -> rider_internal_id if available else rider_id
    Bull map:          name(lower) -> bull_id
    """
    # Riders from XLSX
    rid_df = pd.read_excel(RIDER_XLSX)
    rid_df.columns = [str(c).strip().lower() for c in rid_df.columns]
    name_candidates = [c for c in ("rider","name","rider_name") if c in rid_df.columns]
    rid_name_col = name_candidates[0] if name_candidates else rid_df.select_dtypes(include=["object"]).columns.tolist()[0]
    # prefer internal id, else rider_id, else id
    id_col = None
    for c in ("rider_internal_id","rider_id","id"):
        if c in rid_df.columns:
            id_col = c
            break
    if id_col is None:
        # fallback to best numeric column excluding name
        best_col = None
        best_valid = -1.0
        for c in rid_df.columns:
            if c == rid_name_col:
                continue
            vals = pd.to_numeric(rid_df[c], errors="coerce")
            valid = 1 - vals.isna().mean()
            if valid > best_valid:
                best_valid, best_col = valid, c
        id_col = best_col
    df_ids = rid_df[[rid_name_col, id_col]].copy()
    df_ids[id_col] = pd.to_numeric(df_ids[id_col], errors="coerce")
    df_ids = df_ids.dropna(subset=[id_col])
    rider_dict = { _norm(r): int(i) for r,i in zip(df_ids[rid_name_col], df_ids[id_col]) if pd.notna(r) and pd.notna(i) }

    # Bulls from XLSX: first column = bull name, second = bull id
    bid_df = pd.read_excel(BULL_XLSX)
    if bid_df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {BULL_XLSX}")
    bull_name_col = bid_df.columns[0]
    bull_id_col   = bid_df.columns[1]
    bid_df = bid_df.dropna(subset=[bull_id_col])
    bull_dict = {
        _norm(b): int(i)
        for b, i in zip(bid_df[bull_name_col], bid_df[bull_id_col])
        if pd.notna(b) and pd.notna(i)
    }

    # Add synonyms → map bad->canonical if canonical exists
    syn = _bull_synonyms()
    for bad, good in syn.items():
        if good in bull_dict and bad not in bull_dict:
            bull_dict[bad] = bull_dict[good]

    return rider_dict, bull_dict

@lru_cache(maxsize=1)
def _load_rider_id_maps():
    """Return (internal_map, legacy_map) built from RIDER_XLSX."""
    rid_df = pd.read_excel(RIDER_XLSX)
    rid_df.columns = [str(c).strip().lower() for c in rid_df.columns]
    name_candidates = [c for c in ("rider","name","rider_name") if c in rid_df.columns]
    rid_name_col = name_candidates[0] if name_candidates else rid_df.select_dtypes(include=["object"]).columns.tolist()[0]
    internal_map = {}
    legacy_map = {}
    if "rider_internal_id" in rid_df.columns:
        df_i = rid_df[[rid_name_col, "rider_internal_id"]].copy()
        df_i["rider_internal_id"] = pd.to_numeric(df_i["rider_internal_id"], errors="coerce")
        df_i = df_i.dropna(subset=["rider_internal_id"])
        internal_map = { _norm(r): int(i) for r,i in zip(df_i[rid_name_col], df_i["rider_internal_id"]) if pd.notna(r) and pd.notna(i) }
    id_col = "rider_id" if "rider_id" in rid_df.columns else ("id" if "id" in rid_df.columns else None)
    if id_col is not None:
        df_l = rid_df[[rid_name_col, id_col]].copy()
        df_l[id_col] = pd.to_numeric(df_l[id_col], errors="coerce")
        df_l = df_l.dropna(subset=[id_col])
        legacy_map = { _norm(r): int(i) for r,i in zip(df_l[rid_name_col], df_l[id_col]) if pd.notna(r) and pd.notna(i) }
    return internal_map, legacy_map

@lru_cache(maxsize=1)
def _load_shap_explainer():
    model = _load_model()
    return shap.TreeExplainer(model)

def _align_to_features(df_row: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    for c in features:
        if c not in df_row.columns:
            df_row[c] = pd.NA
    df_row = df_row.loc[:, [c for c in features if c in df_row.columns]]
    df_row = df_row.apply(pd.to_numeric, errors="coerce")
    return df_row


# -------------------- Public API --------------------
def predict_one(rider_name: str, bull_name: str, *, event_date: str = DEFAULT_DATE) -> dict:
    final_data = _load_final_data()
    features = list(_load_features())
    bst = _load_model()

    rider_union, bull_dict = _load_id_maps()
    rid_internal_map, rid_legacy_map = _load_rider_id_maps()
    key = _normalize_rider_name(rider_name)
    rid_internal = rid_internal_map.get(key)
    rid_legacy = rid_legacy_map.get(key) if rid_internal is None else None
    bid = bull_dict.get(_normalize_bull_name(bull_name))
    if rid_internal is None and rid_legacy is None:
        raise ValueError(f"[ERROR] Rider not found (any ID): '{rider_name}'")
    if bid is None:
        raise ValueError(f"[ERROR] Bull not found (ID): '{bull_name}'")

    start_dt = pd.to_datetime(event_date)
    if rid_internal is not None:
        row = solo_data_pull(
            _load_final_data(),
            rider_id=None,
            bull_id=bid,
            rider_internal_id=rid_internal,
            start_date=start_dt,
        )
    else:
        row = solo_data_pull(
            _load_final_data(),
            rider_id=rid_legacy,
            bull_id=bid,
            rider_internal_id=None,
            start_date=start_dt,
        )
    row = _align_to_features(row, features)

    dm = xgb.DMatrix(row.values, feature_names=features)
    prob = float(bst.predict(dm)[0])

    explainer = _load_shap_explainer()
    sv = explainer(row)
    shap_values = sv.values[0].tolist()
    base_log = float(sv.base_values[0])
    base_prob = 1 / (1 + np.exp(-base_log))

    return {
        "rider": rider_name,
        "bull": bull_name,
        "event_date": event_date,
        "probability": round(prob, 4),
        "base_probability": round(base_prob, 4),
        "features": row.columns.tolist(),
        "feature_values": row.iloc[0].tolist(),
        "shap_values": shap_values,
    }

def predict_multiple(rider_names: list[str], bull_names: list[str], *, event_date: str = DEFAULT_DATE) -> list[dict]:
    final_data = _load_final_data()
    features = list(_load_features())
    bst = _load_model()
    rider_union, bull_dict = _load_id_maps()
    rid_internal_map, rid_legacy_map = _load_rider_id_maps()
    start_date = pd.to_datetime(event_date)
    avg_bull_features = load_average_bull_features()

    out: list[dict] = []

    for r in rider_names:
        key = _normalize_rider_name(r)
        rid_internal = rid_internal_map.get(key)
        rid_legacy = rid_legacy_map.get(key) if rid_internal is None else None
        if rid_internal is None and rid_legacy is None:
            continue
        for b in bull_names:
            bid = bull_dict.get(_normalize_bull_name(b))
            used_baseline = False
            if bid is None:
                print(f"[baseline] Bull not found: '{b}'. Using average bull features.")
                used_baseline = True
                if rid_internal is not None:
                    row = solo_data_pull(final_data, rider_id=None, bull_id=-1, rider_internal_id=rid_internal, start_date=start_date)
                else:
                    row = solo_data_pull(final_data, rider_id=rid_legacy, bull_id=-1, rider_internal_id=None, start_date=start_date)
                for col in avg_bull_features.index:
                    if col in row.columns:
                        row[col] = avg_bull_features[col]
            else:
                if rid_internal is not None:
                    row = solo_data_pull(final_data, rider_id=None, bull_id=bid, rider_internal_id=rid_internal, start_date=start_date)
                else:
                    row = solo_data_pull(final_data, rider_id=rid_legacy, bull_id=bid, rider_internal_id=None, start_date=start_date)
            row = _align_to_features(row, features)
            dm = xgb.DMatrix(row.values, feature_names=features)
            prob = float(bst.predict(dm)[0])
            out.append({"rider": r, "bull": b, "probability": round(prob, 4)})
    return out
