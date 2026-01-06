# quality_checks.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA / HYGIENE CHECKS
# ─────────────────────────────────────────────────────────────────────────────
def validate_rides_schema(df: pd.DataFrame) -> None:
    """Minimal schema checks for rides_data_final.csv input to metrics."""
    required = {
        "master_id": "int64",
        "rider_internal_id": None,  # allow nullable Int64 -> dtype may vary
        "bull_id": None,
        "qr": None,
        "rider_score": None,
        "bull_score": None,
        "event_start_date": None,
        "out": None,
        "hand": None,
        "pbr": None,
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AssertionError(f"[schema] rides_data missing columns: {missing}")

    # basic value constraints
    for c in ["qr", "pbr"]:
        if c in df.columns:
            bad = ~df[c].isin([0, 1]).fillna(False)
            if bad.any():
                raise AssertionError(f"[schema] column {c} must be 0/1 only")

    if df["master_id"].duplicated().any():
        dups = df.loc[df["master_id"].duplicated(), "master_id"].head(5).tolist()
        raise AssertionError(f"[schema] master_id has duplicates, e.g. {dups}")

    # event dates must parse
    try:
        pd.to_datetime(df["event_start_date"])
    except Exception as e:
        raise AssertionError(f"[schema] event_start_date not parseable: {e}")

def validate_final_schema(df: pd.DataFrame) -> None:
    """Basic checks on final_data just before training."""
    must_have = [
        "master_id","bull_id","rider_internal_id","event_start_date","qr",
        "b_qrp","b_qrp_short","r_qrp_long","r_qrp_short"
    ]
    miss = [c for c in must_have if c not in df.columns]
    if miss:
        raise AssertionError(f"[schema] final_data missing: {miss}")

# ─────────────────────────────────────────────────────────────────────────────
# LEAKAGE GUARDS
# ─────────────────────────────────────────────────────────────────────────────
def _expected_b_qrp_from_rides(rides: pd.DataFrame) -> pd.Series:
    """Expected prior bull QR%: per-bull mean of previous qr values."""
    rides = rides.sort_values(["bull_id", "out"])
    grp = rides.groupby("bull_id", group_keys=False)
    # mean of previous outcomes = mean of qr shifted then expanding
    exp = grp["qr"].apply(lambda s: s.shift().expanding().mean())
    return exp

def leakage_guard_bull(rides_data: pd.DataFrame,
                       bull_overall: pd.DataFrame,
                       sample_n: int = 1000,
                       atol: float = 1e-9) -> None:
    """Assert bull prior QR (b_qrp) uses only *past* outs."""
    cols = ["master_id","bull_id","out","qr"]
    m = bull_overall.merge(rides_data[cols], on=["master_id","bull_id","out"], how="left")

    # recompute expected prior mean from rides_data
    exp = _expected_b_qrp_from_rides(m[["bull_id","out","qr"]])
    m = m.assign(_exp_b_qrp=exp.values)

    # sample for speed
    if len(m) > sample_n:
        m = m.sample(sample_n, random_state=42)

    # compare expected vs provided b_qrp
    diff = (m["b_qrp"] - m["_exp_b_qrp"]).abs().fillna(0)
    if not (diff <= atol).all():
        bad = m.loc[diff > atol, ["bull_id","out","b_qrp","_exp_b_qrp"]].head(10)
        raise AssertionError(f"[leak] b_qrp mismatch on {int((diff>atol).sum())} rows.\n{bad}")

def leakage_guard_rider_short(rides_data: pd.DataFrame,
                              rider_df: pd.DataFrame,
                              short_rides: int = 25,
                              id_col: str = "rider_internal_id",
                              sample_n: int = 1000,
                              atol: float = 1e-9) -> None:
    """Assert r_qrp_short equals mean of last `short_rides` *previous* rides."""
    cols = ["master_id", id_col, "qr", "event_start_date", "ride_id"]
    r = rides_data[cols].sort_values([id_col, "event_start_date", "ride_id"])
    g = r.groupby(id_col, group_keys=False)
    exp_short = g["qr"].apply(lambda s: s.shift().rolling(short_rides, min_periods=1).mean())
    check = rider_df[["master_id","r_qrp_short"]].merge(
        r[["master_id"]].assign(_exp_r_qrp_short=exp_short.values),
        on="master_id", how="left"
    )

    if len(check) > sample_n:
        check = check.sample(sample_n, random_state=42)

    diff = (check["r_qrp_short"] - check["_exp_r_qrp_short"]).abs().fillna(0)
    if not (diff <= atol).all():
        bad = check.loc[diff > atol, ["master_id","r_qrp_short","_exp_r_qrp_short"]].head(10)
        raise AssertionError(f"[leak] r_qrp_short mismatch on {int((diff>atol).sum())} rows.\n{bad}")

def leakage_guard_rider_long(rides_data: pd.DataFrame,
                             rider_df: pd.DataFrame,
                             long_days: int = 1095,
                             id_col: str = "rider_internal_id",
                             sample_n: int = 1000,
                             atol: float = 1e-9) -> None:
    """Assert r_qrp_long uses only prior rides within long_days window."""
    r = rides_data[[id_col, "master_id", "qr", "event_start_date"]].copy()
    r["event_start_date"] = pd.to_datetime(r["event_start_date"])
    r = r.sort_values([id_col, "event_start_date"])

    # time-based rolling mean over prior window (closed='left' excludes current)
    exp_list = []
    for rid, sub in r.groupby(id_col, sort=False):
        s = sub.set_index("event_start_date")["qr"]
        exp = s.rolling(f"{long_days}D", closed="left").mean()
        out = pd.DataFrame({"master_id": sub["master_id"].values,
                            "_exp_r_qrp_long": exp.values})
        exp_list.append(out)
    exp_long = pd.concat(exp_list, ignore_index=True)

    check = rider_df[["master_id","r_qrp_long"]].merge(exp_long, on="master_id", how="left")

    if len(check) > sample_n:
        check = check.sample(sample_n, random_state=42)

    diff = (check["r_qrp_long"] - check["_exp_r_qrp_long"]).abs()
    # allow NaNs to match (both NaN is fine)
    mismatch = diff[~(check["r_qrp_long"].isna() & check["_exp_r_qrp_long"].isna())].fillna(0) > atol
    if mismatch.any():
        bad = check.loc[mismatch, ["master_id","r_qrp_long","_exp_r_qrp_long"]].head(10)
        raise AssertionError(f"[leak] r_qrp_long mismatch on {int(mismatch.sum())} rows.\n{bad}")

def run_data_hygiene_checks(
    rides_data: pd.DataFrame,
    bull_overall: pd.DataFrame,
    rider_df: pd.DataFrame,
    *,
    short_rides: int = 25,
    long_days: int = 1095,
    id_col: str = "rider_internal_id",
) -> None:
    """Run schema + leakage guards (raises AssertionError on failure)."""
    validate_rides_schema(rides_data)
    leakage_guard_bull(rides_data, bull_overall)
    leakage_guard_rider_short(rides_data, rider_df, short_rides=short_rides, id_col=id_col)
    leakage_guard_rider_long(rides_data, rider_df, long_days=long_days, id_col=id_col)
    # If no exception raised:
    print("[qa] Data hygiene & leakage checks: PASSED")
