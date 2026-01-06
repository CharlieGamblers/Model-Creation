# build_dataset.py  •  internal-ID only (hardened)
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Downcasting object dtype arrays"
)


from bull_metrics import calculate_bull_data, calculate_bull_data_by_hand
from rider_metrics import calculate_rider_data

# ─────────────────────────────────────────────────────────────────────────────
#  0 .  Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
cur = Path(__file__).resolve()
for p in cur.parents:
    if p.name == "Bull_Model":
        ROOT_DIR = p
        break
else:
    raise FileNotFoundError("folder 'Bull_Model' not found up-tree")

DATA_DIR  = ROOT_DIR / "Data" / "Processed"
RAW_FILE  = DATA_DIR / "rides_data_clean.csv"
RIDES_OUT = DATA_DIR / "rides_data_final.csv"
FINAL_OUT = DATA_DIR / "final_data.csv"

# Make Predict package importable and pull shared constants
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from Predict.config import (
    BULL_SHORT as BULL_SHORT,
    RIDER_SHORT_RIDES as RIDER_SHORT_RIDES,
    RIDER_LONG_DAYS as LONG_DAYS,
    K_PRIOR as K_PRIOR,
)

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────
# 1.  Load raw rides (bull IDs := stock_id) + REQUIRE rider_internal_id
# ─────────────────────────────────────────────────────────────────────────
print("[build] loading", RAW_FILE)
df = pd.read_csv(RAW_FILE)

# Basic dtype/column hygiene
for c in ("event_start_date",):
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

# Bull IDs
if "bull_id" not in df.columns:
    if "stock_id" in df.columns:
        df["bull_id"] = pd.to_numeric(df["stock_id"], errors="raise")
        print("[build] bull_id column created from stock_id")
    else:
        raise KeyError("Raw file lacks both bull_id and stock_id columns.")
df["bull_id"] = df["bull_id"].astype("Int64")

# Rider INTERNAL IDs (required)
if "rider_internal_id" not in df.columns:
    raise KeyError("rides_data_clean.csv must include 'rider_internal_id' from clean_data.py.")
df["rider_internal_id"] = pd.to_numeric(df["rider_internal_id"], errors="coerce").astype("Int64")

# Rider name (kept as metadata)
df["rider"] = df.get("rider", "").astype(str).str.strip()

# Ensure bull text columns exist (for human-readable 'bull')
for col in ("stock_no", "stock_name"):
    if col not in df.columns:
        df[col] = ""
df["stock_no"] = df["stock_no"].astype(str).str.strip()
df["stock_name"] = df["stock_name"].astype(str).str.strip()
df["bull"] = (df["stock_no"] + " " + df["stock_name"]).str.strip()

# Chronological sort + sequences (by INTERNAL ID)
sort_cols = [c for c in ["rider", "event_start_date", "perf", "go", "comments"] if c in df.columns]
df = df.sort_values(sort_cols, na_position="last")

df["ride_id"]      = df.groupby("rider_internal_id").cumcount() + 1
df["bull_ride_id"] = df.groupby("bull_id").cumcount() + 1

# 'out' (attempt number) — default to bull_ride_id if not present
if "out" not in df.columns:
    df["out"] = df["bull_ride_id"]

# Ensure score / PBR / master_id
if "rider_score" not in df.columns and "ride_score" in df.columns:
    df["rider_score"] = df["ride_score"]
if "bull_score" not in df.columns and "stock_score" in df.columns:
    df["bull_score"] = df["stock_score"]
if "pbr" not in df.columns:
    df["pbr"] = (df.get("sanction", "") == "PBR").astype(int)

df = df.reset_index(drop=True)
df["master_id"] = np.arange(1, len(df) + 1, dtype="int64")

# 🔒 Columns to keep for modeling (NO legacy rider_id in keys)
RIDES_COLS = [
    "master_id",
    "ride_id", "bull_ride_id",
    "rider_internal_id", "rider", "hand",
    "bull_id", "bull", "stock_no",
    "rider_score", "bull_score", "time", "comments", "qr",
    "event_start_date", "pbr", "event_id",
    "score_flag", "time_flag", "dq_flag", "rr_flag", "decl_flag",
    "out",
]

# Create final rides_data frame
missing_cols = [c for c in RIDES_COLS if c not in df.columns]
for c in missing_cols:
    # Fill missing optional columns with sensible defaults
    if c in {"hand", "comments", "event_id", "stock_no", "bull"}:
        df[c] = ""
    elif c in {"rider_score", "bull_score", "time", "qr", "score_flag", "time_flag", "dq_flag", "rr_flag", "decl_flag"}:
        df[c] = np.nan
    elif c == "event_start_date":
        df[c] = pd.NaT
    else:
        df[c] = np.nan

rides_data = (
    df[RIDES_COLS]
    .drop_duplicates()
    .assign(event_start_date=lambda d: pd.to_datetime(d["event_start_date"], errors="coerce"))
)

RIDES_OUT.parent.mkdir(parents=True, exist_ok=True)
rides_data.to_csv(RIDES_OUT, index=False)
print(f"[build] rides_data_final.csv → {RIDES_OUT}   rows: {len(rides_data):,}")

# Convenience exports (internal IDs only are required for modeling)
rider_list_out = ROOT_DIR / "Important Documents" / "rider_id_list.xlsx"
rider_id_list = (
    rides_data[["rider", "rider_internal_id", "hand"]]
    .drop_duplicates()
    .sort_values(["rider_internal_id", "rider"])
)
rider_list_out.parent.mkdir(parents=True, exist_ok=True)
rider_id_list.to_excel(rider_list_out, index=False)
print(f"[build] Rider list saved → {rider_list_out}")

bull_list_out = ROOT_DIR / "Important Documents" / "bull_id_list.xlsx"
bull_id_list = (
    rides_data[["bull", "bull_id"]]
    .drop_duplicates()
    .sort_values("bull_id")
)
bull_id_list.to_excel(bull_list_out, index=False)
print(f"[build] Bull ID list saved → {bull_list_out}")

# ─────────────────────────────────────────────────────────────────────────────
# 2 .  Rider / Bull metrics
# ─────────────────────────────────────────────────────────────────────────────
print("[build] computing bull metrics …")
# These functions should only rely on columns present in rides_data.
bull_overall = calculate_bull_data(rides_data, range=BULL_SHORT)
bull_hand    = calculate_bull_data_by_hand(rides_data, range=BULL_SHORT)

print("[build] computing rider metrics …")
# calculate_rider_data must key by rider_internal_id internally (or take id_col)
rider_df = calculate_rider_data(
    rides_data,
    short_rides=RIDER_SHORT_RIDES,
    long_days=LONG_DAYS,
    w1=0.15,
)

# After computing bull_overall and before merging
print("[build] computing and saving average bull features …")
PREDICT_DIR = ROOT_DIR / "Predict"
PREDICT_DIR.mkdir(parents=True, exist_ok=True)

# Get the latest row per bull (highest 'out' value)
latest_bull_rows = bull_overall.sort_values(["bull_id", "out"]).groupby("bull_id", as_index=False).tail(1)

# Columns to average: all numeric columns except IDs and flags
exclude_cols = {"master_id", "bull_id", "out", "new_bull_flag", "few_rides_flag"}
bull_feature_cols = [col for col in latest_bull_rows.select_dtypes(include="number").columns if col not in exclude_cols]

avg_bull_features = latest_bull_rows[bull_feature_cols].mean(numeric_only=True)
avg_bull_features_df = avg_bull_features.to_frame().T
avg_bull_features_df.to_csv(PREDICT_DIR / "average_bull_features.csv", index=False)
print(f"[build] Average bull features saved → {PREDICT_DIR / 'average_bull_features.csv'}")

# ─────────────────────────────────────────────────────────────────────────────
# 3 .  Merge everything  (join with INTERNAL ID)
# ─────────────────────────────────────────────────────────────────────────────
print("[build] merging …")
# Expect bull_* metrics keyed by at least ["bull_id", "out"] and a unique row id (e.g., master_id or merge key)
# If your calculate_* return shapes differ, adjust join keys accordingly.
bull_data = bull_overall.merge(
    bull_hand,
    on=[c for c in ["master_id", "bull_id", "out"] if c in bull_overall.columns and c in bull_hand.columns],
    how="left",
)

final_data = (
    bull_data
    .merge(rider_df, on=[c for c in ["master_id"] if c in rider_df.columns], how="left")
    .merge(
        rides_data,
        on=[c for c in [
            "master_id", "ride_id", "out",
            "rider_internal_id", "hand",
            "bull_id", "event_start_date", "pbr"
        ] if c in bull_data.columns or c in rides_data.columns],
        how="left",
    )
)

# ─────────────────────────────────────────────────────────────────────────────
# 3B.  ROE columns  — grouped by INTERNAL rider IDs
# ─────────────────────────────────────────────────────────────────────────────
final_data = final_data.assign(
    b_ROE_temp=lambda d: d["r_qrp_long"] - d["qr"],
    r_ROE_temp=lambda d: d["qr"] - d["b_qrp"],
)

def _running_avg(s: pd.Series) -> pd.Series:
    shifted = s.shift().fillna(0)
    return (shifted.cumsum() / pd.Series(range(len(s))).replace(0, np.nan)).fillna(0)

def _running_avg_mask(s: pd.Series, mask: pd.Series) -> pd.Series:
    shifted = s.shift().where(mask.shift()).fillna(0)
    denom   = mask.shift().cumsum().replace(0, np.nan)
    return (shifted.cumsum() / denom).fillna(0)

# Guard against missing groups if earlier merges filtered something
if "rider_internal_id" in final_data.columns:
    r_grp = final_data.groupby("rider_internal_id")["r_ROE_temp"]
    final_data["r_ROE"]          = r_grp.transform(_running_avg)
    final_data["r_ROE_positive"] = r_grp.transform(lambda s: _running_avg_mask(s, s > 0))
    final_data["r_ROE_negative"] = r_grp.transform(lambda s: _running_avg_mask(s, s < 0))
else:
    final_data["r_ROE"] = final_data["r_ROE_positive"] = final_data["r_ROE_negative"] = 0.0

if "bull_id" in final_data.columns:
    b_grp = final_data.groupby("bull_id")["b_ROE_temp"]
    final_data["b_ROE"]          = b_grp.transform(_running_avg)
    final_data["b_ROE_positive"] = b_grp.transform(lambda s: _running_avg_mask(s, s > 0))
    final_data["b_ROE_negative"] = b_grp.transform(lambda s: _running_avg_mask(s, s < 0))
else:
    final_data["b_ROE"] = final_data["b_ROE_positive"] = final_data["b_ROE_negative"] = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 3C.  League prior + smoothed bull QR
# ─────────────────────────────────────────────────────────────────────────────
print("[build] computing league prior & shrinkage …")
final_data = final_data.sort_values("event_start_date").reset_index(drop=True)

final_data["league_outs_prior"] = np.arange(len(final_data))
final_data["league_succ_prior"] = final_data["qr"].shift().fillna(0).cumsum()
final_data["league_qr_mean"]    = (
    final_data["league_succ_prior"] /
    final_data["league_outs_prior"].replace(0, np.nan)
).bfill().fillna(0.26)

final_data["bull_outs_prior"] = final_data.groupby("bull_id").cumcount()
final_data["bull_succ_prior"] = (
    final_data.groupby("bull_id")["qr"]
    .apply(lambda s: s.shift().fillna(0).cumsum())
    .reset_index(level=0, drop=True)
)

final_data["b_qrp_smooth"] = (
    (final_data["bull_succ_prior"] + K_PRIOR * final_data["league_qr_mean"]) /
    (final_data["bull_outs_prior"] + K_PRIOR)
)

# ─────────────────────────────────────────────────────────────────────────────
# 3D.  Previous matchup features (Prev_Matchups, Previous_Rides)
# ─────────────────────────────────────────────────────────────────────────────
print("[build] computing previous matchup features …")
# Data already sorted chronologically above

# Group by rider-bull pair and compute cumulative counts/sums
if "rider_internal_id" in final_data.columns:
    matchup_key = ["rider_internal_id", "bull_id"]
else:
    matchup_key = ["rider_id", "bull_id"]

# Count previous matchups (cumulative count excluding current row)
final_data["Prev_Matchups"] = (
    final_data.groupby(matchup_key)
    .cumcount()
    .astype(int)
)

# Sum of QR for previous matchups (cumulative sum excluding current row)
final_data["Previous_Rides"] = (
    final_data.groupby(matchup_key)["qr"]
    .transform(lambda s: s.shift().fillna(0).cumsum())
    .astype(float)
)

# ─────────────────────────────────────────────────────────────────────────────
# 4 .  Save
# ─────────────────────────────────────────────────────────────────────────────
num_cols = final_data.select_dtypes("number").columns
final_data[num_cols] = final_data[num_cols].where(~final_data[num_cols].isna(), np.nan)

FINAL_OUT.parent.mkdir(parents=True, exist_ok=True)
final_data.to_csv(FINAL_OUT, index=False)
print(f"[build] final_data.csv → {FINAL_OUT}  |  rows: {len(final_data):,}")
