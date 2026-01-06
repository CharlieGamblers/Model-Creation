# clean_data.py — keep original rider_id, assign persistent rider_internal_id (simple)
# NEW: also writes rides_data_full.csv (research) + rides_data_clean.csv (modeling)

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Downcasting object dtype arrays"
)

# ─────────────────────────────────────────────────────────────────────────────
# Resolve repo root (…/Bull_Model)
# ─────────────────────────────────────────────────────────────────────────────
CUR = Path(__file__).resolve()
for p in CUR.parents:
    if p.name == "Bull_Model":
        ROOT = p
        break
else:
    ROOT = CUR.parents[2]

# Paths
RAW_DATA         = ROOT / "Data" / "Raw" / "base_data.csv"
RIDER_INFO       = ROOT / "Data" / "Processed" / "rider_info.csv"

# Output datasets
FULL_OUT         = ROOT / "Data" / "Processed" / "rides_data_full.csv"   # research/audit (keeps RR/DQ/decl)
CLEAN_OUT        = ROOT / "Data" / "Processed" / "rides_data_clean.csv"  # modeling (drops RR/DQ/decl)

RIDER_LOOKUP_XLSX = ROOT / "Important Documents" / "rider_id_list.xlsx"  # persistent mapping
BULL_LOOKUP_XLSX  = ROOT / "Important Documents" / "bull_id_list.xlsx"

# Ensure output dirs
CLEAN_OUT.parent.mkdir(parents=True, exist_ok=True)
RIDER_LOOKUP_XLSX.parent.mkdir(parents=True, exist_ok=True)
BULL_LOOKUP_XLSX.parent.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
base_df   = pd.read_csv(RAW_DATA)
riders_df = pd.read_csv(RIDER_INFO)

# Normalize column names
base_df.columns   = base_df.columns.str.strip().str.lower()
riders_df.columns = riders_df.columns.str.strip().str.lower()

# External rider_id stays as-is (numeric for merge safety)
base_df["rider_id"]   = pd.to_numeric(base_df.get("rider_id"), errors="coerce")
riders_df["rider_id"] = pd.to_numeric(riders_df.get("rider_id"), errors="coerce")

# ─────────────────────────────────────────────────────────────────────────────
# Merge only 'hand' from rider_info (keep base_df rider name unchanged)
# ─────────────────────────────────────────────────────────────────────────────
merged = base_df.merge(riders_df[["rider_id", "hand"]], how="left", on="rider_id")

# ─────────────────────────────────────────────────────────────────────────────
# Cleaning steps (your original logic, made robust)
# ─────────────────────────────────────────────────────────────────────────────
if "rowid" in merged.columns:
    merged = merged.drop_duplicates(subset="rowid")
else:
    merged = merged.drop_duplicates()
print("After drop_duplicates:", merged.shape)

if "isout" in merged.columns:
    merged = merged[merged["isout"].fillna(0) != 0]
print("After isout filter:", merged.shape)

merged = merged.dropna(subset=["go"])
print("After dropna go:", merged.shape)

merged["go"] = merged["go"].astype(str)
merged["go"] = merged["go"].str.replace("F", "", regex=False)
merged["go"] = merged["go"].str.replace("01", "1", regex=False)
merged["go"] = merged["go"].str.replace("02", "2", regex=False)
merged["go"] = merged["go"].replace({
    "S": "Special (Houston)",
    "MO": "Money Ride",
    "B": "Bonus Ride",
    "": "Final Round",
    "5w": "5"
})

if {"qr", "ride_score"}.issubset(merged.columns):
    merged = merged[~((merged["qr"] == 1) & (merged["ride_score"] == 0))]
print("After qr/ride_score filter:", merged.shape)

if "sanction" in merged.columns:
    merged = merged[merged["sanction"].isin(["PBR", "PRCA", "PRCAX"])]
if "event_id" in merged.columns:
    merged = merged[~merged["event_id"].astype(str).str.contains("AZ", na=False)]
print("After sanction/event_id filter:", merged.shape)

# Ensure comments/perf exist and normalize for flag detection
merged["comments"] = merged.get("comments", pd.Series(index=merged.index, dtype=object))
merged["perf"]     = merged.get("perf", pd.Series(index=merged.index, dtype=object))

# Keep original behavior of replacing "*" with NaN
merged["comments"] = merged["comments"].replace("*", np.nan)

comments_l = merged["comments"].fillna("").astype(str).str.lower()
perf_str   = merged["perf"].fillna("").astype(str)
perf_l     = perf_str.str.lower()

# ─────────────────────────────────────────────────────────────────────────────
# Flags (more robust: case-insensitive + word boundaries where appropriate)
# ─────────────────────────────────────────────────────────────────────────────
merged["score_flag"]    = comments_l.str.contains(r"\(nm\)", na=False).astype(int)
merged["shootout_flag"] = perf_l.str.contains(r"f", na=False).astype(int)

event_id_s = merged.get("event_id", pd.Series(index=merged.index, dtype=str)).fillna("").astype(str)
merged["UTB_flag"]      = event_id_s.str.contains("AB", na=False).astype(int)

merged["time_flag"]     = comments_l.str.contains(r"\bnt\b", na=False).astype(int)
merged["dq_flag"]       = comments_l.str.contains(r"\bdq\b", na=False).astype(int)
merged["rr_flag"]       = comments_l.str.contains(r"\brr\b", na=False).astype(int)
merged["decl_flag"]     = comments_l.str.contains(r"\bdecl\b", na=False).astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENT INTERNAL RIDER IDs (simple)
# - Keep original 'rider_id' column (external)
# - Assign 'rider_internal_id' to every unique rider name
# - Stable across runs: load existing mapping; only add IDs for new names
# ─────────────────────────────────────────────────────────────────────────────
# Basic name cleanup
if "rider" not in merged.columns:
    merged["rider"] = ""
merged["rider"] = merged["rider"].astype(str).str.strip()
merged = merged[merged["rider"] != ""]  # drop blank names

# Load existing lookup (if present)
if RIDER_LOOKUP_XLSX.exists():
    lookup_df = pd.read_excel(RIDER_LOOKUP_XLSX)
    lookup_df.columns = lookup_df.columns.str.strip().str.lower()
    # expected columns: rider, rider_internal_id (we keep rider_id and hand too if present)
    if "rider" not in lookup_df.columns or "rider_internal_id" not in lookup_df.columns:
        raise ValueError(
            f"Existing lookup {RIDER_LOOKUP_XLSX} missing required columns: 'rider', 'rider_internal_id'."
        )
else:
    lookup_df = pd.DataFrame(columns=["rider", "rider_id", "rider_internal_id", "hand"])

def _key(s: str) -> str:
    return str(s).strip().lower()

# Build dict from existing (case-insensitive)
existing_map = {
    _key(r): int(i)
    for r, i in zip(lookup_df.get("rider", []), lookup_df.get("rider_internal_id", []))
    if pd.notna(r) and pd.notna(i)
}
next_internal = (max(existing_map.values()) + 1) if existing_map else 1

# Assign internal IDs to all current unique names (reusing existing where present)
current_names = sorted(merged["rider"].unique(), key=lambda s: s.lower())
for name in current_names:
    k = _key(name)
    if k not in existing_map:
        existing_map[k] = next_internal
        next_internal += 1

# Attach 'rider_internal_id' to merged
merged["rider_internal_id"] = merged["rider"].map(lambda s: existing_map[_key(s)]).astype(int)

# Build refreshed lookup (preserve original rider_id + hand for convenience)
if "event_start_date" in merged.columns:
    merged["event_start_date"] = pd.to_datetime(merged["event_start_date"], errors="coerce")

lookup_new = (
    merged.sort_values("event_start_date")  # latest last
    .groupby("rider", as_index=False)
    .agg({
        "rider_id": "last",   # original external ID (may be NaN if not present)
        "hand": "last"
    })
)

lookup_new["rider_internal_id"] = lookup_new["rider"].map(lambda s: existing_map[_key(s)]).astype(int)
lookup_new = lookup_new.loc[:, ["rider", "rider_id", "rider_internal_id", "hand"]].sort_values(
    "rider", key=lambda s: s.str.lower()
)

# Write lookup back (stable IDs preserved)
lookup_new.to_excel(RIDER_LOOKUP_XLSX, index=False)
print(f"Rider lookup saved → {RIDER_LOOKUP_XLSX}  (rows: {len(lookup_new)})")

# ─────────────────────────────────────────────────────────────────────────────
# Bull lookup table (simple, from stock_no + stock_name)
# ─────────────────────────────────────────────────────────────────────────────
for col in ["stock_no", "stock_name", "stock_id"]:
    if col not in merged.columns:
        merged[col] = np.nan

bulls = (
    merged[["stock_no", "stock_name", "stock_id"]]
    .assign(
        stock_no=lambda d: d["stock_no"].astype(str).str.strip(),
        stock_name=lambda d: d["stock_name"].astype(str).str.strip(),
        bull=lambda d: d["stock_no"] + " " + d["stock_name"],
        bull_id=lambda d: d["stock_id"],
    )
    .loc[:, ["bull", "bull_id"]]
    .dropna(subset=["bull", "bull_id"])
    .drop_duplicates()
    .sort_values("bull_id")
)

bulls.to_excel(BULL_LOOKUP_XLSX, index=False)
print(f"Bull lookup saved → {BULL_LOOKUP_XLSX}  (rows: {len(bulls)})")

# ─────────────────────────────────────────────────────────────────────────────
# Split output: FULL (research) and CLEAN (modeling)
# CLEAN excludes: dq_flag==1 OR rr_flag==1 OR decl_flag==1
# ─────────────────────────────────────────────────────────────────────────────
# Save full first
merged.to_csv(FULL_OUT, index=False)
print(f"Full cleaned data saved → {FULL_OUT}  (rows: {len(merged)})")

# Build clean subset for modeling
dq = merged["dq_flag"].fillna(0).astype(int)
rr = merged["rr_flag"].fillna(0).astype(int)
decl = merged["decl_flag"].fillna(0).astype(int)

is_clean_attempt = (dq == 0) & (rr == 0) & (decl == 0)
clean = merged.loc[is_clean_attempt].copy()

clean.to_csv(CLEAN_OUT, index=False)
print(f"Modeling-clean data saved → {CLEAN_OUT}  (rows: {len(clean)})")
print(f"Dropped for modeling: {len(merged) - len(clean):,}")

# Optional quick audit counts
print("Flag counts in FULL (rows with flag==1):")
print("  dq_flag  :", int((dq == 1).sum()))
print("  rr_flag  :", int((rr == 1).sum()))
print("  decl_flag:", int((decl == 1).sum()))
