# model_train.py  •  internal-ID only (hardened)
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
import xgboost as xgb


# ─────────────────────────────────────────────────────────────────────────────
# Paths (bootstrap early so Predict/ imports work when run as a script)
# ─────────────────────────────────────────────────────────────────────────────
cur = Path(__file__).resolve()
for cand in cur.parents:
    if cand.name == "Bull_Model":
        ROOT = cand
        break
else:
    ROOT = cur.parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PREDICT_DIR = ROOT / "Predict"
if str(PREDICT_DIR) not in sys.path:
    sys.path.insert(0, str(PREDICT_DIR))

DATA_DIR = ROOT / "Data" / "Processed"
MODEL_DIR = ROOT / "Models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FINAL_DATA = DATA_DIR / "final_data.csv"
MODEL_FILE = MODEL_DIR / "xgb_model.json"
FEAT_FILE  = MODEL_DIR / "feature_cols.txt"

# Train cutoff (no leakage) — use Predict package config
from Predict.config import DEFAULT_DATE as DAY_BEFORE_EVENT

from Predict.feature_engineering import solo_data_pull  # must accept rider_internal_id

# ─────────────────────────────────────────────────────────────────────────────
# Load dataset
# ─────────────────────────────────────────────────────────────────────────────
print("[model] loading final_data …", FINAL_DATA)
final_data = pd.read_csv(FINAL_DATA, parse_dates=["event_start_date"])
print("[model] data shape:", final_data.shape)

required_cols = {"rider_internal_id", "bull_id", "qr", "event_start_date"}
missing = required_cols - set(final_data.columns)
if missing:
    raise KeyError(f"final_data.csv missing required columns: {sorted(missing)}")

# ─────────────────────────────────────────────────────────────────────────────
# Derive feature columns from solo_data_pull (internal-ID path)
# ─────────────────────────────────────────────────────────────────────────────
eligible = final_data.loc[final_data["event_start_date"] < DAY_BEFORE_EVENT, ["rider_internal_id", "bull_id"]]
eligible = eligible.dropna().astype({"rider_internal_id": int, "bull_id": int})
if eligible.empty:
    raise RuntimeError("No rows before DAY_BEFORE_EVENT to probe feature columns.")

sample_internal = int(eligible.iloc[0]["rider_internal_id"])
sample_bull     = int(eligible.iloc[0]["bull_id"])

print("[model] probing feature columns via solo_data_pull …")
probe = solo_data_pull(
    final_data,
    rider_id=None,                  # legacy param unused
    bull_id=sample_bull,
    rider_internal_id=sample_internal,
    start_date=DAY_BEFORE_EVENT,
)

# Drop obvious non-features from the probe
non_features = {
    "qr", "rider_id", "rider_internal_id", "bull_id", "event_start_date",
    "rider", "bull", "hand", "pbr", "ride_id", "bull_ride_id", "master_id",
    # any other label/ID/meta columns you know about can be added here
}
feature_cols = [c for c in probe.columns if c not in non_features]

# Optionally include derived columns computed in build (if present in final_data)
for extra in ["b_ROE_positive", "b_ROE_negative", "b_ROE", "r_ROE"]:
    if extra in final_data.columns and extra not in feature_cols:
        feature_cols.append(extra)

if not feature_cols:
    raise RuntimeError("No feature columns inferred from solo_data_pull.")

# Persist the feature list (what predictions.py will load if model lacks names)
FEAT_FILE.write_text("\n".join(feature_cols), encoding="utf-8")
print(f"[model] feature list saved → {FEAT_FILE}  (n={len(feature_cols)})")

# ─────────────────────────────────────────────────────────────────────────────
# Build training frame (strictly before cutoff)
# ─────────────────────────────────────────────────────────────────────────────
train_df = final_data.loc[final_data["event_start_date"] < DAY_BEFORE_EVENT, :].copy()

# Ensure all features exist in train_df (add missing as NaN)
for col in feature_cols:
    if col not in train_df.columns:
        train_df[col] = np.nan

# Select X / y
use_cols = feature_cols + ["qr"]
ml_data = train_df.loc[:, use_cols]

# Coerce numerics and clean NaN/Inf for XGBoost
X = ml_data.drop(columns=["qr"])
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)  # simple impute; swap for better if desired
y = ml_data["qr"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X.columns.tolist())
dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=X.columns.tolist())

# ─────────────────────────────────────────────────────────────────────────────
# Train model
# ─────────────────────────────────────────────────────────────────────────────
params = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "eval_metric": "auc",
    "eta": 0.05,
    "max_depth": 5,
    "gamma": 0.1,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "subsample": 0.9,
    "alpha": 1.0,
    "lambda": 1.0,
    "nthread": 0,          # let xgb decide; set to a number to pin threads
    "tree_method": "hist", # faster on tabular
}

print("[model] cross-validating …")
cv_res = xgb.cv(
    params=params,
    dtrain=dtrain,
    nfold=3,
    num_boost_round=1000,
    early_stopping_rounds=20,
    verbose_eval=False,
    seed=123,
)
best_n = int(cv_res["test-auc-mean"].idxmax() + 1)
print(f"[model] best_iter: {best_n}")

bst = xgb.train(params, dtrain, num_boost_round=best_n)
bst.save_model(str(MODEL_FILE))
print(f"[model] booster saved → {MODEL_FILE}")

# Redundant but handy: persist feature names used for training
FEAT_FILE.write_text("\n".join(X.columns.tolist()), encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────────────────
pred_test = bst.predict(dtest)
auc = roc_auc_score(y_test, pred_test)
print(f"[model] AUC on hold-out: {auc:.3f}")

preds_bin = (pred_test > 0.5).astype(int)
cm = confusion_matrix(y_test, preds_bin)
print("[model] confusion matrix:\n", cm)

# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────
def importance_frame(booster: xgb.Booster, imp_type: str) -> pd.DataFrame:
    imp = booster.get_score(importance_type=imp_type)
    return (
        pd.DataFrame({"feature": list(imp.keys()), imp_type: list(imp.values())})
        .sort_values(imp_type, ascending=False)
        .reset_index(drop=True)
    )

imp_gain   = importance_frame(bst, "gain")
imp_cover  = importance_frame(bst, "cover")
imp_weight = importance_frame(bst, "weight")

importance_matrix = (
    imp_gain
    .merge(imp_cover, on="feature", how="outer")
    .merge(imp_weight, on="feature", how="outer")
    .fillna(0)
    .sort_values("gain", ascending=False)
)

top_n = 30
print(f"\n[model] Feature-importance matrix (top {top_n} by gain):")
print(importance_matrix.head(top_n).to_string(index=False))

importance_matrix.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
print(f"[model] Full matrix saved → {MODEL_DIR/'feature_importance.csv'}")
