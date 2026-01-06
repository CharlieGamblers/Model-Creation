# model_train_time_split.py
# Time-based split evaluation on AT+AB (Teams + UTB) + retrain full up-to-date production model on AT+AB.

from __future__ import annotations

from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, log_loss, brier_score_loss
import xgboost as xgb


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap repo root & Predict imports (same approach as model_train.py)
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

# Output artifacts
MODEL_FILE_PROD = MODEL_DIR / "xgb_model.json"         # production (latest)
MODEL_FILE_EVAL = MODEL_DIR / "xgb_model_eval.json"    # evaluation window model
FEAT_FILE       = MODEL_DIR / "feature_cols.txt"
IMP_FILE_PROD   = MODEL_DIR / "feature_importance.csv"
IMP_FILE_EVAL   = MODEL_DIR / "feature_importance_eval.csv"

# Predict config
from Predict.config import DEFAULT_DATE as DAY_BEFORE_EVENT
from Predict.feature_engineering import solo_data_pull


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--test_start",
        type=str,
        default=None,
        help=(
            "ISO date (YYYY-MM-DD) that starts the AT+AB OOS test window. "
            "Train uses dates < test_start. Test uses AT/AB rides with dates >= test_start. "
            "If omitted: uses last 20%% of AT/AB events (by event_start_date) before DAY_BEFORE_EVENT."
        ),
    )
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--num_boost_round", type=int, default=5000)
    ap.add_argument("--early_stopping_rounds", type=int, default=75)
    ap.add_argument("--verbose_eval", type=int, default=100)
    ap.add_argument(
        "--primary_metric",
        choices=["logloss", "auc"],
        default="logloss",
        help="Primary training target metric. For betting-style probs, use logloss (recommended).",
    )
    return ap.parse_args()


def is_at(df: pd.DataFrame) -> pd.Series:
    return df["event_id"].astype(str).str.startswith("AT")


def is_ab(df: pd.DataFrame) -> pd.Series:
    return df["event_id"].astype(str).str.startswith("AB")


def is_atab(df: pd.DataFrame) -> pd.Series:
    return is_at(df) | is_ab(df)


def infer_feature_cols(final_data: pd.DataFrame, cutoff: pd.Timestamp) -> list[str]:
    """Infer feature columns via solo_data_pull probe, mirroring model_train.py behavior."""
    eligible = final_data.loc[:, ["rider_internal_id", "bull_id", "event_start_date"]].dropna()
    eligible = eligible.astype({"rider_internal_id": int, "bull_id": int})
    if eligible.empty:
        raise RuntimeError("No eligible rows to probe feature columns.")

    sample_internal = int(eligible.iloc[0]["rider_internal_id"])
    sample_bull = int(eligible.iloc[0]["bull_id"])

    print("[model] probing feature columns via solo_data_pull …")
    probe = solo_data_pull(
        final_data,
        rider_id=None,  # legacy param
        bull_id=sample_bull,
        rider_internal_id=sample_internal,
        start_date=cutoff,
    )

    non_features = {
        "qr", "rider_id", "rider_internal_id", "bull_id", "event_start_date",
        "rider", "bull", "hand", "pbr", "ride_id", "bull_ride_id", "master_id",
        "event_id", "sanction", "go", "round", "perf", "comments", "rowid",
    }
    feature_cols = [c for c in probe.columns if c not in non_features]

    # Keep compatibility with build columns that may not be in probe output
    for extra in ["b_ROE_positive", "b_ROE_negative", "b_ROE", "r_ROE"]:
        if extra in final_data.columns and extra not in feature_cols:
            feature_cols.append(extra)

    if not feature_cols:
        raise RuntimeError("No feature columns inferred from solo_data_pull.")

    return feature_cols


def make_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame | None,
    y_valid: pd.Series | None,
    feature_cols: list[str],
    seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    verbose_eval: int,
    primary_metric: str,
) -> xgb.Booster:
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)

    evals = [(dtrain, "train")]
    dvalid = None
    if X_valid is not None and y_valid is not None:
        dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=feature_cols)
        evals.append((dvalid, "test"))

    eval_metric = ["logloss", "auc"] if primary_metric == "logloss" else ["auc", "logloss"]

    params = {
        "objective": "binary:logistic",
        "eval_metric": eval_metric,
        "eta": 0.03,
        "max_depth": 5,
        "min_child_weight": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": seed,
        "tree_method": "hist",
    }

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds if dvalid is not None else None,
        verbose_eval=verbose_eval if dvalid is not None else False,
    )
    return model


def feature_importance_matrix(model: xgb.Booster) -> pd.DataFrame:
    score_gain = model.get_score(importance_type="gain")
    score_cover = model.get_score(importance_type="cover")
    score_weight = model.get_score(importance_type="weight")

    imp_gain = pd.DataFrame({"feature": list(score_gain.keys()), "gain": list(score_gain.values())})
    imp_cover = pd.DataFrame({"feature": list(score_cover.keys()), "cover": list(score_cover.values())})
    imp_weight = pd.DataFrame({"feature": list(score_weight.keys()), "weight": list(score_weight.values())})

    return (
        imp_gain
        .merge(imp_cover, on="feature", how="outer")
        .merge(imp_weight, on="feature", how="outer")
        .fillna(0)
        .sort_values("gain", ascending=False)
    )


def main() -> None:
    args = parse_args()

    # Coerce DEFAULT_DATE to Timestamp once (your config returns a string)
    cutoff = pd.to_datetime(DAY_BEFORE_EVENT)

    print("[model] loading final_data …", FINAL_DATA)
    final_data = pd.read_csv(FINAL_DATA, parse_dates=["event_start_date"])
    print("[model] data shape:", final_data.shape)

    required_cols = {"rider_internal_id", "bull_id", "qr", "event_start_date", "event_id"}
    missing = required_cols - set(final_data.columns)
    if missing:
        raise KeyError(f"final_data.csv missing required columns: {sorted(missing)}")

    # Restrict to the level you care about: AT+AB only
    final_data = final_data.loc[is_atab(final_data), :].copy()
    print("[model] after AT+AB filter:", final_data.shape)

    # Enforce global cutoff (no leakage beyond configured date)
    final_data = final_data.loc[final_data["event_start_date"] < cutoff, :].copy()
    print(f"[model] after cutoff (< {cutoff.date()}):", final_data.shape)

    # Infer features once
    feature_cols = infer_feature_cols(final_data, cutoff)
    FEAT_FILE.write_text("\n".join(feature_cols), encoding="utf-8")
    print(f"[model] feature list saved → {FEAT_FILE}  (n={len(feature_cols)})")

    # Ensure all inferred features exist
    for col in feature_cols:
        if col not in final_data.columns:
            final_data[col] = np.nan

    # ─────────────────────────────────────────────────────────────────────────
    # 1) TIME-BASED EVALUATION (AT+AB OOS window)
    # ─────────────────────────────────────────────────────────────────────────
    atab_events = (
        final_data.loc[:, ["event_id", "event_start_date"]]
        .dropna(subset=["event_id", "event_start_date"])
        .drop_duplicates()
        .sort_values("event_start_date")
    )
    if atab_events.empty:
        raise RuntimeError("No AT/AB events found before cutoff.")

    if args.test_start is None:
        n = len(atab_events)
        idx = int(np.floor(n * 0.80))
        idx = min(max(idx, 1), n - 1)
        test_start_dt = pd.to_datetime(atab_events.iloc[idx]["event_start_date"])
        print(f"[model] default test_start = {test_start_dt.date()} (last 20% of AT+AB events as OOS)")
    else:
        test_start_dt = pd.to_datetime(args.test_start)
        print(f"[model] user test_start = {test_start_dt.date()}")

    train_df = final_data.loc[final_data["event_start_date"] < test_start_dt, :].copy()
    test_df  = final_data.loc[final_data["event_start_date"] >= test_start_dt, :].copy()

    print("[eval] eval-train rows:", len(train_df))
    print("[eval] eval-test rows (AT+AB OOS):", len(test_df))

    if train_df.empty or test_df.empty:
        raise RuntimeError("Evaluation split empty. Use a different --test_start.")

    X_train = make_matrix(train_df, feature_cols)
    y_train = train_df["qr"].astype(int)

    X_test = make_matrix(test_df, feature_cols)
    y_test = test_df["qr"].astype(int)

    eval_model = train_xgb(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_test,
        y_valid=y_test,
        feature_cols=feature_cols,
        seed=args.seed,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=args.verbose_eval,
        primary_metric=args.primary_metric,
    )

    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)
    p_test = eval_model.predict(dtest)

    auc = roc_auc_score(y_test, p_test)
    ll = log_loss(y_test, p_test, labels=[0, 1])
    bs = brier_score_loss(y_test, p_test)

    y_hat = (p_test >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_hat)

    print("\n[eval] AT+AB OOS window metrics")
    print(f"  test_start: {test_start_dt.date()}")
    print(f"  AUC      : {auc:.4f}")
    print(f"  LogLoss  : {ll:.4f}")
    print(f"  Brier    : {bs:.4f}")
    print("  Confusion @0.5:\n", cm)

    eval_model.save_model(str(MODEL_FILE_EVAL))
    imp_eval = feature_importance_matrix(eval_model)
    imp_eval.to_csv(IMP_FILE_EVAL, index=False)
    print(f"[eval] saved eval model → {MODEL_FILE_EVAL}")
    print(f"[eval] saved eval importance → {IMP_FILE_EVAL}")

    # ─────────────────────────────────────────────────────────────────────────
    # 2) PRODUCTION RETRAIN (up-to-date model on all AT+AB rows up to cutoff)
    # ─────────────────────────────────────────────────────────────────────────
    prod_df = final_data.copy()
    print("\n[prod] training production model on ALL AT+AB rows up to cutoff …")
    print("[prod] rows:", len(prod_df))

    X_prod = make_matrix(prod_df, feature_cols)
    y_prod = prod_df["qr"].astype(int)

    # Use best_iteration from eval model to set tree count (reasonable default)
    best_iter = getattr(eval_model, "best_iteration", None)
    if best_iter is None:
        prod_rounds = args.num_boost_round
    else:
        prod_rounds = int(best_iter) + 1

    prod_model = train_xgb(
        X_train=X_prod,
        y_train=y_prod,
        X_valid=None,
        y_valid=None,
        feature_cols=feature_cols,
        seed=args.seed,
        num_boost_round=prod_rounds,
        early_stopping_rounds=0,
        verbose_eval=0,
        primary_metric=args.primary_metric,
    )

    prod_model.save_model(str(MODEL_FILE_PROD))
    imp_prod = feature_importance_matrix(prod_model)
    imp_prod.to_csv(IMP_FILE_PROD, index=False)

    print(f"[prod] saved production model → {MODEL_FILE_PROD}")
    print(f"[prod] saved production importance → {IMP_FILE_PROD}")

    top_n = 25
    print(f"\n[prod] Feature-importance (top {top_n} by gain):")
    print(imp_prod.head(top_n).to_string(index=False))


if __name__ == "__main__":
    main()
