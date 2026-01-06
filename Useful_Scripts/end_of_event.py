# end_of_event.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import unicodedata
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    auc as sklearn_auc,
    roc_curve,
)

# Add parent directory to path to import Predict module
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from Predict.predictions import predict_one
from Predict.config import DEFAULT_DATE, load_model, model_feature_names, FINAL_DATA, FEATURE_LIST, MODEL_FILE, RIDER_XLSX, BULL_XLSX
from Predict.feature_engineering import solo_data_pull, load_average_bull_features
import xgboost as xgb

# Import private functions needed for baseline predictions
from Predict.predictions import (
    _load_id_maps,
    _load_rider_id_maps,
    _normalize_rider_name,
    _normalize_bull_name,
    _align_to_features,
)


def fix_encoding_artifacts(text: str) -> str:
    """
    Fix common encoding artifacts like 'Ã£' -> 'a'.
    Handles cases where UTF-8 encoded characters were incorrectly decoded as Latin-1.
    Converts all accented characters to plain ASCII equivalents.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Common encoding fix mappings for UTF-8->Latin-1 mis-decoding
    # These patterns occur when UTF-8 bytes are read as Latin-1
    encoding_fixes = {
        'Ã£': 'a',   # ã -> a
        'Ã¡': 'a',   # á -> a
        'Ã©': 'e',   # é -> e
        'Ã­': 'i',   # í -> i
        'Ã³': 'o',   # ó -> o
        'Ãº': 'u',   # ú -> u
        'Ã§': 'c',   # ç -> c
        'Ã': 'a',    # à -> a (may have duplicates, but that's ok)
        'Ã': 'A',
        'Ã': 'a',
        'Ã': 'A',
        'Ã': 'e',
        'Ã': 'E',
        'Ã': 'i',
        'Ã': 'I',
        'Ã': 'o',
        'Ã': 'O',
        'Ã': 'u',
        'Ã': 'U',
    }
    
    # Apply encoding fixes (order matters - do multi-char patterns first)
    for bad, good in sorted(encoding_fixes.items(), key=lambda x: -len(x[0])):
        text = text.replace(bad, good)
    
    # Normalize unicode (NFKD) and remove combining characters to convert accented to plain
    # This handles any remaining accented characters and converts them to ASCII
    try:
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
    except Exception:
        pass  # If normalization fails, continue with the text as-is
    
    return text


def find_header_row(raw: pd.DataFrame) -> int:
    """
    Finds the row index that contains the header 'R#' (as in your export).
    """
    for i in range(min(len(raw), 200)):  # safety cap
        row = raw.iloc[i].astype(str).str.strip()
        if (row == "R#").any():
            return i
    raise ValueError("Could not find header row containing 'R#'.")


def normalize_headers(header_row: pd.Series) -> list[str]:
    """
    Turns the header row into usable column names, filling blanks as col_{i}.
    """
    cols = []
    for i, v in enumerate(header_row.tolist()):
        if pd.isna(v) or str(v).strip() == "" or str(v).strip().lower() == "nan":
            cols.append(f"col_{i}")
        else:
            cols.append(str(v).strip())
    return cols

def parse_bull_brand_name(bull_cell: str) -> str:
    """
    For 2-line cells:

    Bull number rules (locked):
      - Bull number = text after the last dash that is NOT just a formatting dash
      - If line1 ends with '-', that dash is part of the bull number

    Examples:
      MDBB-38H-   -> 38H-
      C-B-035     -> 035
      X-Y-7-      -> 7-
      ABC-12A     -> 12A
      ABC12       -> ABC12
      BSHK-7-     -> 7-
    """
    if bull_cell is None or (isinstance(bull_cell, float) and pd.isna(bull_cell)):
        return ""

    s = str(bull_cell).strip()
    if not s:
        return ""

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return ""

    # Single-line: return as-is
    if len(lines) == 1:
        return lines[0]

    line1 = lines[0]
    bull_name = lines[1]

    # No dash case
    if "-" not in line1:
        bull_no = line1.strip()
        return f"{bull_no} {bull_name}".strip()

    # Ends with dash → formatting dash, ignore it for searching
    if line1.endswith("-"):
        trimmed = line1[:-1]  # remove final dash
        last_dash = trimmed.rfind("-")
        if last_dash == -1:
            bull_no = trimmed + "-"
        else:
            bull_no = trimmed[last_dash + 1 :] + "-"
    else:
        last_dash = line1.rfind("-")
        bull_no = line1[last_dash + 1 :]

    return f"{bull_no.strip()} {bull_name}".strip()


def build_table(excel_path: Path, event_date: str = DEFAULT_DATE) -> pd.DataFrame:
    raw = pd.read_excel(excel_path, sheet_name="JudgeRideScores_6Judges", header=None)

    hdr_i = find_header_row(raw)
    headers = normalize_headers(raw.iloc[hdr_i])
    df = raw.iloc[hdr_i + 1 :].copy()
    df.columns = headers

    # Drop fully empty rows
    df = df.dropna(how="all")

    # Core fields (as they appear in your export)
    required = ["R#", "Rider Name", "Bull No. / Name", "Rider\nScore"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}. Found columns: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["Round"] = df["R#"].apply(
        lambda x: f"R{int(x)}" if pd.notna(x) and str(x).strip() != "" else ""
    )
    # Fix encoding artifacts in rider names and bull names
    out["Rider Name"] = df["Rider Name"].astype(str).apply(fix_encoding_artifacts).str.strip()
    out["Bull"] = df["Bull No. / Name"].apply(parse_bull_brand_name).apply(fix_encoding_artifacts)
    out["Rider Score"] = pd.to_numeric(df["Rider\nScore"], errors="coerce")

    # RR columns exist in your file as:
    # - 'RR\nOptn'
    # - 'RR\nTakn'
    rr_optn_col = "RR\nOptn" if "RR\nOptn" in df.columns else None
    rr_takn_col = "RR\nTakn" if "RR\nTakn" in df.columns else None

    if rr_optn_col or rr_takn_col:
        rr_vals = []
        if rr_optn_col:
            rr_vals.append(df[rr_optn_col].astype(str).str.upper().str.strip())
        if rr_takn_col:
            rr_vals.append(df[rr_takn_col].astype(str).str.upper().str.strip())

        has_y = rr_vals[0].eq("Y")
        for s in rr_vals[1:]:
            has_y = has_y | s.eq("Y")
    else:
        has_y = pd.Series(False, index=df.index)

    # Duplicate ride = same (Round, Rider Name) appears more than once
    dup_group = df.groupby(["R#", "Rider Name"]).size()
    dup_keys = set(k for k, n in dup_group.items() if n > 1)

    is_dup = df.apply(lambda r: (r["R#"], r["Rider Name"]) in dup_keys, axis=1)

    # RR_Flag rule:
    # if rider has two rides in the same round, mark the one WITHOUT 'Y'
    # in the RR columns as 1 and all other rides as 0
    out["RR_Flag"] = ((is_dup) & (~has_y)).astype(int)

    # If a rider has duplicate rides in a round, keep ONLY the RR row(s) (RR_Flag==1)
    out["_is_dup_group"] = is_dup.values
    out = out[(~out["_is_dup_group"]) | (out["RR_Flag"] == 1)].copy()
    out = out.drop(columns=["_is_dup_group"])

    # Final cleanup
    out = out[(out["Round"] != "") & (out["Rider Name"].str.lower() != "nan")]

    out = out.reset_index(drop=True)
    
    # Add ride or no ride column (1 if Rider Score > 0, else 0)
    out["Ride_Flag"] = (out["Rider Score"] > 0).astype(int)
    
    # Add predictions for bull vs rider (with baseline support)
    print(f"Generating predictions for {len(out)} rider-bull pairs...")
    
    # Load model components for baseline predictions
    final_data = pd.read_csv(FINAL_DATA, parse_dates=["event_start_date"], low_memory=False)
    bst = load_model(MODEL_FILE)
    features = list(model_feature_names(bst, FEATURE_LIST))
    rider_dict, bull_dict = _load_id_maps()
    rid_internal_map, rid_legacy_map = _load_rider_id_maps()
    avg_bull_features = load_average_bull_features()
    start_date = pd.to_datetime(event_date)
    
    predictions = []
    prediction_probs = []
    baseline_flags = []
    
    for idx, row in out.iterrows():
        rider_name = row["Rider Name"]
        bull_name = row["Bull"]
        used_baseline = False
        
        try:
            # Check if rider and bull are in lookup maps
            rider_key = _normalize_rider_name(rider_name)
            rid_internal = rid_internal_map.get(rider_key)
            rid_legacy = rid_legacy_map.get(rider_key) if rid_internal is None else None
            bull_key = _normalize_bull_name(bull_name)
            bid = bull_dict.get(bull_key)
            
            if rid_internal is None and rid_legacy is None:
                # Rider not found - skip prediction
                predictions.append("")
                prediction_probs.append(None)
                baseline_flags.append(0)
                if idx < 5 or idx % 50 == 0:
                    print(f"  Warning: Rider not found: '{rider_name}'")
                continue
            
            # Try actual prediction first
            if bid is not None:
                # Bull found - use actual prediction
                pred_result = predict_one(rider_name, bull_name, event_date=event_date)
                prob = pred_result["probability"]
                used_baseline = False
            else:
                # Bull not found - use baseline prediction with average bull features
                used_baseline = True
                if idx < 5 or idx % 50 == 0:
                    print(f"  Baseline: Bull not found '{bull_name}'. Using average bull features.")
                
                # Build rider features with dummy bull, then replace bull features with averages
                if rid_internal is not None:
                    row_df = solo_data_pull(
                        final_data,
                        rider_id=None,
                        bull_id=-1,  # Dummy ID for missing bull
                        rider_internal_id=rid_internal,
                        start_date=start_date,
                    )
                else:
                    row_df = solo_data_pull(
                        final_data,
                        rider_id=rid_legacy,
                        bull_id=-1,  # Dummy ID for missing bull
                        rider_internal_id=None,
                        start_date=start_date,
                    )
                
                # Overwrite bull columns with average bull features
                for col in avg_bull_features.index:
                    if col in row_df.columns:
                        row_df[col] = avg_bull_features[col]
                
                # Align to model features and predict
                row_df = _align_to_features(row_df, features)
                dm = xgb.DMatrix(row_df.values, feature_names=features)
                prob = float(bst.predict(dm)[0])
            
            prediction_probs.append(prob)
            # If probability > 0.5, rider wins (successfully rides), else bull wins
            predictions.append("Rider" if prob > 0.5 else "Bull")
            baseline_flags.append(1 if used_baseline else 0)
            
        except Exception as e:
            # If error occurs, mark as missing
            predictions.append("")
            prediction_probs.append(None)
            baseline_flags.append(0)
            if idx < 5 or idx % 50 == 0:
                print(f"  Warning: Could not predict for {rider_name} vs {bull_name}: {e}")
    
    out["Prediction"] = predictions
    out["Prediction_Probability"] = prediction_probs
    out["Baseline_Prediction"] = baseline_flags
    
    print(f"Predictions complete.")
    
    return out


def calculate_evaluation_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Calculate comprehensive evaluation metrics for predictions.
    Returns a tuple: (metrics_df, extra_data_dict) where extra_data contains
    calibration, lift, and decision curve data.
    """
    # Filter out rows with missing predictions or actual outcomes
    valid_mask = df["Prediction_Probability"].notna() & df["Ride_Flag"].notna()
    if valid_mask.sum() == 0:
        empty_df = pd.DataFrame({"Category": ["Error"], "Metric": ["No valid predictions"], "Value": [None]})
        return empty_df, {}
    
    y_true = df.loc[valid_mask, "Ride_Flag"].astype(int)
    y_pred_proba = df.loc[valid_mask, "Prediction_Probability"].astype(float)
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    
    metrics_rows = []
    
    # Basic Classification Metrics
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        metrics_rows.append({"Category": "Classification", "Metric": "AUC (ROC-AUC)", "Value": roc_auc})
    except Exception as e:
        metrics_rows.append({"Category": "Classification", "Metric": "AUC (ROC-AUC)", "Value": f"Error: {e}"})
    
    try:
        pr_auc = sklearn_auc(*precision_recall_curve(y_true, y_pred_proba)[:2])
        metrics_rows.append({"Category": "Classification", "Metric": "PR-AUC", "Value": pr_auc})
    except Exception as e:
        metrics_rows.append({"Category": "Classification", "Metric": "PR-AUC", "Value": f"Error: {e}"})
    
    # Confusion Matrix at 0.5 threshold
    try:
        cm = confusion_matrix(y_true, y_pred_binary)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        metrics_rows.append({"Category": "Confusion Matrix (0.5)", "Metric": "True Negatives", "Value": int(tn)})
        metrics_rows.append({"Category": "Confusion Matrix (0.5)", "Metric": "False Positives", "Value": int(fp)})
        metrics_rows.append({"Category": "Confusion Matrix (0.5)", "Metric": "False Negatives", "Value": int(fn)})
        metrics_rows.append({"Category": "Confusion Matrix (0.5)", "Metric": "True Positives", "Value": int(tp)})
        metrics_rows.append({"Category": "Confusion Matrix (0.5)", "Metric": "Accuracy", "Value": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0})
        metrics_rows.append({"Category": "Confusion Matrix (0.5)", "Metric": "Precision", "Value": tp / (tp + fp) if (tp + fp) > 0 else 0})
        metrics_rows.append({"Category": "Confusion Matrix (0.5)", "Metric": "Recall", "Value": tp / (tp + fn) if (tp + fn) > 0 else 0})
        metrics_rows.append({"Category": "Confusion Matrix (0.5)", "Metric": "F1-Score", "Value": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0})
    except Exception as e:
        metrics_rows.append({"Category": "Confusion Matrix (0.5)", "Metric": "Error", "Value": str(e)})
    
    # Probability Metrics
    try:
        brier = brier_score_loss(y_true, y_pred_proba)
        metrics_rows.append({"Category": "Probability", "Metric": "Brier Score", "Value": brier})
    except Exception as e:
        metrics_rows.append({"Category": "Probability", "Metric": "Brier Score", "Value": f"Error: {e}"})
    
    try:
        logloss = log_loss(y_true, y_pred_proba)
        metrics_rows.append({"Category": "Probability", "Metric": "Log Loss", "Value": logloss})
    except Exception as e:
        metrics_rows.append({"Category": "Probability", "Metric": "Log Loss", "Value": f"Error: {e}"})
    
    # Calibration (simplified - using ECE)
    calibration_data = []
    try:
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        y_pred_proba_array = y_pred_proba.values
        y_true_array = y_true.values
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba_array > bin_lower) & (y_pred_proba_array <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0 and in_bin.any():
                accuracy_in_bin = y_true_array[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba_array[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                calibration_data.append({
                    "Bin": f"{bin_lower:.2f}-{bin_upper:.2f}",
                    "Count": int(in_bin.sum()),
                    "Mean Predicted": avg_confidence_in_bin,
                    "Mean Actual": accuracy_in_bin,
                    "Difference": avg_confidence_in_bin - accuracy_in_bin
                })
        
        metrics_rows.append({"Category": "Calibration", "Metric": "ECE (Expected Calibration Error)", "Value": ece})
    except Exception as e:
        metrics_rows.append({"Category": "Calibration", "Metric": "ECE", "Value": f"Error: {e}"})
    
    # Top-K Hit Rate
    try:
        k_values = [1, 3, 5, 10]
        n_total = len(y_true)
        if n_total > 0:
            sorted_indices = np.argsort(y_pred_proba.values)[::-1]
            y_true_array = y_true.values
            for k in k_values:
                if k <= n_total:
                    top_k_indices = sorted_indices[:k]
                    hit_rate = y_true_array[top_k_indices].sum() / min(k, y_true_array.sum()) if y_true_array.sum() > 0 else 0
                    metrics_rows.append({"Category": "Top-K Hit Rate", "Metric": f"Top-{k} Hit Rate", "Value": hit_rate})
    except Exception as e:
        metrics_rows.append({"Category": "Top-K Hit Rate", "Metric": "Error", "Value": str(e)})
    
    # Lift
    lift_data = []
    try:
        n_bins_lift = 10
        sorted_idx = np.argsort(y_pred_proba.values)[::-1]
        bin_size = len(y_true) // n_bins_lift
        
        overall_rate = y_true.mean()
        y_true_array = y_true.values
        
        for i in range(n_bins_lift):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins_lift - 1 else len(y_true)
            bin_indices = sorted_idx[start_idx:end_idx]
            
            if len(bin_indices) > 0:
                bin_rate = y_true_array[bin_indices].mean()
                lift = bin_rate / overall_rate if overall_rate > 0 else 0
                lift_data.append({
                    "Decile": i + 1,
                    "Count": len(bin_indices),
                    "Response Rate": bin_rate,
                    "Lift": lift
                })
        
        # Average lift for top decile
        if lift_data:
            top_decile_lift = lift_data[0]["Lift"] if lift_data else 0
            metrics_rows.append({"Category": "Lift", "Metric": "Top Decile Lift", "Value": top_decile_lift})
    except Exception as e:
        metrics_rows.append({"Category": "Lift", "Metric": "Error", "Value": str(e)})
    
    # Decision Curve Analysis (simplified - net benefit at various thresholds)
    net_benefit_data = []
    try:
        thresholds = np.arange(0.1, 1.0, 0.1)
        
        n = len(y_true)
        p = y_true.sum() / n if n > 0 else 0  # prevalence
        y_pred_proba_array = y_pred_proba.values
        y_true_array = y_true.values
        
        for threshold in thresholds:
            tp = ((y_pred_proba_array >= threshold) & (y_true_array == 1)).sum()
            fp = ((y_pred_proba_array >= threshold) & (y_true_array == 0)).sum()
            
            # Net benefit = (TP/n) - (FP/n) * (threshold/(1-threshold))
            # Assuming equal harm from false positive and false negative at threshold
            net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold)) if threshold < 1 else (tp / n)
            
            # Treat all (assume all positive)
            treat_all_benefit = p - (1 - p) * (threshold / (1 - threshold)) if threshold < 1 else p
            
            # Treat none (assume all negative)
            treat_none_benefit = 0
            
            net_benefit_data.append({
                "Threshold": threshold,
                "Net Benefit": net_benefit,
                "Treat All Benefit": treat_all_benefit,
                "Treat None Benefit": treat_none_benefit
            })
        
        # Max net benefit
        if net_benefit_data:
            max_net_benefit = max(nb["Net Benefit"] for nb in net_benefit_data)
            metrics_rows.append({"Category": "Decision Curve", "Metric": "Max Net Benefit", "Value": max_net_benefit})
    except Exception as e:
        metrics_rows.append({"Category": "Decision Curve", "Metric": "Error", "Value": str(e)})
        net_benefit_data = []
    
    metrics_df = pd.DataFrame(metrics_rows)
    
    # Store additional data as attributes (using a dict since DataFrame attrs might not persist)
    extra_data = {}
    if 'calibration_data' in locals() and calibration_data:
        extra_data['calibration_data'] = pd.DataFrame(calibration_data)
    if 'lift_data' in locals() and lift_data:
        extra_data['lift_data'] = pd.DataFrame(lift_data)
    if 'net_benefit_data' in locals() and net_benefit_data:
        extra_data['net_benefit_data'] = pd.DataFrame(net_benefit_data)
    
    # Return tuple: (metrics_df, extra_data_dict)
    return metrics_df, extra_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Path to Judge Scores.xlsx")
    ap.add_argument("--out", default="judge_scores_clean.xlsx", help="Output Excel path")
    ap.add_argument("--event-date", default=DEFAULT_DATE, help=f"Event date for predictions (default: {DEFAULT_DATE})")
    args = ap.parse_args()

    excel_path = Path(args.excel)
    out_path = Path(args.out)

    table = build_table(excel_path, event_date=args.event_date)
    
    # Calculate evaluation metrics
    print("Calculating evaluation metrics...")
    metrics_df, extra_data = calculate_evaluation_metrics(table)
    
    # Write to Excel with multiple sheets
    print(f"Writing to Excel file: {out_path.resolve()}")
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        # Main data sheet
        table.to_excel(writer, sheet_name='Predictions', index=False)
        
        # Metrics sheet
        metrics_df.to_excel(writer, sheet_name='Evaluation Metrics', index=False)
        
        # Additional detailed sheets if available
        if 'calibration_data' in extra_data and len(extra_data['calibration_data']) > 0:
            extra_data['calibration_data'].to_excel(writer, sheet_name='Calibration Curve', index=False)
        
        if 'lift_data' in extra_data and len(extra_data['lift_data']) > 0:
            extra_data['lift_data'].to_excel(writer, sheet_name='Lift Analysis', index=False)
        
        if 'net_benefit_data' in extra_data and len(extra_data['net_benefit_data']) > 0:
            extra_data['net_benefit_data'].to_excel(writer, sheet_name='Decision Curve', index=False)

    print(f"Wrote {len(table):,} rows -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
