from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Predict.predictions import predict_one
from Predict.config import DEFAULT_DATE

# ---- CONFIG ----
RIDER_NAME = "Sage Kimzey"
BULL_NAME  = "49G Ah Hell"
EVENT_DATE = DEFAULT_DATE
SHOW_CATEGORY_BAR = False  # set True to also show a Rider/Bull/Hand stacked bar

# ---- HARD-CODED FEATURE → CATEGORY MAP ----
# Extend this dict as you add features.
FEATURE_CATEGORY_MAP = {
    # Rider
    "r_ROE_positive": "Rider",
    "r_ROE_negative": "Rider",
    "r_pbr_rate": "Rider",
    "r_ars_long": "Rider",
    "r_ars_short": "Rider",
    "r_qrp_weighted": "Rider",
    "new_rider_flag": "Rider",

    # Bull
    "b_ROE_positive": "Bull",
    "b_ROE_negative": "Bull",
    "b_abs": "Bull",
    "b_abs_short": "Bull",
    "b_ars": "Bull",
    "b_ars_short": "Bull",
    "b_qrp_smooth": "Bull",
    "b_qrp_short": "Bull",
    "new_bull_flag": "Bull",
    "few_rides_flag": "Bull",
    "few_rides_flag_hand": "Bull",
    "out": "Bull",

    # Hand / Matchup
    "h_abs": "Hand",
    "h_abs_short": "Hand",
    "h_ars": "Hand",
    "h_ars_short": "Hand",
    "h_qrp": "Hand",
    "h_qrp_short": "Hand",
}

def get_category(name: str) -> str:
    return FEATURE_CATEGORY_MAP.get(name, "Other")

if __name__ == "__main__":
    # ---- RUN PREDICTION ----
    result = predict_one(
        rider_name=RIDER_NAME,
        bull_name=BULL_NAME,
        event_date=EVENT_DATE,
    )

    # ---- PRINT FEATURE VALUES ----
    prob = float(result["probability"])
    base_prob = float(result["base_probability"])
    print(f"\n=== Feature Values for {RIDER_NAME} on {BULL_NAME} ===")
    print(f"Predicted Probability: {prob:.4f} ({prob*100:.2f}%)")
    print(f"Baseline Probability:  {base_prob:.4f} ({base_prob*100:.2f}%)")
    print("\nFeature Values:")
    for feature, value in zip(result["features"], result["feature_values"]):
        print(f"  {feature:<30} {value}")

    # ---- SHAP PLOT ----
    features = result["features"]
    shap_values = np.array(result["shap_values"], dtype=float)

    # Scale SHAP contributions to probability space
    prob_diff = prob - base_prob
    tot_shift = shap_values.sum()
    delta_prob = np.zeros_like(shap_values) if np.isclose(tot_shift, 0.0) else shap_values * (prob_diff / tot_shift)

    order = np.argsort(-np.abs(delta_prob))
    features_sorted = np.array(features)[order]
    delta_sorted = delta_prob[order] * 100  # %-points

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(features_sorted))
    ax.barh(y_pos, delta_sorted, color=np.where(delta_sorted > 0, "crimson", "steelblue"))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_sorted)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Δ-probability (percentage-points)")
    ax.set_title(f"{RIDER_NAME} on {BULL_NAME}\nP = {prob*100:.1f}%   (baseline {base_prob*100:.1f}%)")
    plt.tight_layout()
    plt.show()

    # ---- CATEGORY EXPORT (Rider / Bull / Hand) ----
    # Build feature-level table in probability space
    delta_pp = delta_prob * 100.0  # signed %-points
    df = pd.DataFrame({
        "feature": features,
        "delta_pp": delta_pp,
        "delta_pp_abs": np.abs(delta_pp),
        "shap_raw": shap_values,
    })
    df["category"] = df["feature"].map(get_category)

    # Aggregate by category
    cat = (
        df.groupby("category", as_index=False)
          .agg(delta_pp=("delta_pp", "sum"),
               pos_pp=("delta_pp", lambda x: x[x > 0].sum()),
               neg_pp=("delta_pp", lambda x: x[x < 0].sum()),
               abs_pp=("delta_pp_abs", "sum"),
               features=("feature", "count"))
          .sort_values("delta_pp", ascending=False)
    )

    total_shift_pp = df["delta_pp"].sum()
    total_abs_pp   = df["delta_pp_abs"].sum()
    cat["share_of_net"] = np.where(np.isclose(total_shift_pp, 0), 0.0, cat["delta_pp"] / total_shift_pp)
    cat["share_of_abs"] = np.where(np.isclose(total_abs_pp,   0), 0.0, cat["abs_pp"]   / total_abs_pp)

    # Print a concise summary
    print("\n=== Category SHAP Summary (probability space, %-points) ===")
    print(f"Predicted: {prob*100:.2f}%   Baseline: {base_prob*100:.2f}%   Net shift: {total_shift_pp:.2f} pp")
    for _, row in cat.iterrows():
        print(f"  {row['category']:<6} | net: {row['delta_pp']:>6.2f} pp"
              f"  ( +{row['pos_pp']:>5.2f} / {row['neg_pp']:>6.2f} )"
              f"  | abs: {row['abs_pp']:>6.2f} pp"
              f"  | share(abs): {row['share_of_abs']*100:>5.1f}%"
              f"  | nfeat: {int(row['features'])}")

    # Safe filenames
    safe_rider = RIDER_NAME.replace(" ", "_")
    safe_bull  = BULL_NAME.replace(" ", "_").replace("/", "-")
    date_str = str(EVENT_DATE)

    # Export CSVs
    cat_out = cat.copy()
    cat_out.insert(0, "rider", RIDER_NAME)
    cat_out.insert(1, "bull", BULL_NAME)
    cat_out.insert(2, "event_date", date_str)
    cat_out.to_csv(f"shap_breakdown_{safe_rider}_vs_{safe_bull}_{date_str}.csv", index=False)

    df_out = df.copy()
    df_out.insert(0, "rider", RIDER_NAME)
    df_out.insert(1, "bull", BULL_NAME)
    df_out.insert(2, "event_date", date_str)
    df_out.to_csv(f"shap_features_{safe_rider}_vs_{safe_bull}_{date_str}.csv", index=False)

    # Optional: quick Rider vs Bull vs Hand bar
    if SHOW_CATEGORY_BAR:
        cat_plot = cat.set_index("category").reindex(["Rider", "Bull", "Hand", "Other"]).fillna(0.0)
        plt.figure(figsize=(6, 3.8))
        plt.bar(cat_plot.index, cat_plot["delta_pp"])
        plt.axhline(0, color="black", lw=0.8)
        plt.ylabel("Net Δ-probability (pp)")
        plt.title("Category contributions (Rider / Bull / Hand)")
        plt.tight_layout()
        plt.show()


