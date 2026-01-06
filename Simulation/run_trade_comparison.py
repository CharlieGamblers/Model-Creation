# run_trade_comparison.py

import argparse
import numpy as np
import pandas as pd

from data_io import (
    load_schedule,
    load_rosters,
    load_bulls,
    build_roster_map,
)
import prob_models
from trades import run_trade_comparison

RNG_SEED = 42

# File paths (adjust if needed)
SCHEDULE_PATH = "schedule.csv"
ROSTERS_PATH = "rosters.csv"
BULLS_PATH = "bulls.csv"
RIDER_RATINGS_PATH = "rider_ratings.csv"
MATCHUP_PROBS_PATH = "rider_bull_probs.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run trade comparison over N season simulations."
    )
    parser.add_argument(
        "rider_a",
        type=str,
        help="Name of rider A (as it appears in rosters/rider_ratings).",
    )
    parser.add_argument(
        "rider_b",
        type=str,
        help="Name of rider B (as it appears in rosters/rider_ratings).",
    )
    parser.add_argument(
        "--sims",
        "-n",
        type=int,
        default=1000,
        help="Number of seasons to simulate for each scenario (default: 1000).",
    )
    parser.add_argument(
        "--focus-team",
        type=str,
        default=None,
        help="Optional: team code to print a focused row (e.g., 'aus').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    rider_a = args.rider_a
    rider_b = args.rider_b
    n_sims = args.sims
    focus_team = args.focus_team

    print(f"\n=== RUNNING TRADE COMPARISON ===")
    print(f"Rider A: {rider_a}")
    print(f"Rider B: {rider_b}")
    print(f"Simulations per scenario: {n_sims}\n")

    # =========================================================
    # 1. LOAD CORE DATA
    # =========================================================
    schedule_df = load_schedule(SCHEDULE_PATH)
    rosters_df = load_rosters(ROSTERS_PATH)
    bulls_df = load_bulls(BULLS_PATH)
    roster_map = build_roster_map(rosters_df)

    # =========================================================
    # 2. LOAD PROBABILITY MODELS (SET GLOBAL MAPS)
    # =========================================================
    rider_ratings_df = prob_models.load_rider_ratings(RIDER_RATINGS_PATH)
    prob_models.RIDER_PROB_MAP = prob_models.build_rider_prob_map(rider_ratings_df)

    matchup_probs_df = prob_models.load_matchup_probs(MATCHUP_PROBS_PATH)
    prob_models.MATCHUP_PROB_MAP = prob_models.build_matchup_prob_map(matchup_probs_df)

    # =========================================================
    # 3. RUN TRADE COMPARISON
    # =========================================================
    comparison_df = run_trade_comparison(
        rider_a=rider_a,
        rider_b=rider_b,
        n_sims=n_sims,
        schedule_df=schedule_df,
        roster_map=roster_map,
        bulls_df=bulls_df,
        seed=RNG_SEED,
    )

    # Sort by wins delta (biggest positive impact first)
    comparison_df = comparison_df.sort_values("delta_wins", ascending=False)

    print("\n=== TRADE COMPARISON (ALL TEAMS) ===")
    print(comparison_df)

    # =========================================================
    # 4. OPTIONAL: FOCUSED VIEW
    # =========================================================
    if focus_team is not None:
        ft = focus_team.lower().strip()
        focus_rows = comparison_df[comparison_df["team"].str.lower() == ft]
        if not focus_rows.empty:
            print(f"\n=== FOCUSED VIEW: TEAM = {focus_team} ===")
            print(focus_rows)
        else:
            print(f"\n[INFO] Focus team '{focus_team}' not found in results.")

    # =========================================================
    # 5. OPTIONAL: SAVE TO CSV
    # =========================================================
    # import os
    # os.makedirs("outputs", exist_ok=True)
    # out_path = f"outputs/trade_{rider_a.replace(' ', '_')}_for_{rider_b.replace(' ', '_')}.csv"
    # comparison_df.to_csv(out_path, index=False)
    # print(f"\nSaved full comparison to {out_path}")


if __name__ == "__main__":
    main()
