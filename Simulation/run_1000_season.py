# run_1000_season.py

import numpy as np
import pandas as pd

from data_io import (
    load_schedule,
    load_rosters,
    load_bulls,
    load_rider_ratings,
    load_matchup_probs,
    build_roster_map,
)

import prob_models
from seasons import simulate_many_seasons, compute_mvp_standings
from data_io import REGULAR_PRIZE, PLAYOFF_PRIZE, EXTRA_WIN_POOL
from earnings import compute_rider_earnings

RNG_SEED = 42

# File paths (change if needed)
SCHEDULE_PATH = "schedule.csv"
ROSTERS_PATH = "rosters.csv"
BULLS_PATH = "bulls.csv"
RIDER_RATINGS_PATH = "rider_ratings.csv"
MATCHUP_PROBS_PATH = "rider_bull_probs.csv"


def main():
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

    rider_ratings_df = load_rider_ratings(RIDER_RATINGS_PATH)
    prob_models.RIDER_PROB_MAP = prob_models.build_rider_prob_map(rider_ratings_df)

    matchup_probs_df = load_matchup_probs(MATCHUP_PROBS_PATH)
    prob_models.MATCHUP_PROB_MAP = prob_models.build_matchup_prob_map(matchup_probs_df)


    # =========================================================
    # 3. RUN 1000 SEASONS
    # =========================================================
    n_sims = 1000
    many_standings_df, many_mvp_df, many_playoffs_df, many_prizes_df = simulate_many_seasons(
        n_sims,
        schedule_df,
        roster_map,
        bulls_df,
        seed=RNG_SEED,
        lineup_size=5,
        use_smooth_slots=True,
    )

    # =========================================================
    # 4. Rider Earnings Summary
    # =========================================================

    rider_season_earnings, rider_earnings_summary = compute_rider_earnings(
        many_mvp_df,
        many_prizes_df,
        contribution_col="rides_made",   # or "total_rides" or a custom metric
    )

    aus_earnings = (
        rider_earnings_summary
        .query("team == 'aus'")
        .sort_values("mean_rider_prize", ascending=False)
        .reset_index(drop=True)
    )

    print("\n=== AUSTIN GAMBLERS – Rider Value Summary ===")
    print(
        aus_earnings[
            [
                "rider",
                "seasons_appeared",
                "mean_rider_prize",
                "median_rider_prize",
                "p25_rider_prize",
                "p75_rider_prize",
                "iqr_rider_prize",
            ]
        ]
    )

    print("\n=== RIDER EARNINGS SUMMARY (TOP 25 BY MEAN PRIZE) ===")
    print(
        rider_earnings_summary
        .sort_values("mean_rider_prize", ascending=False)
        .head(25)
    )

    # =========================================================
    # 5. TEAM SUMMARY: AVG RESULTS + 1st / TOP3 / TOP5
    # =========================================================
    team_summary = (
        many_standings_df
        .groupby("team", as_index=False)
        .agg(
            avg_wins=("wins", "mean"),
            avg_losses=("losses", "mean"),
            avg_win_pct=("win_pct", "mean"),
            avg_ride_pct=("team_ride_pct", "mean")
                if "team_ride_pct" in many_standings_df.columns
                else ("wins", lambda x: np.nan),
            best_win_pct=("win_pct", "max"),
            worst_win_pct=("win_pct", "min"),
            first_place_pct=("rank", lambda x: np.mean(x == 1)),
            top3_pct=("rank", lambda x: np.mean(x <= 3)),
            top5_pct=("rank", lambda x: np.mean(x <= 5)),
        )
        .sort_values("avg_wins", ascending=False)
    )

    print("\n=== TEAM AVERAGE RESULTS ACROSS 1000 SEASONS ===")
    print(team_summary)

    # =========================================================
    # 6. RIDER MVP SUMMARY
    # =========================================================
    rider_summary = (
        many_mvp_df
        .groupby(["rider", "team"], as_index=False)
        .agg(
            seasons_appeared=("season_id", "nunique"),
            avg_rank=("rank", "mean"),
            avg_ride_pct=("ride_pct", "mean"),
            avg_total_rides=("total_rides", "mean"),
            first_pct=("rank", lambda x: np.mean(x == 1)),
            top3_pct=("rank", lambda x: np.mean(x <= 3)),
            top5_pct=("rank", lambda x: np.mean(x <= 5)),
            top10_pct=("rank", lambda x: np.mean(x <= 10)),
        )
        .sort_values("avg_rank", ascending=True)
    )

    print("\n=== RIDER MVP SUMMARY (TOP 25 BY AVG RANK) ===")
    print(rider_summary.head(25))

    # =========================================================
    # 7. LEAGUE-WIDE RIDE % PER SEASON
    # =========================================================
    league_pct_per_season = (
        many_mvp_df.groupby("season_id")["rides_made"].sum()
        / many_mvp_df.groupby("season_id")["total_rides"].sum()
    )

    print("\n=== LEAGUE-WIDE RIDE % (ACROSS 1000 SEASONS) ===")
    print("Mean:", round(league_pct_per_season.mean(), 4))
    print("Std Dev:", round(league_pct_per_season.std(), 4))
    print("Min:", round(league_pct_per_season.min(), 4))
    print("Max:", round(league_pct_per_season.max(), 4))

    # =========================================================
    # 8. PLAYOFF SUMMARY
    # =========================================================
    playoff_summary = (
        many_playoffs_df
        .groupby("team", as_index=False)
        .agg(
            avg_playoff_finish=("playoff_finish", "mean"),
            first_pct=("playoff_finish", lambda x: np.mean(x == 1)),
            top2_pct=("playoff_finish", lambda x: np.mean(x <= 2)),
            top4_pct=("playoff_finish", lambda x: np.mean(x <= 4)),
            bottom3_pct=("playoff_finish", lambda x: np.mean(x >= 8)),
        )
        .sort_values("avg_playoff_finish")
    )

    print("\n=== PLAYOFF FINISH DISTRIBUTION ACROSS 1000 SEASONS ===")
    print(playoff_summary)

    # =========================================================
    # 9. PRIZE SUMMARY
    # =========================================================
    def p25(x: pd.Series) -> float:
        return np.percentile(x, 25)

    def p75(x: pd.Series) -> float:
        return np.percentile(x, 75)

    prize_summary = (
        many_prizes_df
        .groupby("team", as_index=False)
        .agg(
            mean_total_prize=("total_prize", "mean"),
            median_total_prize=("total_prize", "median"),
            p25_total_prize=("total_prize", p25),
            p75_total_prize=("total_prize", p75),
            iqr_total_prize=("total_prize", lambda x: p75(x) - p25(x)),
            mean_regular_prize=("regular_prize", "mean"),
            mean_playoff_prize=("playoff_prize", "mean"),
            mean_extra_bonus=("extra_win_bonus", "mean"),
        )
        .sort_values("mean_total_prize", ascending=False)
    )

    # Round monetary columns to 2 decimals
    prize_summary = prize_summary.round(2)

    print("\n=== TEAM PRIZE SUMMARY ACROSS 1000 SEASONS ===")
    print(prize_summary)

    # =========================================================
    # 10. OPTIONAL: SAVE OUTPUTS TO CSV
    # =========================================================
    # Create an outputs folder if you want
    # import os
    # os.makedirs("outputs", exist_ok=True)
    #
    # team_summary.to_csv("outputs/team_summary_1000sims.csv", index=False)
    # rider_summary.to_csv("outputs/rider_summary_1000sims.csv", index=False)


if __name__ == "__main__":
    main()
