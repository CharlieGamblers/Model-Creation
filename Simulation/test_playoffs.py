# Simulation/test_playoffs.py

import pandas as pd

from data_io import (
    load_schedule,
    load_rosters,
    load_bulls,
    build_roster_map,
)
import prob_models
from prob_models import (
    load_rider_ratings,
    load_matchup_probs,
    build_rider_prob_map,
    build_matchup_prob_map,
)
from seasons import simulate_season
from playoffs import simulate_playoffs

# Paths (same as your main runner)
SCHEDULE_PATH = "schedule.csv"
ROSTERS_PATH = "rosters.csv"
BULLS_PATH = "bulls.csv"
RIDER_RATINGS_PATH = "rider_ratings.csv"
MATCHUP_PROBS_PATH = "rider_bull_probs.csv"


def main():
    # -----------------------------
    # 1. Load core data
    # -----------------------------
    schedule_df = load_schedule(SCHEDULE_PATH)
    rosters_df = load_rosters(ROSTERS_PATH)
    bulls_df = load_bulls(BULLS_PATH)

    # Build roster_map
    roster_map = build_roster_map(rosters_df)

    # Rider ratings + matchup-level probs
    rider_ratings_df = load_rider_ratings(RIDER_RATINGS_PATH)
    prob_models.RIDER_PROB_MAP = build_rider_prob_map(rider_ratings_df)

    matchup_probs_df = load_matchup_probs(MATCHUP_PROBS_PATH)
    prob_models.MATCHUP_PROB_MAP = build_matchup_prob_map(matchup_probs_df)

    # -----------------------------
    # 2. Run one regular season
    # -----------------------------
    game_results_df, standings_df, rides_df = simulate_season(
        schedule_df,
        roster_map,
        bulls_df,
        lineup_size=5,
    )

    print("\n=== REGULAR SEASON STANDINGS (ONE SEASON) ===")
    print(
        standings_df[["team", "wins", "losses", "win_pct"]]
        .sort_values("win_pct", ascending=False)
        .reset_index(drop=True)
    )

    # Quick sanity: should be 10 teams, no NaNs in win_pct
    print("\nTeams in standings:", len(standings_df["team"].unique()))
    print("Any NaN win_pct? ->", standings_df["win_pct"].isna().any())

    # -----------------------------
    # 3. Run playoffs on that season
    # -----------------------------
    playoff_games_df, playoff_rides_df, placements_df = simulate_playoffs(
        standings_df,
        roster_map,
        bulls_df,
        lineup_size=5,
        use_smooth_slots=True,
    )

    print("\n=== PLAYOFF PLACEMENTS (1–10) ===")
    print(placements_df)

    # Sanity checks on placements
    print("\nUnique playoff finishes:", sorted(placements_df["playoff_finish"].unique()))
    print("Number of teams in placements:", len(placements_df))

    # Optional: show a quick peek at playoff games
    print("\n=== FIRST FEW PLAYOFF GAME ROWS ===")
    print(playoff_games_df)

    print("\n=== FIRST FEW PLAYOFF RIDES ===")
    print(playoff_rides_df.head())


if __name__ == "__main__":
    main()
