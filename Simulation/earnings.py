# earnings.py

import numpy as np
import pandas as pd
from typing import Tuple


def compute_rider_earnings(
    many_mvp_df: pd.DataFrame,
    many_prizes_df: pd.DataFrame,
    contribution_col: str = "rides_made",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Allocate team prize money to riders based on a chosen contribution metric,
    then summarize average earnings per rider.

    Parameters
    ----------
    many_mvp_df : pd.DataFrame
        Long-format rider MVP table across seasons.
        Must contain at least:
            ['season_id', 'team', 'rider', 'rides_made', 'total_rides']
        (and 'contribution_col' if you set it to something else)
    many_prizes_df : pd.DataFrame
        Long-format team prize table across seasons.
        Must contain at least:
            ['season_id', 'team', 'total_prize']
    contribution_col : str, default 'rides_made'
        Column in many_mvp_df to use as the "contribution weight".
        Typical choices:
            - 'rides_made'      (each qualified ride gets equal share)
            - 'total_rides'     (all attempts, regardless of make)
            - or a custom metric you add.

    Returns
    -------
    rider_season_earnings : pd.DataFrame
        One row per (season_id, team, rider) with:
            - season_id
            - team
            - rider
            - contribution
            - team_total_prize
            - rider_prize
    rider_earnings_summary : pd.DataFrame
        One row per (rider, team) with aggregated stats:
            - seasons_appeared
            - mean_rider_prize
            - median_rider_prize
            - p25_rider_prize
            - p75_rider_prize
            - iqr_rider_prize
    """
    required_mvp_cols = {"season_id", "team", "rider", contribution_col}
    missing_mvp = required_mvp_cols - set(many_mvp_df.columns)
    if missing_mvp:
        raise ValueError(
            f"many_mvp_df is missing required columns for earnings calc: {missing_mvp}"
        )

    required_prize_cols = {"season_id", "team", "total_prize"}
    missing_prize = required_prize_cols - set(many_prizes_df.columns)
    if missing_prize:
        raise ValueError(
            f"many_prizes_df is missing required columns: {missing_prize}"
        )

    # 1) Take only the pieces we need from MVP table
    contrib_df = many_mvp_df[["season_id", "team", "rider", contribution_col]].copy()
    contrib_df = contrib_df.rename(columns={contribution_col: "contribution"})

    # 2) Total contribution per team-season
    contrib_df["team_contribution"] = (
        contrib_df.groupby(["season_id", "team"])["contribution"].transform("sum")
    )

    # 3) Merge in team prize for that season
    merged = contrib_df.merge(
        many_prizes_df[["season_id", "team", "total_prize"]],
        on=["season_id", "team"],
        how="left",
    )

    # 4) Compute rider prize = total_prize * contribution / team_contribution
    # Guard against division by zero (e.g. no contributions) or missing prize.
    def _alloc(row):
        team_total = row["total_prize"]
        team_contrib = row["team_contribution"]
        contrib = row["contribution"]

        if pd.isna(team_total) or team_total <= 0 or team_contrib <= 0:
            return 0.0
        return team_total * (contrib / team_contrib)

    merged["rider_prize"] = merged.apply(_alloc, axis=1)

    rider_season_earnings = merged[
        ["season_id", "team", "rider", "contribution", "total_prize", "rider_prize"]
    ].copy()

    # 5) Aggregate across seasons to get a stable "value" per rider
    def p25(x: pd.Series) -> float:
        return float(np.percentile(x, 25))

    def p75(x: pd.Series) -> float:
        return float(np.percentile(x, 75))

    rider_earnings_summary = (
        rider_season_earnings
        .groupby(["rider", "team"], as_index=False)
        .agg(
            seasons_appeared=("season_id", "nunique"),
            mean_rider_prize=("rider_prize", "mean"),
            median_rider_prize=("rider_prize", "median"),
            p25_rider_prize=("rider_prize", p25),
            p75_rider_prize=("rider_prize", p75),
            iqr_rider_prize=("rider_prize", lambda x: p75(x) - p25(x)),
        )
    )

    # Optional: round to cents
    money_cols = [
        "mean_rider_prize",
        "median_rider_prize",
        "p25_rider_prize",
        "p75_rider_prize",
        "iqr_rider_prize",
    ]
    rider_earnings_summary[money_cols] = rider_earnings_summary[money_cols].round(2)

    return rider_season_earnings, rider_earnings_summary
