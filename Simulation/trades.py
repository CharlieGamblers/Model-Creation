# trades.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from data_io import norm          # for case-insensitive rider names
from seasons import simulate_many_seasons, summarize_team_outcomes


# ============================================================
# 1. ROSTER MANIPULATION (TRADE ENGINE)
# ============================================================

def apply_trade_to_roster_map(
    roster_map: Dict[str, List[str]],
    rider_a: str,
    rider_b: str,
) -> Dict[str, List[str]]:
    """
    Return a NEW roster_map where rider_a and rider_b have been swapped
    between whatever teams they are currently on.

    Rider/team names are treated case-insensitively via norm().

    Parameters
    ----------
    roster_map : dict
        {team_code: [rider1, rider2, ...]} using normalized names.
    rider_a : str
        Name of first rider (any casing accepted).
    rider_b : str
        Name of second rider (any casing accepted).

    Returns
    -------
    new_roster_map : dict
        Deep copy of roster_map with the two riders swapped.

    Raises
    ------
    ValueError
        If a rider is not found or both riders are on same team.
    """
    ra = norm(rider_a)
    rb = norm(rider_b)

    team_a = None
    team_b = None

    # Find which teams they currently belong to
    for team, riders in roster_map.items():
        if ra in riders:
            team_a = team
        if rb in riders:
            team_b = team

    if team_a is None:
        raise ValueError(f"Could not find rider '{rider_a}' in any team roster.")
    if team_b is None:
        raise ValueError(f"Could not find rider '{rider_b}' in any team roster.")

    if team_a == team_b:
        raise ValueError(
            f"Both riders are on the same team ({team_a}). "
            f"Trade is only defined between different teams."
        )

    # Deep copy of roster_map so we don't mutate the original
    new_roster_map: Dict[str, List[str]] = {
        t: list(riders) for t, riders in roster_map.items()
    }

    idx_a = new_roster_map[team_a].index(ra)
    idx_b = new_roster_map[team_b].index(rb)

    # Swap riders
    new_roster_map[team_a][idx_a] = rb
    new_roster_map[team_b][idx_b] = ra

    return new_roster_map


# ============================================================
# 2. TEAM OUTCOME SUMMARY
# ============================================================



# ============================================================
# 3. TRADE COMPARISON
# ============================================================

def run_trade_comparison(
    rider_a: str,
    rider_b: str,
    n_sims: int,
    schedule_df: pd.DataFrame,
    roster_map: Dict[str, List[str]],
    bulls_df: pd.DataFrame,
    seed: int = 42,
    lineup_size: int = 5,
    use_smooth_slots: bool = True,
) -> pd.DataFrame:
    """
    Run side-by-side multi-season sims:

    - Baseline: original roster_map
    - Trade: rider_a and rider_b swapped between their current teams

    Metrics compared:
      - avg_wins
      - avg_win_pct
      - mean_total_prize

    Parameters
    ----------
    rider_a, rider_b : str
        Names of the riders to trade.
    n_sims : int
        Number of seasons to simulate for each scenario.
    schedule_df : pd.DataFrame
        Schedule with columns ['game_id', 'home_team', 'away_team'].
    roster_map : dict
        {team_code: [rider1, rider2, ...]} with normalized names.
    bulls_df : pd.DataFrame
        Bulls table with at least 'bull_id'.
    seed : int
        Base RNG seed for reproducibility.
    lineup_size : int
        Number of rides per team per game.
    use_smooth_slots : bool
        Pass-through flag to seasons.simulate_many_seasons.

    Returns
    -------
    comparison_df : pd.DataFrame
        Columns:
          - team
          - avg_wins_base, avg_wins_trade, delta_wins
          - avg_win_pct_base, avg_win_pct_trade, delta_win_pct
          - mean_total_prize_base, mean_total_prize_trade, delta_total_prize
    """
    # --- Baseline ---
    many_standings_base, many_mvp_base, many_playoffs_base, many_prizes_base = (
        simulate_many_seasons(
            n_sims,
            schedule_df,
            roster_map,
            bulls_df,
            seed=seed,
            lineup_size=lineup_size,
            use_smooth_slots=use_smooth_slots,
        )
    )
    base_summary = summarize_team_outcomes(many_standings_base, many_prizes_base)
    base_summary = base_summary.rename(
        columns={
            "avg_wins": "avg_wins_base",
            "avg_win_pct": "avg_win_pct_base",
            "mean_total_prize": "mean_total_prize_base",
        }
    )

    # --- Trade scenario ---
    traded_roster_map = apply_trade_to_roster_map(roster_map, rider_a, rider_b)

    many_standings_trade, many_mvp_trade, many_playoffs_trade, many_prizes_trade = (
        simulate_many_seasons(
            n_sims,
            schedule_df,
            traded_roster_map,
            bulls_df,
            seed=seed,
            lineup_size=lineup_size,
            use_smooth_slots=use_smooth_slots,
        )
    )
    trade_summary = summarize_team_outcomes(many_standings_trade, many_prizes_trade)
    trade_summary = trade_summary.rename(
        columns={
            "avg_wins": "avg_wins_trade",
            "avg_win_pct": "avg_win_pct_trade",
            "mean_total_prize": "mean_total_prize_trade",
        }
    )

    # --- Combine + deltas ---
    comparison = base_summary.merge(trade_summary, on="team", how="outer")

    comparison["delta_wins"] = (
        comparison["avg_wins_trade"] - comparison["avg_wins_base"]
    )
    comparison["delta_win_pct"] = (
        comparison["avg_win_pct_trade"] - comparison["avg_win_pct_base"]
    )
    comparison["delta_total_prize"] = (
        comparison["mean_total_prize_trade"] - comparison["mean_total_prize_base"]
    )

    # Nice rounding
    comparison["avg_wins_base"] = comparison["avg_wins_base"].round(3)
    comparison["avg_wins_trade"] = comparison["avg_wins_trade"].round(3)
    comparison["delta_wins"] = comparison["delta_wins"].round(3)

    comparison["avg_win_pct_base"] = comparison["avg_win_pct_base"].round(4)
    comparison["avg_win_pct_trade"] = comparison["avg_win_pct_trade"].round(4)
    comparison["delta_win_pct"] = comparison["delta_win_pct"].round(4)

    money_cols = [
        "mean_total_prize_base",
        "mean_total_prize_trade",
        "delta_total_prize",
    ]
    comparison[money_cols] = comparison[money_cols].round(2)

    return comparison
