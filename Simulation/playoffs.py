# playoffs.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

import prob_models
from prob_models import get_ride_probability
from data_io import select_bulls
from lineups import select_lineup_smooth_slots, select_lineup
# Import simulate_game lazily to avoid circular import with seasons


# ============================================================
# 1. SEEDING HELPERS
# ============================================================

def seed_teams_from_standings(
    standings_df: pd.DataFrame
) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Take a regular-season standings df (one season) and assign seeds 1–10.
    Lower seed number = better team (1 is best).

    standings_df should have columns at least: 'team', 'win_pct', 'wins'.

    Returns
    -------
    seed_to_team : dict
        {1: "team_code_1", 2: "team_code_2", ...}
    team_to_seed : dict
        {"team_code_1": 1, "team_code_2": 2, ...}
    """
    df = (
        standings_df
        .sort_values(["win_pct", "wins"], ascending=[False, False])
        .reset_index(drop=True)
        .copy()
    )
    df["seed"] = np.arange(1, len(df) + 1)

    seed_to_team = dict(zip(df["seed"], df["team"]))
    team_to_seed = dict(zip(df["team"], df["seed"]))

    return seed_to_team, team_to_seed


def order_three_teams_by_score(team_scores: Dict[str, int]) -> List[str]:
    """
    Given {team: total_score} for 3 teams, return them ordered from
    highest to lowest, breaking ties by random coin flip.

    We break ties by:
      - random shuffle
      - then sort by total_score descending
    """
    teams = list(team_scores.keys())

    # Shuffle first for randomness, then sort by score desc
    prob_models.rng.shuffle(teams)
    teams.sort(key=lambda t: team_scores[t], reverse=True)
    return teams


# ============================================================
# 2. GAME WRAPPERS FOR PLAYOFFS
# ============================================================

def simulate_two_team_playoff_game(
    team1: str,
    team2: str,
    game_id: str,
    roster_map: Dict[str, List[str]],
    bulls_df: pd.DataFrame,
    lineup_size: int = 5,
    use_smooth_slots: bool = True,
) -> Tuple[str, str, dict, List[dict]]:
    """
    Wrapper around seasons.simulate_game for a head-to-head playoff game.

    Returns
    -------
    winner : str
    loser : str
    game_summary : dict
    ride_rows : list[dict]
    """
    # Lazy import to avoid circular dependency
    from seasons import simulate_game
    
    row = pd.Series({"game_id": game_id, "home_team": team1, "away_team": team2})
    game_summary, ride_rows = simulate_game(
        row,
        roster_map,
        bulls_df,
        lineup_size=lineup_size,
        use_smooth_slots=use_smooth_slots,
    )

    winner = game_summary["winner"]
    loser = game_summary["loser"]

    return winner, loser, game_summary, ride_rows


def simulate_three_team_game(
    teams: List[str],
    game_id: str,
    roster_map: Dict[str, List[str]],
    bulls_df: pd.DataFrame,
    lineup_size: int = 5,
    use_smooth_slots: bool = True,
) -> Tuple[str, List[str], List[dict], List[dict]]:
    """
    Simulate a 3-team playoff game (each team gets lineup_size rides, independent).

    Args
    ----
    teams : list[str]
        Exactly 3 team codes.
    game_id : str
        Identifier (e.g., "PO-1", "PO-8").
    roster_map : dict
        {team_code: [rider1, rider2, ...]} normalized.
    bulls_df : pd.DataFrame
        Must contain 'bull_id'.
    lineup_size : int
        Rides per team.
    use_smooth_slots : bool
        If True, use smooth slot model; else chalk.

    Returns
    -------
    winner : str
        Team with highest score (ties broken by coin flip).
    ordered_teams : list[str]
        Teams ordered [1st, 2nd, 3rd].
    game_summaries : list[dict]
        One row per team with total_score + finish_order.
    ride_rows : list[dict]
        One row per ride.
    """
    assert len(teams) == 3, "3-team game requires exactly 3 teams"

    team_scores: Dict[str, int] = {}
    game_summaries: List[dict] = []
    ride_rows: List[dict] = []

    for team in teams:
        roster = roster_map[team]

        # Use same lineup logic as regular season (opt-in smooth slots)
        if use_smooth_slots:
            lineup = select_lineup_smooth_slots(roster, team, lineup_size)
        else:
            lineup = select_lineup(roster, team, lineup_size)

        bulls = select_bulls(bulls_df, len(lineup))

        score = 0
        for rider, bull in zip(lineup, bulls):
            p = get_ride_probability(rider=rider, team=team, bull_id=bull)
            ride_made = int(prob_models.rng.random() < p)
            score += ride_made

            ride_rows.append({
                "game_id": game_id,
                "team": team,
                "rider": rider,
                "bull_id": bull,
                "ride_made": ride_made,
                "ride_prob": p,
            })

        team_scores[team] = score

    ordered = order_three_teams_by_score(team_scores)
    winner = ordered[0]

    for team in ordered:
        game_summaries.append({
            "game_id": game_id,
            "team": team,
            "total_score": team_scores[team],
            "finish_order": ordered.index(team) + 1,  # 1,2,3
            "is_winner": (team == winner),
        })

    return winner, ordered, game_summaries, ride_rows


# ============================================================
# 3. FULL 12-GAME PLAYOFF BRACKET
# ============================================================

def simulate_playoffs(
    season_standings_df: pd.DataFrame,
    roster_map: Dict[str, List[str]],
    bulls_df: pd.DataFrame,
    lineup_size: int = 5,
    use_smooth_slots: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simulate the 12-game PBR Teams playoff bracket.

    Bracket (seeds 1–10):

    Day 1
    -----
    G1: 8 vs 9 vs 10       -> winner to G4, losers = 9th & 10th
    G2: 4 vs 7             -> winner to Day 2, loser to G4
    G3: 5 vs 6             -> winner to Day 2, loser to G4
    G4: L(G2) vs L(G3) vs W(G1) -> winner to Day 2, losers = 7th & 8th

    Day 2
    -----
    Remaining: seeds 1,2,3 + winners of G2,G3,G4 (6 teams)
    G5: 1 vs lowest seed remaining       -> winner to Day 3, loser to G8
    G6: 2 vs 2nd-lowest remaining        -> winner to Day 3, loser to G8
    G7: 3 vs 3rd-lowest remaining        -> winner to Day 3, loser to G8
    G8: L(G5) vs L(G6) vs L(G7)          -> winner to Day 3, losers = 5th & 6th

    Day 3
    -----
    Remaining 4 teams:
      - order by seed (best = highest rank; numerically smallest seed)
    G9: highest vs lowest
    G10: 2nd-highest vs 2nd-lowest
    G11: L(G9) vs L(G10)  -> 3rd & 4th
    G12: W(G9) vs W(G10)  -> 1st & 2nd

    Returns
    -------
    playoff_games_df : pd.DataFrame
        One row per team per playoff game (PO-1..PO-12).
    playoff_rides_df : pd.DataFrame
        One row per ride in the playoffs.
    placements_df : pd.DataFrame
        Columns: team, playoff_finish (1–10).
    """
    seed_to_team, team_to_seed = seed_teams_from_standings(season_standings_df)

    def T(seed: int) -> str:
        """Convenience: team name from seed."""
        return seed_to_team[seed]

    placements: Dict[str, int] = {team: None for team in team_to_seed.keys()}
    all_game_rows: List[dict] = []
    all_ride_rows: List[dict] = []

    # --------------------
    # DAY 1
    # --------------------
    # Game 1: seeds 8 vs 9 vs 10
    g1_id = "PO-1"
    g1_teams = [T(8), T(9), T(10)]
    g1_winner, g1_ordered, g1_summaries, g1_rides = simulate_three_team_game(
        g1_teams, g1_id, roster_map, bulls_df,
        lineup_size=lineup_size,
        use_smooth_slots=use_smooth_slots,
    )
    all_game_rows.extend(g1_summaries)
    all_ride_rows.extend(g1_rides)

    # Losers of Game 1 are 9th and 10th
    g1_losers = g1_ordered[1:]  # 2nd and 3rd
    placements[g1_losers[0]] = 9
    placements[g1_losers[1]] = 10

    # Game 2: 4 vs 7
    g2_id = "PO-2"
    g2_winner, g2_loser, g2_summary, g2_rides = simulate_two_team_playoff_game(
        T(4), T(7), g2_id, roster_map, bulls_df,
        lineup_size=lineup_size,
        use_smooth_slots=use_smooth_slots,
    )
    all_game_rows.append({
        "game_id": g2_id,
        "team": T(4),
        "opponent": T(7),
        "total_score": g2_summary["home_score"],
        "is_winner": (g2_summary["winner"] == T(4)),
    })
    all_game_rows.append({
        "game_id": g2_id,
        "team": T(7),
        "opponent": T(4),
        "total_score": g2_summary["away_score"],
        "is_winner": (g2_summary["winner"] == T(7)),
    })
    all_ride_rows.extend(g2_rides)

    # Game 3: 5 vs 6
    g3_id = "PO-3"
    g3_winner, g3_loser, g3_summary, g3_rides = simulate_two_team_playoff_game(
        T(5), T(6), g3_id, roster_map, bulls_df,
        lineup_size=lineup_size,
        use_smooth_slots=use_smooth_slots,
    )
    all_game_rows.append({
        "game_id": g3_id,
        "team": T(5),
        "opponent": T(6),
        "total_score": g3_summary["home_score"],
        "is_winner": (g3_summary["winner"] == T(5)),
    })
    all_game_rows.append({
        "game_id": g3_id,
        "team": T(6),
        "opponent": T(5),
        "total_score": g3_summary["away_score"],
        "is_winner": (g3_summary["winner"] == T(6)),
    })
    all_ride_rows.extend(g3_rides)

    # Game 4: Loser of 2, Loser of 3, Winner of 1
    g4_id = "PO-4"
    g4_teams = [g2_loser, g3_loser, g1_winner]
    g4_winner, g4_ordered, g4_summaries, g4_rides = simulate_three_team_game(
        g4_teams, g4_id, roster_map, bulls_df,
        lineup_size=lineup_size,
        use_smooth_slots=use_smooth_slots,
    )
    all_game_rows.extend(g4_summaries)
    all_ride_rows.extend(g4_rides)

    # Losers of Game 4 are 7th and 8th
    g4_losers = g4_ordered[1:]
    placements[g4_losers[0]] = 7
    placements[g4_losers[1]] = 8

    # Teams advancing to Day 2:
    remaining = {T(1), T(2), T(3), g2_winner, g3_winner, g4_winner}

    # --------------------
    # DAY 2
    # --------------------
    # Identify other 3 teams besides seeds 1–3
    others = [t for t in remaining if team_to_seed[t] not in (1, 2, 3)]
    # sort by seed descending: WORST seed first
    others.sort(key=lambda t: team_to_seed[t], reverse=True)
    lowest, second_lowest, third_lowest = others

    # Game 5: 1 vs Lowest Seed Remaining
    g5_id = "PO-5"
    g5_winner, g5_loser, g5_summary, g5_rides = simulate_two_team_playoff_game(
        T(1), lowest, g5_id, roster_map, bulls_df,
        lineup_size=lineup_size,
        use_smooth_slots=use_smooth_slots,
    )
    all_game_rows.append({
        "game_id": g5_id,
        "team": T(1),
        "opponent": lowest,
        "total_score": g5_summary["home_score"],
        "is_winner": (g5_summary["winner"] == T(1)),
    })
    all_game_rows.append({
        "game_id": g5_id,
        "team": lowest,
        "opponent": T(1),
        "total_score": g5_summary["away_score"],
        "is_winner": (g5_summary["winner"] == lowest),
    })
    all_ride_rows.extend(g5_rides)

    # Game 6: 2 vs 2nd Lowest Remaining
    g6_id = "PO-6"
    g6_winner, g6_loser, g6_summary, g6_rides = simulate_two_team_playoff_game(
        T(2), second_lowest, g6_id, roster_map, bulls_df,
        lineup_size=lineup_size,
        use_smooth_slots=use_smooth_slots,
    )
    all_game_rows.append({
        "game_id": g6_id,
        "team": T(2),
        "opponent": second_lowest,
        "total_score": g6_summary["home_score"],
        "is_winner": (g6_summary["winner"] == T(2)),
    })
    all_game_rows.append({
        "game_id": g6_id,
        "team": second_lowest,
        "opponent": T(2),
        "total_score": g6_summary["away_score"],
        "is_winner": (g6_summary["winner"] == second_lowest),
    })
    all_ride_rows.extend(g6_rides)

    # Game 7: 3 vs 3rd Lowest Remaining
    g7_id = "PO-7"
    g7_winner, g7_loser, g7_summary, g7_rides = simulate_two_team_playoff_game(
        T(3), third_lowest, g7_id, roster_map, bulls_df,
        lineup_size=lineup_size,
        use_smooth_slots=use_smooth_slots,
    )
    all_game_rows.append({
        "game_id": g7_id,
        "team": T(3),
        "opponent": third_lowest,
        "total_score": g7_summary["home_score"],
        "is_winner": (g7_summary["winner"] == T(3)),
    })
    all_game_rows.append({
        "game_id": g7_id,
        "team": third_lowest,
        "opponent": T(3),
        "total_score": g7_summary["away_score"],
        "is_winner": (g7_summary["winner"] == third_lowest),
    })
    all_ride_rows.extend(g7_rides)

    winners_day2 = {g5_winner, g6_winner, g7_winner}
    losers_day2 = [g5_loser, g6_loser, g7_loser]

    # Game 8: 3-team, losers of 5/6/7
    g8_id = "PO-8"
    g8_winner, g8_ordered, g8_summaries, g8_rides = simulate_three_team_game(
        losers_day2, g8_id, roster_map, bulls_df,
        lineup_size=lineup_size,
        use_smooth_slots=use_smooth_slots,
    )
    all_game_rows.extend(g8_summaries)
    all_ride_rows.extend(g8_rides)

    # Losers of Game 8 are 5th and 6th
    g8_losers = g8_ordered[1:]
    placements[g8_losers[0]] = 5
    placements[g8_losers[1]] = 6

    # Teams advancing to Day 3:
    remaining_day3 = list(winners_day2) + [g8_winner]

    # --------------------
    # DAY 3
    # --------------------
    # Determine seeds among remaining 4
    remaining_day3.sort(key=lambda t: team_to_seed[t])  # ascending: best seed first
    highest = remaining_day3[0]
    second_highest = remaining_day3[1]
    second_lowest = remaining_day3[2]
    lowest = remaining_day3[3]

    # Game 9: Highest vs Lowest
    g9_id = "PO-9"
    g9_winner, g9_loser, g9_summary, g9_rides = simulate_two_team_playoff_game(
        highest, lowest, g9_id, roster_map, bulls_df,
        lineup_size=lineup_size,
        use_smooth_slots=use_smooth_slots,
    )
    all_game_rows.append({
        "game_id": g9_id,
        "team": highest,
        "opponent": lowest,
        "total_score": g9_summary["home_score"],
        "is_winner": (g9_summary["winner"] == highest),
    })
    all_game_rows.append({
        "game_id": g9_id,
        "team": lowest,
        "opponent": highest,
        "total_score": g9_summary["away_score"],
        "is_winner": (g9_summary["winner"] == lowest),
    })
    all_ride_rows.extend(g9_rides)

    # Game 10: 2nd Highest vs 2nd Lowest
    g10_id = "PO-10"
    g10_winner, g10_loser, g10_summary, g10_rides = simulate_two_team_playoff_game(
        second_highest, second_lowest, g10_id, roster_map, bulls_df,
        lineup_size=lineup_size,
        use_smooth_slots=use_smooth_slots,
    )
    all_game_rows.append({
        "game_id": g10_id,
        "team": second_highest,
        "opponent": second_lowest,
        "total_score": g10_summary["home_score"],
        "is_winner": (g10_summary["winner"] == second_highest),
    })
    all_game_rows.append({
        "game_id": g10_id,
        "team": second_lowest,
        "opponent": second_highest,
        "total_score": g10_summary["away_score"],
        "is_winner": (g10_summary["winner"] == second_lowest),
    })
    all_ride_rows.extend(g10_rides)

    # Game 11: Loser of 9 vs Loser of 10 -> 3rd and 4th
    g11_id = "PO-11"
    g11_winner, g11_loser, g11_summary, g11_rides = simulate_two_team_playoff_game(
        g9_loser, g10_loser, g11_id, roster_map, bulls_df,
        lineup_size=lineup_size,
        use_smooth_slots=use_smooth_slots,
    )
    all_game_rows.append({
        "game_id": g11_id,
        "team": g9_loser,
        "opponent": g10_loser,
        "total_score": g11_summary["home_score"],
        "is_winner": (g11_summary["winner"] == g9_loser),
    })
    all_game_rows.append({
        "game_id": g11_id,
        "team": g10_loser,
        "opponent": g9_loser,
        "total_score": g11_summary["away_score"],
        "is_winner": (g11_summary["winner"] == g10_loser),
    })
    all_ride_rows.extend(g11_rides)

    placements[g11_winner] = 3
    placements[g11_loser] = 4

    # Game 12: Winner of 9 vs Winner of 10 -> 1st and 2nd
    g12_id = "PO-12"
    g12_winner, g12_loser, g12_summary, g12_rides = simulate_two_team_playoff_game(
        g9_winner, g10_winner, g12_id, roster_map, bulls_df,
        lineup_size=lineup_size,
        use_smooth_slots=use_smooth_slots,
    )
    all_game_rows.append({
        "game_id": g12_id,
        "team": g9_winner,
        "opponent": g10_winner,
        "total_score": g12_summary["home_score"],
        "is_winner": (g12_summary["winner"] == g9_winner),
    })
    all_game_rows.append({
        "game_id": g12_id,
        "team": g10_winner,
        "opponent": g9_winner,
        "total_score": g12_summary["away_score"],
        "is_winner": (g12_summary["winner"] == g10_winner),
    })
    all_ride_rows.extend(g12_rides)

    placements[g12_winner] = 1
    placements[g12_loser] = 2

    # --------------------
    # Build output DFs
    # --------------------
    playoff_games_df = pd.DataFrame(all_game_rows)
    playoff_rides_df = pd.DataFrame(all_ride_rows)

    placements_df = (
        pd.DataFrame(
            [{"team": team, "playoff_finish": finish}
             for team, finish in placements.items()]
        )
        .sort_values("playoff_finish")
        .reset_index(drop=True)
    )

    return playoff_games_df, playoff_rides_df, placements_df
