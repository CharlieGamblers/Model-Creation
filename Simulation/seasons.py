# seasons.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

import prob_models  # for global RNG control
from prob_models import get_ride_probability
from data_io import select_bulls, RNG_SEED, REGULAR_PRIZE, PLAYOFF_PRIZE, EXTRA_WIN_POOL
from lineups import select_lineup_smooth_slots, select_lineup
from playoffs import simulate_playoffs


# ============================================================
# 1. SINGLE GAME SIM
# ============================================================

def simulate_game(
    game_row: pd.Series,
    roster_map: Dict[str, List[str]],
    bulls_df: pd.DataFrame,
    lineup_size: int = 5,
    use_smooth_slots: bool = True,
) -> Tuple[dict, List[dict]]:
    """
    Simulate one regular-season style game (2 teams, 5 rides each).

    Args
    ----
    game_row : pd.Series
        Must have columns: 'game_id', 'home_team', 'away_team'
        Teams are already normalized (lowercase) from data_io.
    roster_map : dict
        {team_code: [rider1, rider2, ...]} with normalized rider names.
    bulls_df : pd.DataFrame
        Must contain 'bull_id' column.
    lineup_size : int
        Number of riders per team (default 5).
    use_smooth_slots : bool
        If True, use the smooth slot-usage model for lineups.
        If False, use deterministic chalk.

    Returns
    -------
    game_summary : dict
        One row of game-level results.
    ride_rows : list of dict
        One dict per ride (per rider-bull).
    """
    home = game_row["home_team"]
    away = game_row["away_team"]
    game_id = game_row["game_id"]

    # --- get rosters ---
    if home not in roster_map:
        raise ValueError(f"Team {home} not found in roster_map")
    if away not in roster_map:
        raise ValueError(f"Team {away} not found in roster_map")

    home_roster = roster_map[home]
    away_roster = roster_map[away]

    # --- select lineups ---
    if use_smooth_slots:
        home_lineup = select_lineup_smooth_slots(home_roster, home, lineup_size)
        away_lineup = select_lineup_smooth_slots(away_roster, away, lineup_size)
    else:
        # plain chalk lineups
        home_lineup = select_lineup(home_roster, home, lineup_size)
        away_lineup = select_lineup(away_roster, away, lineup_size)

    # --- bulls: one per rider ---
    home_bulls = select_bulls(bulls_df, len(home_lineup))
    away_bulls = select_bulls(bulls_df, len(away_lineup))

    home_scores = []
    away_scores = []
    ride_rows: List[dict] = []

    # --- Home rides ---
    for rider, bull in zip(home_lineup, home_bulls):
        p = get_ride_probability(rider=rider, team=home, bull_id=bull)
        ride_made = int(prob_models.rng.random() < p)

        home_scores.append(ride_made)

        ride_rows.append({
            "game_id": game_id,
            "team": home,
            "opponent": away,
            "rider": rider,
            "bull_id": bull,
            "is_home": True,
            "ride_made": ride_made,
            "ride_prob": p,
        })

    # --- Away rides ---
    for rider, bull in zip(away_lineup, away_bulls):
        p = get_ride_probability(rider=rider, team=away, bull_id=bull)
        ride_made = int(prob_models.rng.random() < p)

        away_scores.append(ride_made)

        ride_rows.append({
            "game_id": game_id,
            "team": away,
            "opponent": home,
            "rider": rider,
            "bull_id": bull,
            "is_home": False,
            "ride_made": ride_made,
            "ride_prob": p,
        })

    home_total = int(sum(home_scores))
    away_total = int(sum(away_scores))

    # --- decide winner (coin flip on ties) ---
    if home_total > away_total:
        winner = home
        loser = away
        result = "home_win"
    elif away_total > home_total:
        winner = away
        loser = home
        result = "away_win"
    else:
        if prob_models.rng.random() < 0.5:
            winner = home
            loser = away
            result = "tie_home_coinflip"
        else:
            winner = away
            loser = home
            result = "tie_away_coinflip"

    game_summary = {
        "game_id": game_id,
        "home_team": home,
        "away_team": away,
        "home_lineup": ", ".join(home_lineup),
        "away_lineup": ", ".join(away_lineup),
        "home_score": home_total,
        "away_score": away_total,
        "result": result,
        "winner": winner,
        "loser": loser,
    }

    return game_summary, ride_rows


# ============================================================
# 2. ONE FULL SEASON
# ============================================================

def simulate_season(
    schedule: pd.DataFrame,
    roster_map: Dict[str, List[str]],
    bulls_df: pd.DataFrame,
    lineup_size: int = 5,
    use_smooth_slots: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simulate a full regular season.

    Args
    ----
    schedule : pd.DataFrame
        Columns: game_id, home_team, away_team (normalized).
    roster_map : dict
        {team_code: [rider1, rider2, ...]}.
    bulls_df : pd.DataFrame
        Contains 'bull_id'.
    lineup_size : int
        Riders per game per team (5).
    use_smooth_slots : bool
        Use slot-usage model if True.

    Returns
    -------
    game_results_df : pd.DataFrame
        One row per game.
    standings_df : pd.DataFrame
        One row per team: wins/losses/win%, + team_ride_pct.
    rides_df : pd.DataFrame
        One row per ride.
    """
    game_results = []
    all_rides = []

    for _, row in schedule.iterrows():
        game_summary, ride_rows = simulate_game(
            row,
            roster_map,
            bulls_df,
            lineup_size=lineup_size,
            use_smooth_slots=use_smooth_slots,
        )
        game_results.append(game_summary)
        all_rides.extend(ride_rows)

    game_results_df = pd.DataFrame(game_results)
    rides_df = pd.DataFrame(all_rides)

    # --- Build standings ---
    teams = sorted(set(game_results_df["home_team"]) | set(game_results_df["away_team"]))

    records = []
    for team in teams:
        wins = (game_results_df["winner"] == team).sum()
        losses = (game_results_df["loser"] == team).sum()

        is_home = game_results_df["home_team"] == team
        is_away = game_results_df["away_team"] == team
        played = is_home | is_away

        # we don't store explicit ties because we coin-flip them
        ties = ((game_results_df["winner"].isna()) & played).sum()

        games = wins + losses + ties
        win_pct = wins / games if games > 0 else np.nan

        records.append({
            "team": team,
            "games": games,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_pct": win_pct,
        })

    standings_df = (
        pd.DataFrame(records)
        .sort_values(by=["win_pct", "wins"], ascending=[False, False])
        .reset_index(drop=True)
    )

    # --- Team riding % ---
    if not rides_df.empty:
        team_ride_stats = (
            rides_df
            .groupby("team", as_index=False)
            .agg(
                team_total_rides=("ride_made", "size"),
                team_rides_made=("ride_made", "sum"),
            )
        )
        team_ride_stats["team_ride_pct"] = (
            team_ride_stats["team_rides_made"] / team_ride_stats["team_total_rides"]
        )
        standings_df = standings_df.merge(team_ride_stats, on="team", how="left")
    else:
        standings_df["team_total_rides"] = 0
        standings_df["team_rides_made"] = 0
        standings_df["team_ride_pct"] = np.nan

    return game_results_df, standings_df, rides_df


# ============================================================
# 3. MVP STANDINGS (PER SEASON)
# ============================================================

def compute_mvp_standings(rides_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simple MVP-style leaderboard from ride-level data.

    For each rider + team:
    - total_rides
    - rides_made
    - ride_pct

    Sorted by: rides_made desc, ride_pct desc, total_rides desc.
    """
    if rides_df.empty:
        return pd.DataFrame(columns=["rider", "team", "total_rides", "rides_made", "ride_pct"])

    required_cols = {"rider", "team", "ride_made"}
    missing = required_cols - set(rides_df.columns)
    if missing:
        raise ValueError(f"rides_df missing columns for MVP standings: {missing}")

    mvp = (
        rides_df
        .groupby(["rider", "team"], as_index=False)
        .agg(
            total_rides=("ride_made", "size"),
            rides_made=("ride_made", "sum"),
        )
    )
    mvp["ride_pct"] = mvp["rides_made"] / mvp["total_rides"]

    mvp = (
        mvp.sort_values(
            by=["rides_made", "ride_pct", "total_rides"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )

    return mvp


# ============================================================
# 4. MANY-SEASON SIM WRAPPER
# ============================================================

def simulate_many_seasons(
    n: int,
    schedule_df: pd.DataFrame,
    roster_map: Dict[str, List[str]],
    bulls_df: pd.DataFrame,
    lineup_size: int = 5,
    seed: int = RNG_SEED,
    use_smooth_slots: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the full season simulation n times (regular season + playoffs + prizes).
    
    Returns
    -------
    many_standings_df : pd.DataFrame
        Long format; one row per team per season.
        Columns include: team, wins, win_pct, team_ride_pct, season_id, rank.
    many_mvp_df : pd.DataFrame
        Long format; MVP table per season with season_id, rank.
    many_playoffs_df : pd.DataFrame
        Long format; playoff placements per season with season_id.
    many_prizes_df : pd.DataFrame
        Long format; prize money per team per season.
    """
    all_standings = []
    all_mvp = []
    all_playoffs = []
    all_prizes = []

    for season_id in range(1, n + 1):
        # Fresh RNG each season (affects select_bulls + ride outcomes)
        prob_models.rng = np.random.default_rng(seed + season_id)

        # --- Regular season ---
        game_results_df, standings_df, rides_df = simulate_season(
            schedule_df,
            roster_map,
            bulls_df,
            lineup_size=lineup_size,
            use_smooth_slots=use_smooth_slots,
        )

        # --- tag & rank standings ---
        standings_df = standings_df.copy()
        standings_df["season_id"] = season_id
        standings_df["rank"] = standings_df["win_pct"].rank(
            method="first", ascending=False
        )
        all_standings.append(standings_df)

        # --- per-season MVP standings ---
        mvp_df = compute_mvp_standings(rides_df).copy()
        mvp_df["season_id"] = season_id
        mvp_df["rank"] = np.arange(1, len(mvp_df) + 1)
        all_mvp.append(mvp_df)

        # --- Playoffs for this season ---
        playoff_games_df, playoff_rides_df, placements_df = simulate_playoffs(
            standings_df, roster_map, bulls_df,
            lineup_size=lineup_size,
            use_smooth_slots=use_smooth_slots,
        )
        placements_df = placements_df.copy()
        placements_df["season_id"] = season_id
        all_playoffs.append(placements_df)

        # =====================================================
        # PRIZE MONEY FOR THIS SEASON
        # =====================================================

        # Regular-season placement prize
        season_prizes = standings_df[["season_id", "team", "wins", "rank"]].copy()
        season_prizes["regular_prize"] = season_prizes["rank"].map(REGULAR_PRIZE).fillna(0.0)

        # Wins after 14
        season_prizes["wins_after_14"] = (season_prizes["wins"] - 14).clip(lower=0)

        total_extra_wins = season_prizes["wins_after_14"].sum()
        if total_extra_wins > 0:
            value_per_extra_win = EXTRA_WIN_POOL / total_extra_wins
        else:
            value_per_extra_win = 0.0

        season_prizes["extra_win_bonus"] = season_prizes["wins_after_14"] * value_per_extra_win

        # Playoff prize: merge by team
        season_prizes = season_prizes.merge(
            placements_df[["team", "playoff_finish"]],
            on="team",
            how="left",
        )
        season_prizes["playoff_prize"] = season_prizes["playoff_finish"].map(
            PLAYOFF_PRIZE
        ).fillna(0.0)

        # Total prize this season
        season_prizes["total_prize"] = (
            season_prizes["regular_prize"]
            + season_prizes["extra_win_bonus"]
            + season_prizes["playoff_prize"]
        )

        all_prizes.append(season_prizes)

    many_standings_df = pd.concat(all_standings, ignore_index=True)
    many_mvp_df = pd.concat(all_mvp, ignore_index=True)
    many_playoffs_df = pd.concat(all_playoffs, ignore_index=True)
    many_prizes_df = pd.concat(all_prizes, ignore_index=True)

    return many_standings_df, many_mvp_df, many_playoffs_df, many_prizes_df


# ============================================================
# 5. OPTIONAL: LEAGUE-WIDE RIDE% PER SEASON
# ============================================================

def summarize_team_outcomes(
    many_standings_df: pd.DataFrame,
    many_prizes_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Helper to build a compact summary per team from a multi-season sim.
    """
    team_summary = (
        many_standings_df
        .groupby("team", as_index=False)
        .agg(
            avg_wins=("wins", "mean"),
            avg_win_pct=("win_pct", "mean"),
        )
    )

    prize_summary = (
        many_prizes_df
        .groupby("team", as_index=False)
        .agg(
            mean_total_prize=("total_prize", "mean"),
        )
    )

    out = team_summary.merge(prize_summary, on="team", how="left")
    return out


def compute_league_ride_pct_per_season(many_mvp_df: pd.DataFrame) -> pd.Series:
    """
    Given many_mvp_df (output of simulate_many_seasons),
    compute league-wide ride% per season:

        sum(rides_made) / sum(total_rides) by season_id
    """
    if many_mvp_df.empty:
        return pd.Series(dtype=float)

    rides_by_season = many_mvp_df.groupby("season_id")["rides_made"].sum()
    outs_by_season = many_mvp_df.groupby("season_id")["total_rides"].sum()
    return rides_by_season / outs_by_season
