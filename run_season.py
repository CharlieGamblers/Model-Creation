import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# ============================================================
# 1. CONFIG
# ============================================================
SCHEDULE_PATH = "schedule.csv"
ROSTERS_PATH = "rosters.csv"
BULLS_PATH = "bulls.csv"
RIDER_RATINGS_PATH = "rider_ratings.csv"   # NEW
MATCHUP_PROBS_PATH = "rider_bull_probs.csv"  # or whatever you call it
ELITE_START_PROB = 1 # just a starting guess, we’ll tune this

MATCHUP_PROB_MAP: Dict = {}  # (rider, bull_id) -> prob
LINEUP_SIZE = 5
BASE_RIDE_PROB = 0.4

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ============================================================
# PRIZE CONFIG
# ============================================================
REGULAR_PRIZE = {
    1: 350_000,
    2: 275_000,
    3: 150_000,
    4: 100_000,
    5: 75_000,
    6: 50_000,
    # 7–10: 0
}

PLAYOFF_PRIZE = {
    1: 250_000,
    2: 150_000,
    3: 75_000,
    4: 25_000,
    # 5–10: 0
}

EXTRA_WIN_POOL = 1_000_000  # league-wide pool for wins after 14

# ============================================================
# SLOT USAGE MODEL (SMOOTH)
# ============================================================

USE_SMOOTH_SLOTS = True  # toggle if you ever want to switch off

SLOT_ALPHA = 1.10
SLOT_BETA  = 0.73
SLOT_SIGMA = 1.15


def build_slot_prob_matrix_smooth(
    max_rank: int = 12,
    max_slot: int = 8,
    alpha: float = SLOT_ALPHA,
    beta: float = SLOT_BETA,
    sigma: float = SLOT_SIGMA,
) -> np.ndarray:
    """
    Python port of your R slot-usage model.

    Returns:
        prob_matrix[r-1, s-1] = P(slot = s | rank = r)

    Ranks and slots are 1-based conceptually, but the matrix is 0-based.
    """
    ranks = np.arange(1, max_rank + 1)
    slots = np.arange(1, max_slot + 1)

    prob = np.zeros((max_rank, max_slot), dtype=float)

    for i, r in enumerate(ranks):
        mu_r = alpha + beta * r
        w = np.exp(-0.5 * ((slots - mu_r) / sigma) ** 2)
        prob[i, :] = w / w.sum()

    return prob


# Build a global matrix once.
# We can allow more ranks than you typically have (e.g. 20) and 8 slots.
MAX_RANK_FOR_MATRIX = 20
MAX_SLOT_FOR_MATRIX = 8
SLOT_PROB_MATRIX = build_slot_prob_matrix_smooth(
    max_rank=MAX_RANK_FOR_MATRIX,
    max_slot=MAX_SLOT_FOR_MATRIX,
)


# Global (filled in main)
RIDER_PROB_MAP: Dict = {}   # NEW

def norm(x: str) -> str:
    """Normalize names to be case-insensitive + strip whitespace."""
    if not isinstance(x, str):
        return ""
    return x.strip().lower()


# ============================================================
# 2. DATA LOADING
# ============================================================

def load_schedule(path: str) -> pd.DataFrame:
    """
    Expected columns: game_id, home_team, away_team
    """
    schedule = pd.read_csv(path)
    required_cols = {"game_id", "home_team", "away_team"}
    missing = required_cols - set(schedule.columns)
    if missing:
        raise ValueError(f"Schedule missing columns: {missing}")

    # Normalize team codes so matching is case-insensitive
    schedule["home_team"] = schedule["home_team"].apply(norm)
    schedule["away_team"] = schedule["away_team"].apply(norm)

    return schedule


def load_rosters(path: str) -> pd.DataFrame:
    """
    Expected columns: team, rider
    """
    rosters = pd.read_csv(path)
    required_cols = {"team", "rider"}
    missing = required_cols - set(rosters.columns)
    if missing:
        raise ValueError(f"Rosters missing columns: {missing}")
    return rosters
def load_bulls(path: str) -> pd.DataFrame:
    """
    Expected columns: at minimum 'Bull'.

    You can optionally have:
    - delivery (L/R)
    - difficulty, etc.
    """
    bulls = pd.read_csv(path)
    if "bull_id" not in bulls.columns:
        raise ValueError(f"Bulls file must have a 'bull_id' column. Found: {list(bulls.columns)}")
    if bulls.empty:
        raise ValueError("Bulls file is empty – need at least one bull.")
    return bulls


def load_rider_ratings(path: str) -> pd.DataFrame:
    """
    Expected columns:
    - rider
    - ride_prob  (float between 0 and 1)

    Optional:
    - team  (if present, we’ll key by (rider, team) first)
    """
    ratings = pd.read_csv(path)

    if "rider" not in ratings.columns or "ride_prob" not in ratings.columns:
        raise ValueError(
            f"rider_ratings must contain 'rider' and 'ride_prob' columns. "
            f"Found: {list(ratings.columns)}"
        )

    # Basic sanity clamp on probabilities
    ratings["ride_prob"] = ratings["ride_prob"].clip(0.0, 1.0)

    return ratings

def load_matchup_probs(path: str) -> pd.DataFrame:
    """
    Expected columns:
    - rider
    - bull_id
    - ride_prob
    """
    df = pd.read_csv(path)

    required = {"rider", "bull_id", "ride_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"matchup_probs missing columns: {missing}. Found: {list(df.columns)}")

    df["ride_prob"] = df["ride_prob"].clip(0.0, 1.0)
    return df


def build_matchup_prob_map(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """
    Build lookup for (rider, bull_id) -> ride_prob.
    Rider and bull_id are normalized so they match roster_map & bulls_df usage.
    """
    prob_map: Dict[Tuple[str, str], float] = {}
    for _, row in df.iterrows():
        rider = norm(row["rider"])
        bull = str(row["bull_id"]).strip()
        prob = float(row["ride_prob"])
        prob_map[(rider, bull)] = prob
    return prob_map


def build_rider_prob_map(ratings_df: pd.DataFrame) -> Dict:
    """
    Build a lookup map for rider probabilities.

    If 'team' column exists, we build:
        map[(rider, team)] = ride_prob
    Always also build:
        map[rider] = ride_prob  (fallback)

    All keys are normalized (lowercase, stripped).
    """
    prob_map: Dict = {}
    has_team = "team" in ratings_df.columns

    for _, row in ratings_df.iterrows():
        rider = norm(row["rider"])
        prob = float(row["ride_prob"])
        prob_map[rider] = prob

        if has_team:
            team = norm(row["team"])
            prob_map[(rider, team)] = prob

    return prob_map


def select_bulls(bulls_df: pd.DataFrame, num_bulls: int) -> List[str]:
    """
    Randomly select num_bulls bull_ids without replacement.

    LATER:
    - We can add logic by delivery, difficulty, pen, etc.
    """
    if len(bulls_df) < num_bulls:
        # allow reuse if the pool is tiny
        idx = rng.choice(len(bulls_df), size=num_bulls, replace=True)
    else:
        idx = rng.choice(len(bulls_df), size=num_bulls, replace=False)

    return list(bulls_df.iloc[idx]["bull_id"].values)



def build_roster_map(rosters: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Convert rosters df into {team: [rider1, rider2, ...]}
    Team and rider names are normalized (lowercase, stripped).
    """
    roster_map: Dict[str, List[str]] = {}

    for _, row in rosters.iterrows():
        team = norm(row["team"])
        rider = norm(row["rider"])
        if team not in roster_map:
            roster_map[team] = []
        roster_map[team].append(rider)

    return roster_map

def apply_trade_to_roster_map(
    roster_map: Dict[str, List[str]],
    rider_a: str,
    rider_b: str
) -> Dict[str, List[str]]:
    """
    Return a NEW roster_map where rider_a and rider_b have been swapped
    between whatever teams they are currently on.

    Rider/team names are treated case-insensitively via norm().
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
# 3. RIDE PROBABILITY HOOK
# ============================================================

def get_ride_probability(
    rider: str,
    team: str,
    bull_id: str = None,
    base_prob: float = BASE_RIDE_PROB
) -> float:
    """
    Probability hierarchy (all normalized):

    1. If (rider, bull_id) is in MATCHUP_PROB_MAP, use that.
    2. Else if rider is in RIDER_PROB_MAP, use rider-level rating.
    3. Else fallback to base_prob.
    """
    r = norm(rider)
    b = str(bull_id).strip() if bull_id is not None else None

    # 1) Rider–bull specific probability
    if b is not None and (r, b) in MATCHUP_PROB_MAP:
        return MATCHUP_PROB_MAP[(r, b)]

    # 2) Rider-level probability
    if r in RIDER_PROB_MAP:
        return RIDER_PROB_MAP[r]

    # 3) Fallback
    return base_prob


# ============================================================
# 4. LINEUP SELECTION
# ============================================================

def select_lineup(
    roster: List[str],
    team: str,
    lineup_size: int = LINEUP_SIZE
) -> List[str]:
    """
    Chalk lineup:

    - Score every rider by get_ride_probability(rider, team).
    - Sort by that probability, descending.
    - Take the top lineup_size riders.

    If roster has <= lineup_size riders, use all of them.
    """
    if len(roster) <= lineup_size:
        return roster.copy()

    scored = []
    for rider in roster:
        p = get_ride_probability(rider=rider, team=team)
        scored.append((rider, p))

    # Sort by probability, highest first
    scored.sort(key=lambda x: x[1], reverse=True)

    # Take top N riders
    top_riders = [r for (r, _) in scored[:lineup_size]]
    return top_riders

def select_lineup_with_availability(
    roster: List[str],
    team: str,
    lineup_size: int = LINEUP_SIZE,
    start_prob: float = ELITE_START_PROB
) -> List[str]:
    """
    Chalk lineup with availability:
    - Score every rider by get_ride_probability(rider, team).
    - Go in descending order of ability.
    - Each rider is 'available' with probability start_prob.
    - Take the best available riders until we have lineup_size.

    If we somehow end up short (extreme randomness), fall back to strict chalk.
    """
    if len(roster) <= lineup_size:
        return roster.copy()

    # Score roster
    scored = []
    for rider in roster:
        p = get_ride_probability(rider=rider, team=team)
        scored.append((rider, p))

    scored.sort(key=lambda x: x[1], reverse=True)

    chosen = []
    for rider, _ in scored:
        if len(chosen) >= lineup_size:
            break

        if rng.random() < start_prob:
            chosen.append(rider)

    # If we didn’t get enough riders (bad luck), just fill in from chalk
    if len(chosen) < lineup_size:
        # add best remaining riders not already chosen
        for rider, _ in scored:
            if len(chosen) >= lineup_size:
                break
            if rider not in chosen:
                chosen.append(rider)

    return chosen


def select_lineup_smooth_slots(
    roster: List[str],
    team: str,
    lineup_size: int = LINEUP_SIZE,
) -> List[str]:
    """
    Slot-smooth lineup selection.

    Steps:
    - Compute base ride_prob per rider (your get_ride_probability, no bull).
    - Rank riders descending by ride_prob (rank 1 = best).
    - For each slot s in 1..lineup_size:
        * Among riders not yet chosen, sample according to
          P(slot=s | rank=r) from SLOT_PROB_MATRIX.

    If roster has <= lineup_size riders, fall back to simple chalk.
    """
    if len(roster) <= lineup_size or not USE_SMOOTH_SLOTS:
        # Fallback to deterministic chalk lineup
        return select_lineup(roster, team, lineup_size)

    # 1) Score and rank riders by "talent"
    riders = []
    for rider in roster:
        p = get_ride_probability(rider=rider, team=team)
        riders.append((rider, p))

    # Sort by probability descending (best first)
    riders.sort(key=lambda x: x[1], reverse=True)
    ranked_riders = [r for (r, _) in riders]  # index 0 = rank 1

    # Helper: get rank index (1-based) for a rider
    rank_index = {rider: i + 1 for i, rider in enumerate(ranked_riders)}

    chosen: List[str] = []

    max_slot_used = min(lineup_size, SLOT_PROB_MATRIX.shape[1])

    for slot in range(1, lineup_size + 1):
        # Candidates: riders not yet chosen
        candidates = [r for r in ranked_riders if r not in chosen]
        if not candidates:
            break

        # If we have more slots than configured in matrix, just clamp to max slot
        slot_idx = min(slot, max_slot_used) - 1

        weights = []
        for r in candidates:
            r_idx = rank_index[r]
            r_idx_clamped = min(r_idx, SLOT_PROB_MATRIX.shape[0])
            w = SLOT_PROB_MATRIX[r_idx_clamped - 1, slot_idx]
            weights.append(w)

        weights = np.array(weights, dtype=float)
        if weights.sum() <= 0:
            # Safeguard: if all weights are 0 (weird edge case), use uniform
            probs = np.full_like(weights, 1.0 / len(weights))
        else:
            probs = weights / weights.sum()

        # Sample one rider for this slot
        chosen_idx = rng.choice(len(candidates), p=probs)
        chosen_rider = candidates[chosen_idx]
        chosen.append(chosen_rider)

    return chosen


# ============================================================
# 5. SIMULATE A SINGLE GAME
# ============================================================

def simulate_game(
    game_row: pd.Series,
    roster_map: Dict[str, List[str]],
    bulls_df: pd.DataFrame,
    lineup_size: int = LINEUP_SIZE
) -> (dict, List[dict]):
    """
    Simulate one game.

    Returns:
    - game_summary: dict with game-level results
    - ride_rows: list of dicts, one per ride
    """
    home = game_row["home_team"]
    away = game_row["away_team"]
    game_id = game_row["game_id"]

    # Get rosters
    if home not in roster_map:
        raise ValueError(f"Team {home} not found in roster_map")
    if away not in roster_map:
        raise ValueError(f"Team {away} not found in roster_map")

    home_roster = roster_map[home]
    away_roster = roster_map[away]

    # Select lineups
    home_lineup = select_lineup_smooth_slots(home_roster, home, lineup_size)
    away_lineup = select_lineup_smooth_slots(away_roster, away, lineup_size)




    # Select bulls (one bull per rider)
    home_bulls = select_bulls(bulls_df, len(home_lineup))
    away_bulls = select_bulls(bulls_df, len(away_lineup))

    home_scores = []
    away_scores = []
    ride_rows: List[dict] = []

    # Home rides
    for rider, bull in zip(home_lineup, home_bulls):
        p = get_ride_probability(rider=rider, team=home, bull_id=bull)

        ride_made = int(rng.random() < p)

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

    # Away rides
    for rider, bull in zip(away_lineup, away_bulls):
        p = get_ride_probability(rider=rider, team=away, bull_id=bull)

        ride_made = int(rng.random() < p)

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

    # Determine winner / loser / tie
    if home_total > away_total:
        winner = home
        loser = away
        result = "home_win"
    elif away_total > home_total:
        winner = away
        loser = home
        result = "away_win"
    else:
        # COIN FLIP FOR TIES
        if rng.random() < 0.5:
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
# 6. SIMULATE FULL SEASON
# ============================================================

def simulate_season(
    schedule: pd.DataFrame,
    roster_map: Dict[str, List[str]],
    bulls_df: pd.DataFrame,
    lineup_size: int = LINEUP_SIZE
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Returns:
    - game_results_df: one row per game
    - standings_df: one row per team with W/L and win%
    - rides_df: one row per ride
    """
    game_results = []
    all_rides = []

    for _, row in schedule.iterrows():
        game_summary, ride_rows = simulate_game(
            row,
            roster_map,
            bulls_df,
            lineup_size
        )
        game_results.append(game_summary)
        all_rides.extend(ride_rows)

    game_results_df = pd.DataFrame(game_results)
    rides_df = pd.DataFrame(all_rides)

    # Build standings
    teams = sorted(set(game_results_df["home_team"]) | set(game_results_df["away_team"]))

    records = []
    for team in teams:
        wins = ((game_results_df["winner"] == team)).sum()
        losses = ((game_results_df["loser"] == team)).sum()

        is_home = game_results_df["home_team"] == team
        is_away = game_results_df["away_team"] == team
        played = is_home | is_away
        ties = ((game_results_df["winner"].isna()) & played).sum()

        games = wins + losses + ties
        win_pct = wins / games if games > 0 else np.nan

        records.append({
            "team": team,
            "games": games,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_pct": win_pct
        })

    standings_df = (
        pd.DataFrame(records)
        .sort_values(by=["win_pct", "wins"], ascending=[False, False])
        .reset_index(drop=True)
    )

    # Compute riding % by team
    team_ride_stats = (
        rides_df
        .groupby("team", as_index=False)
        .agg(
            team_total_rides=("ride_made", "size"),
            team_rides_made=("ride_made", "sum")
        )
    )
    team_ride_stats["team_ride_pct"] = team_ride_stats["team_rides_made"] / team_ride_stats["team_total_rides"]

    # Merge into standings_df
    standings_df = standings_df.merge(team_ride_stats, on="team", how="left")

    return game_results_df, standings_df, rides_df



def compute_mvp_standings(rides_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simple MVP-style leaderboard from ride-level data.

    For each rider (and team), compute:
    - total_rides
    - rides_made
    - ride_pct

    Then sort by rides_made, ride_pct, and total_rides.
    """
    if rides_df.empty:
        return pd.DataFrame(columns=["rider", "team", "total_rides", "rides_made", "ride_pct"])

    # Ensure required columns exist
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
            ascending=[False, False, False]
        )
        .reset_index(drop=True)
    )

    return mvp

def simulate_many_seasons(
    n: int,
    schedule_df: pd.DataFrame,
    roster_map: Dict[str, List[str]],
    bulls_df: pd.DataFrame,
    seed: int = RNG_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the full season simulation n times.
    
    Returns:
    - many_standings_df: long-format team standings (regular season)
    - many_mvp_df: long-format MVP standings (per season)
    - many_playoffs_df: playoff placements (per season)
    - many_prizes_df: prize money per team per season
    """
    all_standings = []
    all_mvp = []
    all_playoffs = []
    all_prizes = []

    for season_id in range(1, n + 1):
        # Reset global RNG seed per season for reproducibility
        global rng
        rng = np.random.default_rng(seed + season_id)

        # --- Regular season ---
        game_results_df, standings_df, rides_df = simulate_season(
            schedule_df, roster_map, bulls_df
        )

        # Rank teams within this season by win_pct (no ties because of coin flip)
        standings_df = standings_df.copy()
        standings_df["season_id"] = season_id
        standings_df["rank"] = standings_df["win_pct"].rank(
            method="first", ascending=False
        )
        all_standings.append(standings_df)

        # --- MVP standings for this season ---
        mvp_df = compute_mvp_standings(rides_df).copy()
        mvp_df["season_id"] = season_id
        mvp_df["rank"] = np.arange(1, len(mvp_df) + 1)
        all_mvp.append(mvp_df)

        # --- Playoffs for this season ---
        playoff_games_df, playoff_rides_df, placements_df = simulate_playoffs(
            standings_df, roster_map, bulls_df
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


def run_trade_comparison(
    rider_a: str,
    rider_b: str,
    n_sims: int,
    schedule_df: pd.DataFrame,
    roster_map: Dict[str, List[str]],
    bulls_df: pd.DataFrame,
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    """
    Run side-by-side 1000-season (or n_sims) sims:

    - Baseline: original roster_map
    - Trade: rider_a and rider_b swapped between their current teams

    Returns:
        comparison_df with columns:
        team,
        avg_wins_base, avg_wins_trade, delta_wins,
        avg_win_pct_base, avg_win_pct_trade, delta_win_pct,
        mean_total_prize_base, mean_total_prize_trade, delta_total_prize
    """
    # --- Baseline ---
    many_standings_base, many_mvp_base, many_playoffs_base, many_prizes_base = (
        simulate_many_seasons(
            n_sims, schedule_df, roster_map, bulls_df, seed=seed
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
            n_sims, schedule_df, traded_roster_map, bulls_df, seed=seed
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

    # Optional: round to 3 decimals for wins, 4 for win%, 2 for money
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

# ============================================================
# PLAYOFF HELPERS
# ============================================================

def seed_teams_from_standings(standings_df: pd.DataFrame) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Take a regular-season standings df (one season) and assign seeds 1–10.
    Lower seed number = better team (1 is best).

    Returns:
        seed_to_team: {1: "team_a", 2: "team_b", ...}
        team_to_seed: {"team_a": 1, "team_b": 2, ...}
    """
    # Sort by win_pct then wins (same logic you used for standings)
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
    """
    teams = list(team_scores.keys())

    # Shuffle first for randomness, then sort by score desc
    rng.shuffle(teams)
    teams.sort(key=lambda t: team_scores[t], reverse=True)
    return teams


def simulate_two_team_game(
    team1: str,
    team2: str,
    game_id: str,
    roster_map: Dict[str, List[str]],
    bulls_df: pd.DataFrame,
    lineup_size: int = LINEUP_SIZE,
) -> Tuple[str, str, dict, List[dict]]:
    """
    Wrapper around simulate_game for a head-to-head playoff game.

    Returns:
        winner, loser, game_summary_dict, ride_rows_list
    """
    row = pd.Series({"game_id": game_id, "home_team": team1, "away_team": team2})
    game_summary, ride_rows = simulate_game(row, roster_map, bulls_df, lineup_size)

    winner = game_summary["winner"]
    loser = game_summary["loser"]

    return winner, loser, game_summary, ride_rows


def simulate_three_team_game(
    teams: List[str],
    game_id: str,
    roster_map: Dict[str, List[str]],
    bulls_df: pd.DataFrame,
    lineup_size: int = LINEUP_SIZE,
) -> Tuple[str, List[str], List[dict], List[dict]]:
    """
    Simulate a 3-team game (each team gets lineup_size rides, independent).

    Returns:
        winner: team name
        ordered_teams: [1st, 2nd, 3rd] by score (ties broken by coin flip)
        game_summaries: list of dicts, one per team (game-level view)
        ride_rows: list of ride-level dicts
    """
    assert len(teams) == 3, "3-team game requires exactly 3 teams"

    team_scores: Dict[str, int] = {}
    game_summaries: List[dict] = []
    ride_rows: List[dict] = []

    for team in teams:
        roster = roster_map[team]
        lineup = select_lineup(roster, team, lineup_size)
        bulls = select_bulls(bulls_df, len(lineup))

        score = 0
        for rider, bull in zip(lineup, bulls):
            p = get_ride_probability(rider=rider, team=team, bull_id=bull)
            ride_made = int(rng.random() < p)
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

def simulate_playoffs(
    season_standings_df: pd.DataFrame,
    roster_map: Dict[str, List[str]],
    bulls_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate the 12-game PBR Teams playoff you described.

    season_standings_df: standings for ONE season (output of simulate_season).
    Returns:
        playoff_games_df: one row per playoff game/team
        playoff_placements_df: one row per team with final playoff finish (1–10)
    """
    seed_to_team, team_to_seed = seed_teams_from_standings(season_standings_df)

    # Convenience: get team by seed number
    def T(seed: int) -> str:
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
        g1_teams, g1_id, roster_map, bulls_df
    )
    all_game_rows.extend(g1_summaries)
    all_ride_rows.extend(g1_rides)

    # Losers of Game 1 are 9th and 10th
    g1_losers = g1_ordered[1:]  # 2nd and 3rd
    placements[g1_losers[0]] = 9
    placements[g1_losers[1]] = 10

    # Game 2: 4 vs 7
    g2_id = "PO-2"
    g2_winner, g2_loser, g2_summary, g2_rides = simulate_two_team_game(
        T(4), T(7), g2_id, roster_map, bulls_df
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
    g3_winner, g3_loser, g3_summary, g3_rides = simulate_two_team_game(
        T(5), T(6), g3_id, roster_map, bulls_df
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
        g4_teams, g4_id, roster_map, bulls_df
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
    # sort by seed descending: worst seed first
    others.sort(key=lambda t: team_to_seed[t], reverse=True)
    lowest, second_lowest, third_lowest = others

    # Game 5: 1 vs Lowest Seed Remaining
    g5_id = "PO-5"
    g5_winner, g5_loser, g5_summary, g5_rides = simulate_two_team_game(
        T(1), lowest, g5_id, roster_map, bulls_df
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
    g6_winner, g6_loser, g6_summary, g6_rides = simulate_two_team_game(
        T(2), second_lowest, g6_id, roster_map, bulls_df
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
    g7_winner, g7_loser, g7_summary, g7_rides = simulate_two_team_game(
        T(3), third_lowest, g7_id, roster_map, bulls_df
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
        losers_day2, g8_id, roster_map, bulls_df
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
    g9_winner, g9_loser, g9_summary, g9_rides = simulate_two_team_game(
        highest, lowest, g9_id, roster_map, bulls_df
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
    g10_winner, g10_loser, g10_summary, g10_rides = simulate_two_team_game(
        second_highest, second_lowest, g10_id, roster_map, bulls_df
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
    g11_winner, g11_loser, g11_summary, g11_rides = simulate_two_team_game(
        g9_loser, g10_loser, g11_id, roster_map, bulls_df
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
    g12_winner, g12_loser, g12_summary, g12_rides = simulate_two_team_game(
        g9_winner, g10_winner, g12_id, roster_map, bulls_df
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

# ============================================================
# 7. MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    schedule_df = load_schedule(SCHEDULE_PATH)
    rosters_df = load_rosters(ROSTERS_PATH)
    bulls_df = load_bulls(BULLS_PATH)

    # Load rider ratings and build map
    rider_ratings_df = load_rider_ratings(RIDER_RATINGS_PATH)
    RIDER_PROB_MAP = build_rider_prob_map(rider_ratings_df)

    # NEW: load matchup-level probabilities
    matchup_probs_df = load_matchup_probs(MATCHUP_PROBS_PATH)
    MATCHUP_PROB_MAP = build_matchup_prob_map(matchup_probs_df)

    roster_map = build_roster_map(rosters_df)

    # =========================================================
    # RUN ONE EXAMPLE SEASON & PRINT ONE EXAMPLE GAME
    # =========================================================
    example_game_results_df, example_standings_df, example_rides_df = simulate_season(
        schedule_df, roster_map, bulls_df
    )

    playoff_games_df, playoff_rides_df, playoff_placements_df = simulate_playoffs(
        example_standings_df, roster_map, bulls_df
    )

    rides_team = example_rides_df[example_rides_df["team"] == "aus"]  # example
    total_by_rider = rides_team.groupby("rider")["ride_made"].size().sort_values(ascending=False)
    print(total_by_rider.head(8))

    # Pick the first game in the schedule as the example
    example_id = schedule_df.iloc[0]["game_id"]

    example_game = example_game_results_df[
        example_game_results_df["game_id"] == example_id
    ].iloc[0]

    example_rides = example_rides_df[example_rides_df["game_id"] == example_id]

    print("\n=== EXAMPLE GAME (SUMMARY) ===")
    print(example_game)

    print("\n=== EXAMPLE GAME (RIDE-LEVEL DETAILS) ===")
    print(example_rides)
    

    # =========================================================
    # RUN 1000 SEASONS
    # =========================================================
    # === RUN 1000 SEASONS + PLAYOFFS + PRIZES ===
    n_sims = 1000
    many_standings_df, many_mvp_df, many_playoffs_df, many_prizes_df = simulate_many_seasons(
        n_sims, schedule_df, roster_map, bulls_df
    )


    print("\n=== EXAMPLE OF FIRST FEW STANDINGS ROWS ===")
    print(many_standings_df.head())


    # =========================================================
    # TEAM SUMMARY: avg results + 1st / top3 / top5 %
    # =========================================================
    team_summary = (
        many_standings_df
        .groupby("team", as_index=False)
        .agg(
            avg_wins=("wins", "mean"),
            avg_losses=("losses", "mean"),
            avg_win_pct=("win_pct", "mean"),
            avg_ride_pct=("team_ride_pct", "mean"),      # NEW
            best_win_pct=("win_pct", "max"),
            worst_win_pct=("win_pct", "min"),
            first_place_pct=("rank", lambda x: np.mean(x == 1)),
            top3_pct=("rank", lambda x: np.mean(x <= 3)),
            top5_pct=("rank", lambda x: np.mean(x <= 5))
        )
        .sort_values("avg_wins", ascending=False)
    )


    print("\n=== TEAM AVERAGE RESULTS ACROSS 1000 SEASONS ===")
    print(team_summary)

    # =========================================================
    # RIDER MVP SUMMARY: 1st / top3 / top5 / top10 %
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
    print(rider_summary.head(100))
    # Compute league-wide ride% for each season
    league_pct_per_season = (
        many_mvp_df.groupby("season_id")["rides_made"].sum()
        / many_mvp_df.groupby("season_id")["total_rides"].sum()
    )

    # =========================================================
    # PLAYOFF SUMMARY: distribution of finishes 1–10
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
    # PRIZE SUMMARY (MEAN / MEDIAN / IQR)
    # =========================================================
    # =========================================================
    # PRIZE SUMMARY (MEAN / MEDIAN / P25 / P75 / IQR)
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

    # Example trade: swap Callum Miller and Thiago Salgado
    trade_rider_a = "Callum Miller"
    trade_rider_b = "Riquelmi Santos"

    n_sims = 1

    trade_comparison_df = run_trade_comparison(
        trade_rider_a,
        trade_rider_b,
        n_sims,
        schedule_df,
        roster_map,
        bulls_df,
        seed=RNG_SEED,
    )

    print("\n=== TRADE COMPARISON ===")
    print(trade_comparison_df.sort_values("delta_wins", ascending=False))


    print("\n=== LEAGUE-WIDE RIDE % (ACROSS 1000 SEASONS) ===")
    print("Mean:", round(league_pct_per_season.mean(), 4))
    print("Std Dev:", round(league_pct_per_season.std(), 4))
    print("Min:", round(league_pct_per_season.min(), 4))
    print("Max:", round(league_pct_per_season.max(), 4))

# =========================================================
# EXPORT ALL RESULTS TO EXCEL
# =========================================================

OUTPUT_XLSX = "simulation_outputs_full.xlsx"

with pd.ExcelWriter(OUTPUT_XLSX) as writer:


    # ---------- Single season ----------
    example_game_results_df.to_excel(
        writer, sheet_name="Example_Game_Results", index=False
    )
    example_standings_df.to_excel(
        writer, sheet_name="Example_Standings", index=False
    )
    example_rides_df.to_excel(
        writer, sheet_name="Example_Rides", index=False
    )

    playoff_games_df.to_excel(
        writer, sheet_name="Playoff_Games", index=False
    )
    playoff_rides_df.to_excel(
        writer, sheet_name="Playoff_Rides", index=False
    )
    playoff_placements_df.to_excel(
        writer, sheet_name="Playoff_Placements", index=False
    )

    # ---------- Multi-season raw ----------
    many_standings_df.to_excel(
        writer, sheet_name="All_Standings_Long", index=False
    )
    many_mvp_df.to_excel(
        writer, sheet_name="All_MVP_Long", index=False
    )
    many_playoffs_df.to_excel(
        writer, sheet_name="All_Playoffs_Long", index=False
    )
    many_prizes_df.to_excel(
        writer, sheet_name="All_Prizes_Long", index=False
    )

    # ---------- Aggregated summaries ----------
    team_summary.to_excel(
        writer, sheet_name="Team_Summary", index=False
    )
    rider_summary.to_excel(
        writer, sheet_name="Rider_Summary", index=False
    )
    playoff_summary.to_excel(
        writer, sheet_name="Playoff_Summary", index=False
    )
    prize_summary.to_excel(
        writer, sheet_name="Prize_Summary", index=False
    )

    # ---------- Trade analysis ----------
    trade_comparison_df.to_excel(
        writer, sheet_name="Trade_Comparison", index=False
    )

print(f"\nAll outputs written to {OUTPUT_XLSX}")

