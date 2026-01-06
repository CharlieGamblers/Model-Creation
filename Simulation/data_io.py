import pandas as pd
import numpy as np
from typing import Dict, List

# ============================================================
# 1. CONFIG
# ============================================================
SCHEDULE_PATH = "schedule.csv"
ROSTERS_PATH = "rosters.csv"
BULLS_PATH = "bulls.csv"
RIDER_RATINGS_PATH = "rider_ratings.csv"
MATCHUP_PROBS_PATH = "rider_bull_probs.csv"
LINEUP_SIZE = 5

RNG_SEED = 42

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
# NAME NORMALIZATION
# ============================================================

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




def select_bulls(bulls_df: pd.DataFrame, num_bulls: int) -> List[str]:
    """
    Randomly select num_bulls bull_ids without replacement.

    LATER:
    - We can add logic by delivery, difficulty, pen, etc.
    """
    # Import here to avoid circular import
    import prob_models
    if len(bulls_df) < num_bulls:
        # allow reuse if the pool is tiny
        idx = prob_models.rng.choice(len(bulls_df), size=num_bulls, replace=True)
    else:
        idx = prob_models.rng.choice(len(bulls_df), size=num_bulls, replace=False)

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
