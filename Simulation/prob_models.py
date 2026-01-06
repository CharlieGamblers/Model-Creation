# prob_models.py

import numpy as np
import pandas as pd
from typing import Dict, Tuple

# ============================================================
# NAME NORMALIZATION
# ============================================================

def norm(x: str) -> str:
    """Normalize names to be case-insensitive + strip whitespace."""
    if not isinstance(x, str):
        return ""
    return x.strip().lower()

# RNG configuration - import from data_io to avoid duplication
from data_io import RNG_SEED

# ============================================================
# CONFIG / GLOBALS
# ============================================================

BASE_RIDE_PROB = 0.4

# RNG setup - initialized here, can be reset per season
rng = np.random.default_rng(RNG_SEED)

# These get filled in at runtime by your runner
RIDER_PROB_MAP: Dict = {}                  # rider -> ride_prob
MATCHUP_PROB_MAP: Dict[Tuple[str, str], float] = {}  # (rider, bull_id) -> ride_prob

# ============================================================
# PROBABILITY MAP BUILDERS
# ============================================================

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

# ============================================================
# SLOT USAGE MODEL (SMOOTH)
# ============================================================

USE_SMOOTH_SLOTS = True  # toggle externally if needed

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
    Returns:
        prob_matrix[r-1, s-1] = P(slot = s | rank = r)
    """
    ranks = np.arange(1, max_rank + 1)
    slots = np.arange(1, max_slot + 1)

    prob = np.zeros((max_rank, max_slot), dtype=float)

    for i, r in enumerate(ranks):
        mu_r = alpha + beta * r
        w = np.exp(-0.5 * ((slots - mu_r) / sigma) ** 2)
        prob[i, :] = w / w.sum()

    return prob

# Precompute default matrix; other modules can import this
MAX_RANK_FOR_MATRIX = 20
MAX_SLOT_FOR_MATRIX = 8

SLOT_PROB_MATRIX = build_slot_prob_matrix_smooth(
    max_rank=MAX_RANK_FOR_MATRIX,
    max_slot=MAX_SLOT_FOR_MATRIX,
)

# ============================================================
# DATA LOADING (for probability models)
# ============================================================

def load_rider_ratings(path: str) -> pd.DataFrame:
    """
    Expected columns:
    - rider
    - ride_prob  (float between 0 and 1)

    Optional:
    - team  (if present, we'll key by (rider, team) first)
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

# ============================================================
# PROBABILITY MAP BUILDERS
# ============================================================

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
        map[(rider, team)] = ride_prob  (optional, if you ever want it)
    Always also build:
        map[rider] = ride_prob

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

# ============================================================
# CORE RIDE PROBABILITY HOOK
# ============================================================

def get_ride_probability(
    rider: str,
    team: str,
    bull_id: str = None,
    base_prob: float = BASE_RIDE_PROB,
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
