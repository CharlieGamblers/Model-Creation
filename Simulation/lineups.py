# lineups.py

from typing import List, Dict
import numpy as np

import prob_models
from prob_models import get_ride_probability, SLOT_PROB_MATRIX

LINEUP_SIZE_DEFAULT = 5


def select_lineup(
    roster: List[str],
    team: str,
    lineup_size: int = LINEUP_SIZE_DEFAULT
) -> List[str]:
    """
    Deterministic chalk lineup:

    - Score every rider by base ride probability (vs generic bull).
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


def _compute_team_skill_weights(
    roster: List[str],
    team: str
) -> Dict[str, float]:
    """
    Compute *relative* skill weights for each rider on a team.

    weight[rider] ∝ base ride_prob(rider, team)

    These are used to make must-start guys (0.75+ QR) show up
    much more often than role players, and to distinguish teams
    with 3 elites (AUS) from teams with 1 elite (TEX).
    """
    skill: Dict[str, float] = {}
    for rider in roster:
        p = get_ride_probability(rider=rider, team=team)
        # small epsilon so true zeros don't kill weights
        skill[rider] = max(float(p), 1e-6)

    total = sum(skill.values())
    if total <= 0:
        # Degenerate case: fall back to uniform
        n = len(roster)
        return {r: 1.0 / n for r in roster}

    return {r: v / total for r, v in skill.items()}


def select_lineup_smooth_slots(
    roster: List[str],
    team: str,
    lineup_size: int = LINEUP_SIZE_DEFAULT,
    use_smooth_slots: bool = True,
) -> List[str]:
    """
    Slot-smooth lineup selection with SKILL WEIGHTING.

    Steps:
    1. Compute base skill for each rider:
         skill[r] = get_ride_probability(r, team)
    2. Rank riders by skill (rank 1 = best).
    3. For each slot s in 1..lineup_size:
         - Among riders not yet chosen, give each candidate a weight:
               weight[r] = SLOT_PROB_MATRIX[rank(r), s] * skill_weight[r]
           where skill_weight[r] is that rider's relative strength
           inside THIS team.
         - Sample one rider for that slot with probability
           ∝ weight[r].

    This does two things simultaneously:
      • Uses your calibrated slot curve (ranks vs slots)
      • Respects team-specific star depth (multiple elites vs one)
    """
    # Fallback: small rosters or disabled smooth model
    if len(roster) <= lineup_size or not use_smooth_slots:
        return select_lineup(roster, team, lineup_size)

    # 1) Compute base skill per rider
    #    and sort by skill to get ranks (1 = best).
    rider_skill: Dict[str, float] = {}
    for rider in roster:
        p = get_ride_probability(rider=rider, team=team)
        rider_skill[rider] = max(float(p), 1e-6)

    ranked_riders = sorted(
        roster,
        key=lambda r: rider_skill[r],
        reverse=True
    )  # index 0 == rank 1

    rank_index: Dict[str, int] = {r: i + 1 for i, r in enumerate(ranked_riders)}

    # 2) Team-relative skill weights
    skill_weight = _compute_team_skill_weights(ranked_riders, team)

    chosen: List[str] = []

    max_rank_rows = SLOT_PROB_MATRIX.shape[0]
    max_slot_cols = SLOT_PROB_MATRIX.shape[1]

    for slot in range(1, lineup_size + 1):
        candidates = [r for r in ranked_riders if r not in chosen]
        if not candidates:
            break

        # Clamp slot index if we ever use more slots than configured
        slot_idx = min(slot, max_slot_cols) - 1

        weights = []
        for r in candidates:
            r_idx = rank_index[r]
            r_idx_clamped = min(r_idx, max_rank_rows)

            # Base slot curve weight for this rank/slot
            slot_w = float(SLOT_PROB_MATRIX[r_idx_clamped - 1, slot_idx])

            # Skill modifier: stronger riders get extra mass
            w = slot_w * skill_weight[r]
            weights.append(w)

        w_arr = np.asarray(weights, dtype=float)
        if w_arr.sum() <= 0:
            # Shouldn't happen often, but guard anyway
            probs = np.full(len(candidates), 1.0 / len(candidates))
        else:
            probs = w_arr / w_arr.sum()

        # Use the shared RNG so results are reproducible with seed
        chosen_idx = prob_models.rng.choice(len(candidates), p=probs)
        chosen_rider = candidates[chosen_idx]
        chosen.append(chosen_rider)

    return chosen
