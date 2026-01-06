# lineups.py

from itertools import combinations, permutations
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
from .predictions import predict_multiple
from .config import DEFAULT_DATE

def top_k_lineups(predictions: List[Dict], k: int = 5) -> List[Dict]:
    """
    Exhaustively evaluate ALL valid lineups:
      - choose m riders (m = #bulls),
      - assign them to ALL m! permutations of the bulls,
      - score by sum of ride probabilities (missing pairs count as 0.0),
      - return the global top-K.

    WARNING: This is factorial. With many riders/bulls, it will be slow by design.
    """
    # Build lookup and entity sets
    prob_lookup = {(p["rider"], p["bull"]): float(p["probability"]) for p in predictions}
    riders_all = sorted({p["rider"] for p in predictions})
    bulls_all  = sorted({p["bull"]  for p in predictions})

    m = len(bulls_all)
    if m == 0 or len(riders_all) < m:
        return []

    all_lineups: List[Dict] = []

    # Exhaustive: every size-m subset of riders, and every permutation of bulls
    for riders_subset in combinations(riders_all, m):
        for bulls_perm in permutations(bulls_all, m):
            lineup = []
            total = 0.0

            for r, b in zip(riders_subset, bulls_perm):
                # DO NOT SKIP: treat missing as 0.0 so we truly score every lineup
                prob = prob_lookup.get((r, b), 0.0)
                total += prob
                lineup.append({"rider": r, "bull": b, "probability": round(prob, 4)})

            all_lineups.append({
                "total_probability": round(total, 4),
                "lineup": lineup,
            })

    # Sort globally and cap to K
    all_lineups.sort(key=lambda x: -x["total_probability"])
    return all_lineups[:k]
def extract_lineup_insights(lineups: List[Dict], min_appearances: int = 2) -> Dict:
    """
    Summarize patterns across top lineups: recurring matchups, frequent riders/bulls.
    """
    matchup_counts = Counter()
    rider_counts = Counter()
    bull_counts = Counter()
    matchup_probs = defaultdict(list)

    for entry in lineups:
        for match in entry["lineup"]:
            rider = match["rider"]
            bull = match["bull"]
            prob = match["probability"]
            key = f"{rider} on {bull}"
            matchup_counts[key] += 1
            rider_counts[rider] += 1
            bull_counts[bull] += 1
            matchup_probs[key].append(prob)

    insights = {
        "recurring_matchups": [],
        "most_used_riders": rider_counts.most_common(),
        "most_used_bulls": bull_counts.most_common()
    }

    for matchup, count in matchup_counts.items():
        if count >= min_appearances:
            avg_prob = sum(matchup_probs[matchup]) / len(matchup_probs[matchup])
            insights["recurring_matchups"].append({
                "matchup": matchup,
                "count": count,
                "avg_probability": round(avg_prob, 4)
            })

    insights["recurring_matchups"].sort(key=lambda x: (-x["count"], -x["avg_probability"]))
    return insights

def get_top_lineups_with_insights(riders, bulls, event_date=DEFAULT_DATE, k=5):
    """
    Generate top-k lineups for given riders & bulls and print insights in a readable format.
    """
    results = predict_multiple(riders, bulls, event_date=event_date)
    lineups = top_k_lineups(results, k=k)
    insights = extract_lineup_insights(lineups)

    # --- Print Top Lineups ---
    print("\n=== Top Lineups ===")
    for i, entry in enumerate(lineups, 1):
        print(f"\nLineup {i} | Total Probability: {entry['total_probability']:.2f}")
        for match in entry["lineup"]:
            print(f"  - {match['rider']} on {match['bull']} → {match['probability']:.2%}")

    # --- Print Insights ---
    print("\n=== Insights Summary ===")

    if insights["recurring_matchups"]:
        print("\n-- Recurring Matchups --")
        for match in insights["recurring_matchups"]:
            print(f"  - {match['matchup']} | Appears {match['count']} times | Avg Prob: {match['avg_probability']:.2%}")
    else:
        print("\n-- Recurring Matchups -- None found --")

    print("\n-- Most Used Riders --")
    if insights["most_used_riders"]:
        for rider, count in insights["most_used_riders"]:
            print(f"  - {rider}: {count} times")
    else:
        print("  None")

    print("\n-- Most Used Bulls --")
    if insights["most_used_bulls"]:
        for bull, count in insights["most_used_bulls"]:
            print(f"  - {bull}: {count} times")
    else:
        print("  None")
    
    
