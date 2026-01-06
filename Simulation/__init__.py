"""Simulation package public API.

This package provides a decoupled season-simulation pipeline:

Functions exported
------------------
- build_qr_table                       (qr_table)
- pre_generate_bull_pens               (pens)
- pre_generate_lineups                 (lineups)
- assign_lineups_to_schedule           (assign)
- simulate_season_from_lineups_fast    (simulate)
- aggregate_season_results             (aggregate)

Usage
-----
from simulation import (
    build_qr_table, pre_generate_bull_pens, pre_generate_lineups,
    assign_lineups_to_schedule, simulate_season_from_lineups_fast,
    aggregate_season_results,
)
"""

from .qr_table import build_qr_table
from .pens import pre_generate_bull_pens
from .lineups import pre_generate_lineups
from .assign import assign_lineups_to_schedule
from .simulate import simulate_season_from_lineups_fast
from .aggregate import aggregate_season_results
from .types import BullPen, TeamLineup, GameAssignment, Team


__all__ = [
    "build_qr_table",
    "pre_generate_bull_pens",
    "pre_generate_lineups",
    "assign_lineups_to_schedule",
    "simulate_season_from_lineups_fast",
    "aggregate_season_results",
    "Team",
]

__version__ = "0.1.0"
