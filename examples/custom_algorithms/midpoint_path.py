"""Example custom path algorithm plugin for Swarm Squad Ep1."""

from __future__ import annotations

import numpy as np


def midpoint_path(start, goal, jamming_zones, **kwargs):
    """
    Simple custom path planner.

    Returns a three-waypoint route (start -> midpoint -> goal).
    The runtime normalizes start/goal if they are omitted.
    """
    start = np.array(start, dtype=float)
    goal = np.array(goal, dtype=float)
    midpoint = (start + goal) / 2.0
    midpoint[2] = max(start[2], goal[2])  # keep altitude transitions simple
    return [start, midpoint, goal]
