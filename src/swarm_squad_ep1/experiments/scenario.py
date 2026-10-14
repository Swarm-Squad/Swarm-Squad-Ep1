"""
Scenario definitions (JSON) and seeded instances.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import swarm_squad_ep1.config as config

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIO_DIR = REPO_ROOT / "experiments" / "scenarios"

MODES = {mode.value: mode for mode in config.ObstacleMode}


@dataclass
class Scenario:
    name: str
    label: str
    mode: config.ObstacleMode
    obstacles: np.ndarray  # (k, 3): x, y, radius
    position_jitter: float
    obstacle_jitter: float
    source: str

    @classmethod
    def load(cls, name: str) -> "Scenario":
        path = SCENARIO_DIR / f"{name}.json"
        spec = json.loads(path.read_text())
        return cls(
            name=spec["name"],
            label=spec["label"],
            mode=MODES[spec["mode"]],
            obstacles=np.array(spec["obstacles"], dtype=float),
            position_jitter=float(spec["jitter"]["initial_positions"]),
            obstacle_jitter=float(spec["jitter"]["obstacle_centers"]),
            source=spec.get("source", ""),
        )

    def instance(self, seed: int) -> dict:
        """
        Initial positions and obstacles for one seed.

        Each coordinate is jittered uniformly in [-j, j]; obstacle radii are
        not changed. Every supervisor sees the same instance for a seed.
        """
        rng = np.random.default_rng(seed)
        base = config.INITIAL_SWARM_POSITIONS.astype(float)
        positions = base + rng.uniform(
            -self.position_jitter, self.position_jitter, size=base.shape
        )
        obstacles = self.obstacles.copy()
        obstacles[:, :2] += rng.uniform(
            -self.obstacle_jitter, self.obstacle_jitter, size=(len(obstacles), 2)
        )
        return {
            "seed": seed,
            "positions": positions,
            "obstacles": obstacles,
            "destination": config.DEFAULT_DESTINATION.astype(float).copy(),
        }
