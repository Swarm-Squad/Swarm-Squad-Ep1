"""
Scenario definitions for the research harness.

A ``Scenario`` bundles everything needed for one headless simulation run:
geometry, attack/countermeasure toggles, comm model, and RNG seed.

Keep this module pure dataclasses (no FastAPI, no globals) so scenarios
can be pickled, emitted to JSON, and composed into large matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JammingZoneSpec:
    center: tuple[float, float, float]
    radius: float
    obstacle_type: str = "low_jam"  # "physical" | "low_jam" | "high_jam"


@dataclass
class SpoofingZoneSpec:
    center: tuple[float, float, float]
    radius: float
    spoof_type: str = "phantom"  # "phantom" | "position_falsification" | "coordinate"
    phantom_count: int = 2
    falsification_magnitude: float = 8.0
    coordinate_vector: tuple[float, float, float] = (10.0, 10.0, 0.0)


@dataclass
class Scenario:
    name: str
    seed: int = 0
    num_agents: int = 5
    destination: tuple[float, float, float] = (35.0, 150.0, 30.0)
    agent_init_positions: Optional[list[tuple[float, float, float]]] = None
    jamming_zones: list[JammingZoneSpec] = field(default_factory=list)
    spoofing_zones: list[SpoofingZoneSpec] = field(default_factory=list)
    mavlink_enabled: bool = True
    crypto_enabled: bool = False
    crypto_algorithm: str = "hmac_sha256"
    llm_assistance_enabled: bool = False
    comm_model: str = "v2v"  # "v2v" | "legacy"
    path_algorithm: str = "astar"
    formation_type: str = "communication_aware"
    max_steps: int = 800
    dt: float = 0.5
    success_radius: float = 3.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "num_agents": self.num_agents,
            "destination": list(self.destination),
            "agent_init_positions": (
                [list(p) for p in self.agent_init_positions]
                if self.agent_init_positions
                else None
            ),
            "jamming_zones": [j.__dict__ for j in self.jamming_zones],
            "spoofing_zones": [s.__dict__ for s in self.spoofing_zones],
            "mavlink_enabled": self.mavlink_enabled,
            "crypto_enabled": self.crypto_enabled,
            "crypto_algorithm": self.crypto_algorithm,
            "llm_assistance_enabled": self.llm_assistance_enabled,
            "comm_model": self.comm_model,
            "path_algorithm": self.path_algorithm,
            "formation_type": self.formation_type,
            "max_steps": self.max_steps,
            "dt": self.dt,
            "success_radius": self.success_radius,
        }


# --------------------------------------------------------------------------
# Preset builders used by experiments.py
# --------------------------------------------------------------------------


def _default_starts(num_agents: int) -> list[tuple[float, float, float]]:
    """Line-formation start 40 m south of origin."""
    spacing = 6.0
    total_width = spacing * (num_agents - 1)
    x0 = -total_width / 2.0
    return [(x0 + i * spacing, -40.0, 2.0) for i in range(num_agents)]


def baseline_scenario(seed: int = 0, name: str = "baseline") -> Scenario:
    """Empty map, no attacks, no LLM."""
    return Scenario(
        name=name,
        seed=seed,
        agent_init_positions=_default_starts(5),
    )


def jamming_scenario(
    jam_type: str = "low_jam",
    seed: int = 0,
    llm: bool = False,
) -> Scenario:
    return Scenario(
        name=f"jam_{jam_type}_llm={int(llm)}",
        seed=seed,
        agent_init_positions=_default_starts(5),
        jamming_zones=[
            JammingZoneSpec(
                center=(10.0, 40.0, 10.0), radius=15.0, obstacle_type=jam_type
            ),
        ],
        llm_assistance_enabled=llm,
    )


def spoofing_scenario(
    spoof_type: str = "phantom",
    crypto: bool = False,
    crypto_algorithm: str = "hmac_sha256",
    seed: int = 0,
    llm: bool = False,
) -> Scenario:
    return Scenario(
        name=f"spoof_{spoof_type}_crypto={int(crypto)}_llm={int(llm)}",
        seed=seed,
        agent_init_positions=_default_starts(5),
        spoofing_zones=[
            SpoofingZoneSpec(
                center=(10.0, 40.0, 10.0), radius=25.0, spoof_type=spoof_type
            ),
        ],
        crypto_enabled=crypto,
        crypto_algorithm=crypto_algorithm,
        llm_assistance_enabled=llm,
    )


def combined_scenario(
    jam_type: str = "high_jam",
    spoof_type: str = "phantom",
    crypto: bool = True,
    crypto_algorithm: str = "hmac_sha256",
    llm: bool = True,
    seed: int = 0,
) -> Scenario:
    return Scenario(
        name=f"combo_{jam_type}_{spoof_type}_c{int(crypto)}_l{int(llm)}",
        seed=seed,
        agent_init_positions=_default_starts(5),
        jamming_zones=[
            JammingZoneSpec(
                center=(5.0, 30.0, 8.0), radius=15.0, obstacle_type=jam_type
            ),
        ],
        spoofing_zones=[
            SpoofingZoneSpec(
                center=(15.0, 60.0, 12.0), radius=25.0, spoof_type=spoof_type
            ),
        ],
        crypto_enabled=crypto,
        crypto_algorithm=crypto_algorithm,
        llm_assistance_enabled=llm,
    )


# --------------------------------------------------------------------------
# Educational presets (freshman-friendly)
# --------------------------------------------------------------------------

_EDUCATION_PRESETS: dict[str, dict] = {
    "intro_baseline": {
        "title": "Baseline Flight",
        "summary": "No attacks. Learn formations and path planning in 3D space.",
        "jamming_zones": [],
        "spoofing_zones": [],
        "crypto_enabled": False,
        "llm_assistance_enabled": False,
    },
    "intro_jamming": {
        "title": "Jamming Demo",
        "summary": "Single low-power jamming zone to study communication degradation.",
        "jamming_zones": [
            JammingZoneSpec(
                center=(10.0, 40.0, 10.0), radius=15.0, obstacle_type="low_jam"
            ),
        ],
        "spoofing_zones": [],
        "crypto_enabled": False,
        "llm_assistance_enabled": True,
    },
    "intro_spoofing": {
        "title": "Spoofing Demo",
        "summary": "Single spoofing zone to visualize phantom/falsified traffic effects.",
        "jamming_zones": [],
        "spoofing_zones": [
            SpoofingZoneSpec(
                center=(10.0, 40.0, 10.0), radius=25.0, spoof_type="phantom"
            ),
        ],
        "crypto_enabled": True,
        "llm_assistance_enabled": False,
    },
    "intro_combined": {
        "title": "Combined Attack Demo",
        "summary": "Jamming plus spoofing; shows why layered defenses matter.",
        "jamming_zones": [
            JammingZoneSpec(
                center=(5.0, 30.0, 8.0), radius=15.0, obstacle_type="high_jam"
            ),
        ],
        "spoofing_zones": [
            SpoofingZoneSpec(
                center=(15.0, 60.0, 12.0),
                radius=25.0,
                spoof_type="position_falsification",
            ),
        ],
        "crypto_enabled": True,
        "llm_assistance_enabled": True,
    },
}


def get_education_presets() -> dict[str, dict]:
    """Return preset metadata for GUI/API selection."""
    out: dict[str, dict] = {}
    for key, preset in _EDUCATION_PRESETS.items():
        out[key] = {
            "title": preset["title"],
            "summary": preset["summary"],
            "jamming_zone_count": len(preset["jamming_zones"]),
            "spoofing_zone_count": len(preset["spoofing_zones"]),
            "crypto_enabled": preset["crypto_enabled"],
            "llm_assistance_enabled": preset["llm_assistance_enabled"],
        }
    return out


def build_education_scenario(preset: str, seed: int = 0) -> Scenario:
    """Build a ready-to-run Scenario from an educational preset key."""
    if preset not in _EDUCATION_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. Available: {sorted(_EDUCATION_PRESETS)}"
        )

    p = _EDUCATION_PRESETS[preset]
    return Scenario(
        name=f"{preset}_seed{seed}",
        seed=seed,
        agent_init_positions=_default_starts(5),
        jamming_zones=list(p["jamming_zones"]),
        spoofing_zones=list(p["spoofing_zones"]),
        crypto_enabled=bool(p["crypto_enabled"]),
        llm_assistance_enabled=bool(p["llm_assistance_enabled"]),
        path_algorithm="astar",
        formation_type="communication_aware",
    )
