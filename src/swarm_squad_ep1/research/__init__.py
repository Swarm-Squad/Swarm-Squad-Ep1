"""
Research / evaluation harness for Swarm Squad Ep1.

Offline A/B runner that executes scenarios headlessly and records the
metrics needed to demonstrate:

1. Combined jamming + spoofing is more damaging than either alone
2. LLM-assisted coordination always improves under dual attack
3. Path planning algorithm trade-offs under attack
4. Cryptographic authentication method comparison
"""

from swarm_squad_ep1.research.experiments import EXPERIMENTS, run_experiment
from swarm_squad_ep1.research.runner import Result, run_scenario
from swarm_squad_ep1.research.scenarios import (
    Scenario,
    build_education_scenario,
    get_education_presets,
)

__all__ = [
    "Scenario",
    "Result",
    "run_scenario",
    "EXPERIMENTS",
    "run_experiment",
    "get_education_presets",
    "build_education_scenario",
]
