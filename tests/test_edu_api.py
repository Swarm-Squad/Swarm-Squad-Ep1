from __future__ import annotations

import pytest

from swarm_squad_ep1.client import SwarmSquadClient
from swarm_squad_ep1.research.scenarios import build_education_scenario


def test_list_presets_contains_expected_keys():
    presets = SwarmSquadClient.list_presets()
    assert {
        "intro_baseline",
        "intro_jamming",
        "intro_spoofing",
        "intro_combined",
    } <= set(presets)


def test_build_preset_scenario_has_expected_toggles():
    scenario = SwarmSquadClient.build_preset_scenario("intro_combined", seed=3)
    assert scenario.crypto_enabled is True
    assert scenario.llm_assistance_enabled is True
    assert len(scenario.jamming_zones) == 1
    assert len(scenario.spoofing_zones) == 1


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        build_education_scenario("does_not_exist", seed=0)


def test_headless_runner_smoke():
    scenario = SwarmSquadClient.build_preset_scenario("intro_jamming", seed=9)
    scenario.max_steps = 15  # keep unit test fast
    scenario.llm_assistance_enabled = False
    result = SwarmSquadClient.run_headless_scenario(
        scenario, keep_trace=False, verbose=False
    )
    assert result.total_steps > 0
    assert 0.0 <= result.avg_comm_quality <= 1.0
