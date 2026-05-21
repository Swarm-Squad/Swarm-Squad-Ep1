"""End-to-end smoke for the research harness (3 short scenarios, no LLM)."""
from __future__ import annotations

import pytest

from swarm_squad_ep1.research.runner import run_scenario
from swarm_squad_ep1.research.scenarios import (
    baseline_scenario,
    combined_scenario,
    jamming_scenario,
    spoofing_scenario,
)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: baseline_scenario(seed=1, name="t_baseline"),
        lambda: jamming_scenario(jam_type="high_jam", seed=2, llm=False),
        lambda: spoofing_scenario(spoof_type="phantom", crypto=True, seed=3, llm=False),
        lambda: combined_scenario(seed=4, llm=False, crypto=True),
    ],
)
def test_scenario_runs_without_error(factory):
    sc = factory()
    sc.max_steps = 40  # keep unit test fast
    sc.llm_assistance_enabled = False  # no external LLM in unit tests
    res = run_scenario(sc, keep_trace=False, verbose=False)

    assert res.total_steps > 0
    assert 0.0 <= res.final_Jn <= 1.0
    assert 0.0 <= res.avg_Jn <= 1.0
    assert 0.0 <= res.avg_comm_quality <= 1.0

    for v in (res.detection_rate, res.false_positive_rate, res.precision):
        assert 0.0 <= v <= 1.0
