"""
Tests for the headless experiment harness and the controller fixes it relies on.

Run with: uv run pytest src/swarm_squad_ep1/tests/test_experiments.py
"""

import contextlib
import dataclasses
import os

import numpy as np
import pytest

import swarm_squad_ep1.config as config
from swarm_squad_ep1.controllers.controller_factory import ControllerType
from swarm_squad_ep1.experiments import compass, linkmodel, metrics, power
from swarm_squad_ep1.experiments.runner import run_one
from swarm_squad_ep1.experiments.scenario import Scenario
from swarm_squad_ep1.experiments.supervisors import (
    INSTRUCTION_PATTERN,
    ExperimentFactory,
    RuleSupervisor,
    Supervisor,
)
from swarm_squad_ep1.models.swarm_state import SwarmState


@pytest.fixture(autouse=True)
def restore_mode():
    mode = config.OBSTACLE_MODE
    yield
    config.OBSTACLE_MODE = mode


def quiet():
    return contextlib.redirect_stdout(open(os.devnull, "w"))


def make_state(positions, obstacles=(), mode=config.ObstacleMode.HARD):
    config.OBSTACLE_MODE = mode
    state = SwarmState()
    state.swarm_position = np.array(positions, dtype=float)
    state.initial_positions = state.swarm_position.copy()
    n = len(state.swarm_position)
    state.swarm_size = n
    state.swarm_control_ui = np.zeros((n, 2))
    state.agent_status = np.ones(n, dtype=bool)
    state.jamming_affected = np.zeros(n, dtype=bool)
    state.jamming_depth = np.zeros(n)
    state.line_colors = np.zeros((n, n, 3))
    state.communication_qualities_matrix = np.zeros((n, n))
    state.distances_matrix = np.zeros((n, n))
    state.neighbor_agent_matrix = np.zeros((n, n))
    state.obstacles = [tuple(o) for o in obstacles]
    state.obstacle_mode = mode
    return state


def test_link_model_matches_swarm_state():
    rng = np.random.default_rng(3)
    positions = rng.uniform(-40, 40, size=(7, 2))
    jammer = (0.0, 0.0, 15.0)
    state = make_state(positions, [jammer], config.ObstacleMode.LOW_POWER_JAMMING)
    with quiet():
        state.update_matrices()
        state.update_performance_metrics()
    q, a = linkmodel.link_quality(positions, [jammer])
    assert np.allclose(q, state.communication_qualities_matrix)
    assert np.allclose(a, state.neighbor_agent_matrix)
    assert linkmodel.jn(q, a) == pytest.approx(state.Jn[-1], abs=1e-4)
    assert np.allclose(
        linkmodel.penetration_depth(positions, [jammer]) * state.jamming_affected,
        state.jamming_depth,
    )


def test_jamming_depth_follows_current_position():
    jammer = (0.0, 0.0, 10.0)
    state = make_state(
        [[12.0, 0.0], [60.0, 0.0]], [jammer], config.ObstacleMode.LOW_POWER_JAMMING
    )
    with quiet():
        state.update_matrices()
        deep = state.jamming_depth[0]
        state.swarm_position[0] = [18.0, 0.0]
        state.update_matrices()
    assert deep == pytest.approx(0.8)
    assert state.jamming_depth[0] == pytest.approx(0.2)


def test_bearings_match_parser_and_round_trip():
    for name in compass.BEARING_NAMES:
        assert compass.quantize(compass.unit_vector(name)) == name

    state = make_state(config.INITIAL_SWARM_POSITIONS)
    with quiet():
        factory = ExperimentFactory(state, lambda s: Supervisor(s))
    sup = factory.controllers[ControllerType.LLM]
    commands = [
        (i % 7, 5 + i % 6, name) for i, name in enumerate(compass.BEARING_NAMES)
    ]
    for agent, distance, name in commands:
        sup.current_feedback = compass.command_text([(agent, distance, name)])
        sup._parse_llm_feedback_to_control_inputs()
        assert sup.llm_affected_agents == [agent]
        expected = 0.1 * distance * compass.unit_vector(name)
        assert np.allclose(sup.llm_control_inputs[agent], expected, atol=1e-3)


def test_parser_reads_multiword_bearings():
    text = (
        "Agent-1: Move 8 units north by northwest to escape jamming. "
        "Agent-2: Reposition 10 units west by southwest to exit interference field."
    )
    found = INSTRUCTION_PATTERN.findall(text)
    assert [f[2] for f in found] == ["north by northwest", "west by southwest"]


def test_command_expires_after_lifetime():
    state = make_state(config.INITIAL_SWARM_POSITIONS)
    with quiet():
        factory = ExperimentFactory(state, lambda s: Supervisor(s))
        sup = factory.controllers[ControllerType.LLM]
        sup.enabled = True
        sup.feedback_queue.put("Agent-2: Move 5 units north.")
        sup._clock = lambda: 0.0  # never request
        factory.step()
        assert sup.llm_affected_agents == [2]
        for _ in range(sup.llm_control_lifetime):
            factory.step()
    assert sup.llm_affected_agents == []


def test_c2_gate_drops_commands_and_freezes_telemetry():
    jammer = (0.0, 0.0, 10.0)
    positions = [[12.0, 0.0], [40.0, 0.0], [40.0, 20.0]]
    state = make_state(positions, [jammer], config.ObstacleMode.LOW_POWER_JAMMING)
    with quiet():
        factory = ExperimentFactory(state, lambda s: Supervisor(s, c2_gate=True))
        sup = factory.controllers[ControllerType.LLM]
        sup.gamma = np.ones(3, dtype=bool)
        sup._update_snapshot()  # telemetry before agent 0 is known to be jammed
        state.update_matrices()
    sup.gamma = sup._reachability()
    assert sup.gamma.tolist() == [False, True, True]

    state.swarm_position[0] = [11.0, 0.0]
    view = sup.perceived_state()
    assert np.allclose(view.swarm_position[0], positions[0])
    assert not view.jamming_affected[0]

    sup.current_feedback = "Agent-0: Move 5 units east. Agent-1: Move 5 units north."
    sup.decisions.append({"kind": "test"})
    sup._parse_llm_feedback_to_control_inputs()
    assert sup.llm_affected_agents == [1]
    assert sup.decisions[-1]["dropped_c2"] == [0]


def test_rule_applies_evasion_hint():
    jammer = (0.0, 0.0, 10.0)
    positions = [[15.0, 0.0], [45.0, 0.0], [45.0, 25.0]]
    state = make_state(positions, [jammer], config.ObstacleMode.LOW_POWER_JAMMING)
    with quiet():
        factory = ExperimentFactory(state, lambda s: RuleSupervisor(s))
        state.update_matrices()
        sup = factory.controllers[ControllerType.LLM]
        commands = sup.decide(state)
    assert commands[0][0] == 0
    assert commands[0][2] == "east"


def test_reactive_run_reproduces_gui_run():
    """Reactive-only, unjittered Obstacles I took 489 steps in the GUI."""
    scenario = dataclasses.replace(
        Scenario.load("obstacles_1"), position_jitter=0.0, obstacle_jitter=0.0
    )
    with quiet():
        summary = run_one(scenario, "none", 0, _tmp_dir("nominal"))
    assert summary["success"]
    assert summary["steps"] == 489
    assert summary["jn_final"] == pytest.approx(0.9723, abs=1e-4)


def test_seeded_runs_are_deterministic():
    scenario = Scenario.load("jamming")
    with quiet():
        a = run_one(scenario, "rule", 4, _tmp_dir("det_a"))
        b = run_one(scenario, "rule", 4, _tmp_dir("det_b"))
    assert a["steps"] == b["steps"]
    assert a["jn_min_mission"] == b["jn_min_mission"]


def test_threshold_stats():
    jn = [0.95] * 5 + [0.9] * 3 + [0.95] * 30
    out = metrics.threshold_stats(jn, pt=0.94, hold=20)
    assert out["steps_below_pt"] == 3
    assert out["first_below"] == 5
    assert out["recovery_steps"] == 3


def test_integrate_energy():
    t = np.linspace(0.0, 10.0, 101)
    samples = np.column_stack([t, np.full_like(t, 100.0)])
    assert power.integrate_energy(samples, 2.0, 4.0) == pytest.approx(200.0)
    assert power.integrate_energy(samples, 2.0, 4.0, idle_w=40.0) == pytest.approx(
        120.0
    )


def _tmp_dir(name):
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp(prefix=f"icc_{name}_"))
