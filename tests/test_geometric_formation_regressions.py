from __future__ import annotations

import numpy as np

from swarm_squad_ep1.algo.base import JammingZone, ObstacleType
from swarm_squad_ep1.algo.controller import UnifiedController
from swarm_squad_ep1.simulation.agents import AgentState


def _make_agents() -> dict[str, AgentState]:
    positions = [
        [-2.0, 0.0, 5.0],
        [0.0, 0.0, 5.0],
        [2.0, 0.0, 5.0],
        [4.0, 0.0, 5.0],
        [-4.0, 0.0, 5.0],
    ]
    return {
        f"agent{i}": AgentState(agent_id=f"agent{i}", position=pos.copy())
        for i, pos in enumerate(positions, start=1)
    }


def _centroid_step_toward_destination(
    formation: str, destination: tuple[float, float, float]
) -> float:
    controller = UnifiedController(formation_type=formation, path_algorithm="direct")
    agents = _make_agents()
    commands = controller.compute_commands(agents, destination, [], dt=0.1)

    old_center = np.mean(
        [np.array(agent.position, dtype=float) for agent in agents.values()], axis=0
    )
    new_center = np.mean(
        [np.array(commands[aid].target_position, dtype=float) for aid in agents], axis=0
    )
    center_motion = new_center - old_center
    to_dest = np.array(destination, dtype=float) - old_center
    return float(np.dot(center_motion[:2], to_dest[:2]))


def test_v_wedge_column_move_toward_destination():
    destination = (0.0, 100.0, 5.0)
    for formation in ("v_formation", "wedge", "column"):
        projection = _centroid_step_toward_destination(formation, destination)
        assert projection > 0.0, (
            f"{formation} projected opposite destination direction (dot={projection:.3f})"
        )


def test_geometric_formation_jamming_changes_motion():
    destination = (0.0, 100.0, 5.0)

    baseline_controller = UnifiedController(
        formation_type="wedge", path_algorithm="direct"
    )
    baseline_agents = _make_agents()
    baseline = baseline_controller.compute_commands(
        baseline_agents, destination, [], dt=0.1
    )

    jammed_controller = UnifiedController(
        formation_type="wedge", path_algorithm="direct"
    )
    jammed_agents = _make_agents()
    high_jam = JammingZone(
        id="jam-zone",
        center=[0.0, 0.0, 5.0],
        radius=12.0,
        obstacle_type=ObstacleType.HIGH_JAM,
    )
    jammed = jammed_controller.compute_commands(
        jammed_agents, destination, [high_jam], dt=0.1
    )

    per_agent_delta = []
    baseline_speed = []
    jammed_speed = []
    for agent_id in baseline:
        base_target = np.array(baseline[agent_id].target_position, dtype=float)
        jam_target = np.array(jammed[agent_id].target_position, dtype=float)
        per_agent_delta.append(float(np.linalg.norm(jam_target - base_target)))
        baseline_speed.append(float(np.linalg.norm(baseline[agent_id].velocity)))
        jammed_speed.append(float(np.linalg.norm(jammed[agent_id].velocity)))

    assert max(per_agent_delta) > 0.05, "jamming produced no meaningful motion change"
    assert np.mean(jammed_speed) < np.mean(baseline_speed), (
        "jamming should reduce average geometric-formation movement speed"
    )


def test_comm_quality_is_computed_for_all_models_and_formations():
    destination = (0.0, 100.0, 5.0)
    for formation in ("communication_aware", "wedge"):
        for path_algorithm in ("direct", "astar"):
            for use_v2v in (True, False):
                controller = UnifiedController(
                    formation_type=formation, path_algorithm=path_algorithm
                )
                controller.use_v2v_channel = use_v2v
                agents = _make_agents()
                controller.compute_commands(agents, destination, [], dt=0.1)
                qualities = controller.get_all_agent_comm_quality()

                assert set(qualities.keys()) == set(agents.keys())
                for value in qualities.values():
                    assert np.isfinite(value)
                    assert 0.0 <= float(value) <= 1.5
