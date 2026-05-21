from __future__ import annotations

import numpy as np

from swarm_squad_ep1.algo.base import JammingZone, ObstacleType
from swarm_squad_ep1.algo.jamming_response import JammingResponse


def _zone(
    zone_id: str, obstacle_type: ObstacleType, radius: float = 10.0
) -> JammingZone:
    return JammingZone(
        id=zone_id,
        center=[0.0, 0.0, 0.0],
        radius=radius,
        obstacle_type=obstacle_type,
    )


def test_inside_jamming_field_detection_and_safe_position_tracking():
    handler = JammingResponse()
    low_jam = _zone("low", ObstacleType.LOW_JAM)

    outside_pos = np.array([50.0, 0.0, 0.0])
    inside, effect, nearest = handler.is_inside_jamming_field(
        "agent1", outside_pos, [low_jam]
    )
    assert inside is False
    assert effect == 0.0
    assert nearest is None
    assert "agent1" in handler.last_safe_positions

    inside_pos = np.array([2.0, 0.0, 0.0])
    inside, effect, nearest = handler.is_inside_jamming_field(
        "agent1", inside_pos, [low_jam]
    )
    assert inside is True
    assert effect > 0.0
    assert nearest is not None
    assert nearest.id == "low"


def test_compute_response_for_physical_obstacle_pushes_agent_away():
    handler = JammingResponse()
    physical = _zone("physical", ObstacleType.PHYSICAL, radius=6.0)

    pos = np.array([1.0, 0.0, 0.0])
    vel = np.array([1.0, 0.0, 0.0])
    goal = np.array([20.0, 0.0, 0.0])

    response = handler.compute_response("agent1", pos, vel, goal, [physical])
    assert response.shape == (3,)
    # At x=1 with center at origin, repulsion should point to +x.
    assert response[0] > 0.0


def test_high_jam_trapped_triggers_emergency_escape_to_safe_position():
    handler = JammingResponse()
    high_jam = _zone("high", ObstacleType.HIGH_JAM, radius=8.0)

    pos = np.array([1.0, 0.0, 0.0])  # Deep inside zone -> trapped
    vel = np.array([1.0, 0.0, 0.0])
    goal = np.array([50.0, 0.0, 0.0])

    handler.last_safe_positions["agent9"] = np.array([20.0, 0.0, 0.0])

    response = handler.compute_response("agent9", pos, vel, goal, [high_jam])
    assert np.linalg.norm(response) > np.linalg.norm(vel)
    assert response[0] > 0.0


def test_proactive_avoidance_nonzero_near_active_zone():
    handler = JammingResponse()
    low_jam = _zone("low", ObstacleType.LOW_JAM, radius=10.0)
    position = np.array([5.0, 0.0, 0.0])

    avoidance = handler.compute_proactive_avoidance(position, [low_jam])
    assert avoidance.shape == (3,)
    assert np.linalg.norm(avoidance) > 0.0
