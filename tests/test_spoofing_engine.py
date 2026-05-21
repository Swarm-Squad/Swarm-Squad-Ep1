from __future__ import annotations

import time

from swarm_squad_ep1.algo.mavlink import MAVLinkMessage, MessageType
from swarm_squad_ep1.algo.spoofing import SpoofingEngine, SpoofingZone, SpoofType


def _position_msg(agent_id: str, position: list[float]) -> MAVLinkMessage:
    return MAVLinkMessage(
        msg_type=MessageType.GLOBAL_POSITION_INT,
        sender_id=agent_id,
        sequence=0,
        timestamp=time.time(),
        payload={
            "position": list(position),
            "velocity": [0.0, 0.0, 0.0],
            "heading": 0.0,
        },
        is_spoofed=False,
    )


def test_phantom_spoofing_injects_messages():
    engine = SpoofingEngine()
    zone = SpoofingZone(
        id="zone_phantom",
        center=[0.0, 0.0, 0.0],
        radius=20.0,
        spoof_type=SpoofType.PHANTOM,
        phantom_count=3,
    )

    out = engine.process([], [zone], {})
    assert len(out) == 3
    assert all(msg.is_spoofed for msg in out)
    assert all(msg.sender_id.startswith("phantom_zone_phantom_") for msg in out)


def test_position_falsification_marks_message_spoofed():
    engine = SpoofingEngine()
    zone = SpoofingZone(
        id="zone_falsify",
        center=[0.0, 0.0, 0.0],
        radius=30.0,
        spoof_type=SpoofType.POSITION_FALSIFICATION,
        falsification_magnitude=6.0,
    )
    true_pos = [2.0, 1.0, 0.0]
    msg = _position_msg("agent1", true_pos)

    out = engine.process([msg], [zone], {"agent1": true_pos})
    assert len(out) == 1
    spoofed = out[0]
    assert spoofed.is_spoofed is True
    assert spoofed.payload["position"] != true_pos


def test_coordinate_attack_applies_uniform_offset():
    engine = SpoofingEngine()
    vec = [10.0, -5.0, 2.0]
    zone = SpoofingZone(
        id="zone_coord",
        center=[0.0, 0.0, 0.0],
        radius=40.0,
        spoof_type=SpoofType.COORDINATE,
        coordinate_vector=vec,
    )
    true_pos = [3.0, 4.0, 1.0]
    msg = _position_msg("agent2", true_pos)

    out = engine.process([msg], [zone], {"agent2": true_pos})
    attacked = out[0]
    assert attacked.is_spoofed is True
    assert attacked.payload["position"] == [true_pos[i] + vec[i] for i in range(3)]
