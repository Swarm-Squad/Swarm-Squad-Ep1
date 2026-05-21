from __future__ import annotations

import time
from dataclasses import dataclass

from swarm_squad_ep1.algo.mavlink import MAVLinkBus, MAVLinkMessage, MessageType


@dataclass
class _AgentStub:
    position: list[float]
    velocity: list[float]


def test_broadcast_sequence_and_queue_lifecycle():
    bus = MAVLinkBus(packet_loss_base=0.0)

    msg1 = bus.broadcast("agent1", [0.0, 0.0, 0.0])
    msg2 = bus.broadcast("agent1", [1.0, 0.0, 0.0])
    msg3 = bus.broadcast("agent2", [0.0, 1.0, 0.0])

    assert msg1.sequence == 0
    assert msg2.sequence == 1
    assert msg3.sequence == 0
    assert bus.stats.messages_sent == 3
    assert len(bus.get_messages()) == 3

    bus.clear_queue()
    assert bus.get_messages() == []


def test_packet_loss_and_perceived_state_with_ground_truth_fallback():
    bus = MAVLinkBus(packet_loss_base=0.0)
    bus.broadcast("agent1", [2.0, 0.0, 0.0], velocity=[0.1, 0.0, 0.0])
    bus.apply_packet_loss(comm_qualities={"agent1": 1.0})

    ground_truth = {
        "agent1": _AgentStub(position=[2.0, 0.0, 0.0], velocity=[0.1, 0.0, 0.0]),
        "agent2": _AgentStub(position=[9.0, 9.0, 0.0], velocity=[0.0, 0.0, 0.0]),
    }
    bus.build_perceived_state(ground_truth)

    perceived = bus.get_perceived_positions(["agent1", "agent2"])
    assert "agent1" in perceived
    assert "agent2" in perceived  # fallback from ground truth (no message for agent2)
    assert bus.stats.messages_received >= 1


def test_spoof_tracking_for_phantoms_and_falsification_offsets():
    bus = MAVLinkBus(packet_loss_base=0.0)

    phantom = MAVLinkMessage(
        msg_type=MessageType.GLOBAL_POSITION_INT,
        sender_id="phantom_zone1_1",
        sequence=0,
        timestamp=time.time(),
        payload={"position": [5.0, 5.0, 0.0], "velocity": [0.0, 0.0, 0.0]},
        is_spoofed=True,
    )
    bus.inject(phantom)

    falsified = MAVLinkMessage(
        msg_type=MessageType.GLOBAL_POSITION_INT,
        sender_id="agent1",
        sequence=0,
        timestamp=time.time(),
        payload={"position": [12.0, 8.0, 0.0], "velocity": [0.0, 0.0, 0.0]},
        is_spoofed=True,
    )
    bus.inject(falsified)

    ground_truth = {
        "agent1": _AgentStub(position=[10.0, 10.0, 0.0], velocity=[0, 0, 0])
    }
    bus.build_perceived_state(ground_truth)

    assert "phantom_zone1_1" in bus.get_phantom_ids()
    offsets = bus.get_falsification_offsets()
    assert "agent1" in offsets
    assert offsets["agent1"] == [2.0, -2.0, 0.0]
