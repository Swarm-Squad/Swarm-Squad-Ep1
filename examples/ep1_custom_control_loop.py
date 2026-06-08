"""Phased full-control script demo for Swarm Squad Ep1."""

from __future__ import annotations

import math
import time
from typing import Any

from swarm_squad_ep1.client import SwarmSquadClient

CUSTOM_PATH_NAME = "midpoint_demo"
CUSTOM_CRYPTO_NAME = "xor_hmac_demo"
MISSION_DESTINATION = (35.0, 150.0, 30.0)


def _phase(title: str) -> None:
    print(f"\n=== {title} ===")


def _normalize2d(x: float, y: float) -> tuple[float, float]:
    mag = math.hypot(x, y)
    if mag <= 1e-9:
        return 0.0, 0.0
    return x / mag, y / mag


def _active_avoidance_zones(state: dict[str, Any]) -> list[dict[str, float]]:
    zones: list[dict[str, float]] = []

    for zone in state.get("jamming_zones", []):
        if not zone.get("active", True):
            continue
        center = zone.get("center", [0.0, 0.0, 0.0])
        radius = float(zone.get("radius", 0.0))
        zones.append(
            {
                "x": float(center[0]),
                "y": float(center[1]),
                "radius": radius,
                "weight": 2.6,
            }
        )

    for zone in state.get("spoofing_zones", []):
        if not zone.get("active", True):
            continue
        center = zone.get("center", [0.0, 0.0, 0.0])
        radius = float(zone.get("radius", 0.0))
        zones.append(
            {
                "x": float(center[0]),
                "y": float(center[1]),
                "radius": radius,
                "weight": 2.0,
            }
        )

    return zones


def scripted_swarm_policy(
    state: dict[str, Any], _step_idx: int
) -> list[dict[str, float | str]]:
    """Steer toward mission end while repelling around attack zones."""
    commands: list[dict[str, float | str]] = []
    zones = _active_avoidance_zones(state)

    for agent_id, agent in state.get("agents", {}).items():
        if str(agent_id).startswith("phantom_"):
            continue

        pos = agent.get("position", [0.0, 0.0, 0.0])
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        gx, gy, _ = MISSION_DESTINATION

        # Attractive direction toward destination.
        to_goal_x, to_goal_y = gx - px, gy - py
        goal_dir_x, goal_dir_y = _normalize2d(to_goal_x, to_goal_y)

        # Repulsive/tangential bias around active jamming + spoofing zones.
        avoid_x = 0.0
        avoid_y = 0.0
        for zone in zones:
            zx, zy, radius, weight = (
                zone["x"],
                zone["y"],
                zone["radius"],
                zone["weight"],
            )
            dx, dy = px - zx, py - zy
            dist = math.hypot(dx, dy)
            influence_radius = radius + 18.0
            if dist >= influence_radius:
                continue

            nx, ny = _normalize2d(dx, dy)
            # If exactly at center, push opposite goal direction.
            if nx == 0.0 and ny == 0.0:
                nx, ny = -goal_dir_x, -goal_dir_y

            proximity = (influence_radius - dist) / max(influence_radius, 1e-6)
            avoid_x += nx * proximity * weight
            avoid_y += ny * proximity * weight

            # Add a tangential component when inside/near zone for realistic sidestep.
            if dist < radius + 3.0:
                tx, ty = -ny, nx
                avoid_x += tx * 1.4
                avoid_y += ty * 1.4

        move_x = goal_dir_x + (avoid_x * 1.8)
        move_y = goal_dir_y + (avoid_y * 1.8)
        dir_x, dir_y = _normalize2d(move_x, move_y)
        if dir_x == 0.0 and dir_y == 0.0:
            dir_x, dir_y = goal_dir_x, goal_dir_y

        step = min(1.0, math.hypot(to_goal_x, to_goal_y))
        next_x = px + dir_x * step
        next_y = py + dir_y * step
        next_z = pz + max(min(MISSION_DESTINATION[2] - pz, 0.5), -0.5)

        commands.append(
            {
                "agent": agent_id,
                "x": next_x,
                "y": next_y,
                "z": next_z,
            }
        )
    return commands


def _swarm_center_distance_to_goal(
    state: dict[str, Any], destination: tuple[float, float, float]
) -> float:
    real_positions = []
    for agent_id, agent in state.get("agents", {}).items():
        if str(agent_id).startswith("phantom_"):
            continue
        pos = agent.get("position", [0.0, 0.0, 0.0])
        real_positions.append((float(pos[0]), float(pos[1]), float(pos[2])))

    if not real_positions:
        return float("inf")

    cx = sum(p[0] for p in real_positions) / len(real_positions)
    cy = sum(p[1] for p in real_positions) / len(real_positions)
    cz = sum(p[2] for p in real_positions) / len(real_positions)
    dx = destination[0] - cx
    dy = destination[1] - cy
    dz = destination[2] - cz
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _wait_for_destination(
    client: SwarmSquadClient,
    destination: tuple[float, float, float],
    *,
    timeout_s: float = 120.0,
    poll_s: float = 0.5,
) -> tuple[bool, float, dict[str, Any]]:
    deadline = time.time() + timeout_s
    last_distance = float("inf")
    latest_results: dict[str, Any] = {}

    while time.time() < deadline:
        state = client.simulation_state()
        last_distance = _swarm_center_distance_to_goal(state, destination)
        latest_results = client.simulation_results()

        if latest_results.get("destination_reached"):
            return True, last_distance, latest_results
        if not state.get("running", False):
            break
        time.sleep(poll_s)

    return (
        bool(latest_results.get("destination_reached")),
        last_distance,
        latest_results,
    )


def main() -> None:
    client = SwarmSquadClient()
    created_agent_id: str | None = None
    jamming_zone_id: str | None = None
    spoofing_zone_id: str | None = None

    try:
        # 1) State reset + config discovery.
        _phase("1) Reset + config discovery")
        client.stop_simulation()
        client.reset_simulation()
        config = client.simulation_config()
        print(
            "Available:",
            {
                "formations": len(config.get("formations", [])),
                "path_algorithms": len(config.get("path_algorithms", [])),
                "crypto_algorithms": len(config.get("crypto_algorithms", [])),
            },
        )

        # 2) Custom path + custom crypto registration.
        _phase("2) Register custom path + custom crypto")
        client.register_custom_algorithm(
            name=CUSTOM_PATH_NAME,
            import_path="examples.custom_algorithms.midpoint_path:midpoint_path",
            description="Midpoint path demo",
            replace=True,
        )
        client.register_custom_crypto_algorithm(
            name=CUSTOM_CRYPTO_NAME,
            sign_import_path="examples.custom_algorithms.xor_hmac_crypto:xor_hmac_sign",
            verify_import_path="examples.custom_algorithms.xor_hmac_crypto:xor_hmac_verify",
            description="XOR + HMAC demo crypto",
            replace=True,
        )

        # 3) Agent add/remove.
        _phase("3) Agent add/remove")
        created = client.add_agent(x=25.0, y=18.0, z=5.0)
        created_agent_id = created["agent"]["agent_id"]
        print("Added agent:", created_agent_id)
        client.update_agent(created_agent_id, position=(26.5, 18.5, 5.0))
        client.remove_agent(created_agent_id)
        created = client.add_agent(x=24.0, y=20.0, z=5.0)
        created_agent_id = created["agent"]["agent_id"]
        print("Re-added agent for remainder of demo:", created_agent_id)

        # 4) Jamming/spoofing add/list/update/delete.
        _phase("4) Jamming + spoofing zone controls")
        jam = client.add_jamming_zone(
            center=(36.0, 38.0, 8.0),
            radius=9.0,
            jam_type="low_jam",
            intensity=0.7,
        )
        jamming_zone_id = jam["zone"]["id"]
        client.update_jamming_zone(
            jamming_zone_id,
            radius=10.5,
            obstacle_type="high_jam",
            intensity=0.9,
        )
        print("Jamming zones:", client.list_jamming_zones()["count"])

        spoof = client.add_spoofing_zone(
            center=(46.0, 40.0, 8.0),
            radius=11.0,
            spoof_type="phantom",
            phantom_count=2,
        )
        spoofing_zone_id = spoof["zone"]["id"]
        client.update_spoofing_zone(
            spoofing_zone_id,
            spoof_type="coordinate",
            coordinate_vector=(7.0, -3.0, 0.0),
        )
        print("Spoofing zones:", client.list_spoofing_zones()["count"])
        # Demonstrate delete/recreate path for zone lifecycle.
        client.delete_spoofing_zone(spoofing_zone_id)
        spoofing_zone_id = client.add_spoofing_zone(
            center=(48.0, 44.0, 8.0),
            radius=9.0,
            spoof_type="position_falsification",
            falsification_magnitude=5.0,
        )["zone"]["id"]

        # 5) Formation/path switching.
        _phase("5) Formation + path switching")
        client.set_algorithm(formation="v_formation", path_algorithm="astar")
        client.set_algorithm(formation="line", path_algorithm=CUSTOM_PATH_NAME)
        client.set_algorithm(
            formation="communication_aware", path_algorithm=CUSTOM_PATH_NAME
        )

        # 6) Comm model toggling.
        _phase("6) Communication model toggling")
        client.set_comm_model("legacy")
        client.set_comm_model("v2v_channel")

        # 7) Crypto off/on + built-in/custom switching.
        _phase("7) Crypto off/on + algorithm switching")
        client.set_crypto_auth(False, algorithm="hmac_sha256")
        client.set_crypto_auth(True, algorithm="hmac_sha256")
        client.set_crypto_auth(True, algorithm=CUSTOM_CRYPTO_NAME)

        # 8) LLM assistance toggling.
        _phase("8) LLM assistance toggling")
        client.set_llm_assistance(False)
        client.set_llm_assistance(True)

        # 9) Script-driven loop execution with live GUI visualization.
        _phase("9) Start sim + run script loop")
        # Remove the demo-only extra agent before mission run for stable convergence.
        if created_agent_id:
            client.remove_agent(created_agent_id)
            print("Removed extra demo agent before mission sprint:", created_agent_id)
            created_agent_id = None
        client.start_simulation(
            formation="v_formation",
            path_algorithm=CUSTOM_PATH_NAME,
            crypto_auth=True,
            crypto_algorithm=CUSTOM_CRYPTO_NAME,
            destination=MISSION_DESTINATION,
            default_obstacle_type="low_jam",
        )
        trace = client.run_script_control_loop(
            scripted_swarm_policy,
            steps=500,
            step_interval_s=0.05,
            auto_simulate_step=True,
        )
        print(f"Control loop steps: {len(trace)}")
        # Clear any residual manual targets before collecting final state.
        client.clear_all_llm_targets()
        client.set_llm_assistance(False)
        reached, remaining_dist, run_results = _wait_for_destination(
            client, MISSION_DESTINATION, timeout_s=30.0
        )
        if not reached:
            print("Primary sprint timed out; extending zone-aware sprint...")
            extra_trace = client.run_script_control_loop(
                scripted_swarm_policy,
                steps=500,
                step_interval_s=0.05,
                auto_simulate_step=True,
            )
            print(f"Extended sprint steps: {len(extra_trace)}")
            client.clear_all_llm_targets()
            reached, remaining_dist, run_results = _wait_for_destination(
                client, MISSION_DESTINATION, timeout_s=30.0
            )
        print(
            {
                "destination_reached": reached,
                "remaining_distance": round(remaining_dist, 2),
                "steps": run_results.get("steps"),
            }
        )

        # 10) Metrics + cleanup.
        _phase("10) Collect metrics")
        attack = client.attack_metrics()
        protocol = client.protocol_stats()
        summary = client.simulation_results()
        print(
            {
                "crypto_algorithm": attack.get("crypto_algorithm"),
                "detection_rate": attack.get("detection_rate"),
                "messages_sent": protocol.get("mavlink", {}).get("messages_sent"),
                "destination_reached": summary.get("destination_reached"),
                "duration_seconds": summary.get("duration_seconds"),
            }
        )
    finally:
        # Keep teardown idempotent for reruns.
        try:
            if client.simulation_state().get("running", False):
                client.stop_simulation()
        except Exception:
            pass
        if spoofing_zone_id:
            try:
                client.delete_spoofing_zone(spoofing_zone_id)
            except Exception:
                pass
        if jamming_zone_id:
            try:
                client.delete_jamming_zone(jamming_zone_id)
            except Exception:
                pass
        if created_agent_id:
            try:
                client.remove_agent(created_agent_id)
            except Exception:
                pass
        try:
            client.remove_custom_algorithm(CUSTOM_PATH_NAME)
        except Exception:
            pass
        try:
            client.remove_custom_crypto_algorithm(CUSTOM_CRYPTO_NAME)
        except Exception:
            pass


if __name__ == "__main__":
    main()
