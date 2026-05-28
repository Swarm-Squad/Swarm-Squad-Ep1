"""Phased full-control script demo for Swarm Squad Ep1."""

from __future__ import annotations

from typing import Any

from swarm_squad_ep1.client import SwarmSquadClient

CUSTOM_PATH_NAME = "midpoint_demo"
CUSTOM_CRYPTO_NAME = "xor_hmac_demo"


def _phase(title: str) -> None:
    print(f"\n=== {title} ===")


def scripted_swarm_policy(
    state: dict[str, Any], step_idx: int
) -> list[dict[str, float | str]]:
    """Simple script-owned control policy for live GUI playback."""
    commands: list[dict[str, float | str]] = []
    lateral = 0.4 if step_idx % 2 == 0 else -0.4
    for idx, (agent_id, agent) in enumerate(state.get("agents", {}).items()):
        if str(agent_id).startswith("phantom_"):
            continue
        pos = agent.get("position", [0.0, 0.0, 0.0])
        commands.append(
            {
                "agent": agent_id,
                "x": float(pos[0]) + 0.8,
                "y": float(pos[1]) + (lateral if idx % 2 == 0 else -lateral),
                "z": float(pos[2]),
            }
        )
    return commands


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
        client.start_simulation(
            formation="communication_aware",
            path_algorithm=CUSTOM_PATH_NAME,
            crypto_auth=True,
            crypto_algorithm=CUSTOM_CRYPTO_NAME,
            destination=(90.0, 90.0, 5.0),
            default_obstacle_type="low_jam",
        )
        trace = client.run_script_control_loop(
            scripted_swarm_policy,
            steps=16,
            step_interval_s=0.1,
            auto_simulate_step=True,
        )
        print(f"Control loop steps: {len(trace)}")

        # 10) Metrics + cleanup.
        _phase("10) Collect metrics")
        attack = client.attack_metrics()
        protocol = client.protocol_stats()
        print(
            {
                "crypto_algorithm": attack.get("crypto_algorithm"),
                "detection_rate": attack.get("detection_rate"),
                "messages_sent": protocol.get("mavlink", {}).get("messages_sent"),
            }
        )
    finally:
        # Keep teardown idempotent for reruns.
        try:
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
