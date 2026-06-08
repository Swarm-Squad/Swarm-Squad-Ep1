"""Starter script for the Python + GUI classroom assignment track.

Run this while `swarm-squad-ep1` is active in another terminal.
The GUI at http://localhost:5000 will reflect scenario changes live.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from swarm_squad_ep1.client import SwarmSquadClient

RESULTS_PATH = Path("assignment_starter_results.json")


def wait_until_done(client: SwarmSquadClient, timeout_s: float = 12.0) -> None:
    """Poll simulation state until stop/timeout, then force-stop once."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = client.simulation_state()
        if not state.get("running", False):
            return
        time.sleep(0.25)
    # Timeout: stop explicitly so the script can continue to next scenario.
    client.stop_simulation()


def run_scenario(client: SwarmSquadClient, scenario: dict) -> dict:
    """Apply one scenario and return a compact metrics summary."""
    client.reset_simulation()
    client.clear_jamming_zones()
    client.clear_spoofing_zones()

    client.set_algorithm(
        formation=scenario.get("formation", "communication_aware"),
        path_algorithm=scenario.get("path_algorithm", "astar"),
    )
    client.set_comm_model(scenario.get("comm_model", "v2v_channel"))
    client.set_crypto_auth(
        scenario.get("crypto_enabled", False),
        algorithm=scenario.get("crypto_algorithm", "hmac_sha256"),
    )
    client.set_llm_assistance(False)

    if scenario.get("jamming"):
        client.add_jamming_zone(
            center=(10.0, 45.0, 10.0),
            radius=16.0,
            jam_type=scenario["jamming"],
            intensity=0.85,
        )
    if scenario.get("spoofing"):
        client.add_spoofing_zone(
            center=(24.0, 78.0, 10.0),
            radius=14.0,
            spoof_type=scenario["spoofing"],
            phantom_count=2,
            falsification_magnitude=7.0,
            coordinate_vector=(6.0, -4.0, 0.0),
        )

    client.start_simulation(
        formation=scenario.get("formation", "communication_aware"),
        path_algorithm=scenario.get("path_algorithm", "astar"),
        crypto_auth=scenario.get("crypto_enabled", False),
        crypto_algorithm=scenario.get("crypto_algorithm", "hmac_sha256"),
    )
    wait_until_done(client)
    client.stop_simulation()

    attack = client.attack_metrics()
    protocol = client.protocol_stats()
    results = client.simulation_results()
    return {
        "scenario": scenario["name"],
        "crypto_enabled": attack.get("crypto_enabled"),
        "crypto_algorithm": attack.get("crypto_algorithm"),
        "detection_rate": attack.get("detection_rate"),
        "tp": attack.get("tp"),
        "fp": attack.get("fp"),
        "fn": attack.get("fn"),
        "tn": attack.get("tn"),
        "messages_sent": protocol.get("mavlink", {}).get("messages_sent"),
        "messages_dropped": protocol.get("mavlink", {}).get("messages_dropped"),
        "destination_reached": results.get("destination_reached"),
        "steps": results.get("steps"),
    }


def main() -> None:
    client = SwarmSquadClient()
    scenarios = [
        {"name": "baseline"},
        {"name": "low_jam", "jamming": "low_jam"},
        {"name": "spoof_phantom_crypto_off", "spoofing": "phantom"},
        {
            "name": "spoof_phantom_crypto_on",
            "spoofing": "phantom",
            "crypto_enabled": True,
            "crypto_algorithm": "hmac_sha256",
        },
    ]

    report = []
    for scenario in scenarios:
        print(f"\n=== Running: {scenario['name']} ===")
        summary = run_scenario(client, scenario)
        report.append(summary)
        print(summary)

    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report to {RESULTS_PATH}")
    print("Use the GUI to capture matching screenshots for each scenario.")


if __name__ == "__main__":
    main()
