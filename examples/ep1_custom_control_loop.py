"""Script-driven control loop + custom algorithm registration demo."""

from __future__ import annotations

from swarm_squad_ep1.client import SwarmSquadClient


def scripted_swarm_policy(state: dict, step_idx: int) -> list[dict]:
    """
    Move each real agent forward in +x while keeping current altitude.

    This demonstrates user-defined decision logic running in Python while
    the GUI keeps rendering the live simulation.
    """
    commands: list[dict] = []
    for agent_id, agent in state.get("agents", {}).items():
        if str(agent_id).startswith("phantom_"):
            continue
        pos = agent.get("position", [0.0, 0.0, 0.0])
        commands.append(
            {
                "agent": agent_id,
                "x": float(pos[0]) + 1.0,
                "y": float(pos[1]),
                "z": float(pos[2]),
            }
        )
    return commands


def main() -> None:
    client = SwarmSquadClient()

    # 1) Register a custom path algorithm plugin (optional).
    client.register_custom_algorithm(
        name="midpoint_demo",
        import_path="examples.custom_algorithms.midpoint_path:midpoint_path",
        description="Midpoint demo path",
        replace=True,
    )

    # 2) Configure and start simulation with the custom path algorithm.
    client.set_algorithm(path_algorithm="midpoint_demo")
    client.set_comm_model("v2v_channel")
    client.set_crypto_auth(True, algorithm="hmac_sha256")
    client.set_llm_assistance(True)
    client.start_simulation(path_algorithm="midpoint_demo")

    # 3) Run a user-owned control loop for live interaction.
    trace = client.run_script_control_loop(
        scripted_swarm_policy,
        steps=20,
        step_interval_s=0.1,
        auto_simulate_step=True,
    )
    print(f"Completed scripted loop with {len(trace)} iterations")

    # 4) Clean up.
    client.stop_simulation()


if __name__ == "__main__":
    main()
