# SwarmSquadClient API Reference

`SwarmSquadClient` is the script-facing API for live simulation control.
Methods call the simulation backend and return JSON payloads (or CSV text for
result download).

Import:

```python
from swarm_squad_ep1.client import SwarmSquadClient
```

## Constructor

```python
client = SwarmSquadClient(
    base_url="http://localhost:5001",  # optional
    timeout=10.0,                      # optional
)
```

## Service and simulation lifecycle

`status()`

- Backend status including bounds and mission end.

`simulation_state()`

- Current runtime state (`running`, formation, agents, active zones).

`simulation_config()`

- Current configurable options published by backend:
  formations, path algorithms, crypto algorithms, labels, comm models.

`start_simulation(formation="communication_aware", path_algorithm="astar", crypto_auth=None, crypto_algorithm="hmac_sha256", destination=None, default_obstacle_type=None)`

- Starts simulation loop with optional launch-time destination and obstacle type.
- If `crypto_auth` is set, it also applies crypto enabled/algorithm settings.

`stop_simulation()`

- Stops simulation loop.

`reset_simulation()`

- Reinitializes simulation state.

`simulation_results()`

- Aggregated mission-level results snapshot.

`download_simulation_results(format="json" | "csv")`

- Downloads results as JSON dict or CSV string.

`simulate_step()`

- Advances one simulation tick (manual stepping workflows).

## Agent controls

`agents()`

- Full per-agent map.

`agent(agent_id)`

- One agent snapshot.

`add_agent(x, y, z=0.0)`

- Create an agent.

`update_agent(agent_id, position=None, jammed=None, communication_quality=None)`

- Partial update for one agent.

`remove_agent(agent_id)`

- Delete one agent.

`move_agent(agent, x, y, z=0.0)`

- Set movement target for one agent.

## Jamming zone controls

`add_jamming_zone(center, radius, jam_type="low_jam", intensity=1.0)`

`list_jamming_zones()`

`get_jamming_zone(zone_id)`

`update_jamming_zone(zone_id, center=None, radius=None, obstacle_type=None, intensity=None, active=None)`

`delete_jamming_zone(zone_id)`

`clear_jamming_zones()`

## Spoofing zone controls

`add_spoofing_zone(center, radius, spoof_type="phantom", phantom_count=2, falsification_magnitude=8.0, coordinate_vector=(10.0, 10.0, 0.0))`

`list_spoofing_zones()`

`get_spoofing_zone(zone_id)`

`update_spoofing_zone(zone_id, center=None, radius=None, spoof_type=None, phantom_count=None, falsification_magnitude=None, coordinate_vector=None, active=None)`

`delete_spoofing_zone(zone_id)`

`clear_spoofing_zones()`

## Algorithm controls

`set_algorithm(formation=None, path_algorithm=None, default_obstacle_type=None)`

- Update one or more algorithm knobs.

`path_algorithms()`

- Convenience list of available path algorithms.

`custom_path_algorithms()`

- List of currently registered custom path entries.

`register_custom_algorithm(name, import_path, description="", replace=False, mode="waypoint")`

- Register custom path plugin by import path.

`list_custom_algorithms()`

- List custom path plugins.

`remove_custom_algorithm(name)`

- Unregister custom path plugin.

## Crypto and communication controls

`set_crypto_auth(enabled, algorithm="hmac_sha256")`

`crypto_auth_status()`

`register_custom_crypto_algorithm(name, sign_import_path, verify_import_path, description="", replace=False)`

`list_custom_crypto_algorithms()`

`remove_custom_crypto_algorithm(name)`

`set_v2v_channel(enabled, params=None)`

`v2v_channel_status()`

`set_comm_model("v2v_channel" | "legacy")`

## LLM and metrics controls

`set_llm_assistance(enabled)`

`llm_assistance_status()`

`llm_targets()`

`clear_llm_target(agent_id)`

`clear_all_llm_targets()`

`protocol_stats()`

`attack_metrics()`

`visualization(trail_length="short" | "all")`

## Script loop helper

`run_script_control_loop(controller, steps=100, step_interval_s=0.0, auto_simulate_step=True)`

- Runs `controller(state, step_idx)` each step.
- Each returned command should look like:
  `{"agent": "agent1", "x": 10.0, "y": 5.0, "z": 2.0}`.
- Intended for custom controller logic with live GUI feedback.

## Preset and headless helpers

`apply_preset(preset, seed=0)`

`apply_education_preset(preset, seed=0)` (alias)

`list_presets()`

`build_preset_scenario(preset, seed=0)`

`run_headless_scenario(scenario, keep_trace=False, verbose=False)`

`run_headless_preset(preset, seed=0, keep_trace=False, verbose=False)`

## Example references

- Full live control example: `examples/ep1_custom_control_loop.py`
- Path plugin example: `examples/custom_algorithms/midpoint_path.py`
- Crypto plugin example: `examples/custom_algorithms/xor_hmac_crypto.py`
