# SwarmSquadClient API Reference

`SwarmSquadClient` is the script-facing control API for Swarm Squad Ep1.
It sends HTTP calls to the simulation backend and returns decoded JSON responses.

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

- `base_url`: simulation API endpoint.
- `timeout`: per-request timeout in seconds.

## Lifecycle and status methods

`status()`

- Returns current service status payload, including boundaries and mission end.

`simulation_state()`

- Returns high-level runtime state (`running`, algorithm settings, counters).

`simulation_config()`

- Returns simulation configuration currently active in backend.

`start_simulation(formation="communication_aware", path_algorithm="astar", crypto_auth=None, crypto_algorithm="hmac_sha256")`

- Starts simulation loop.
- Optionally sets crypto state at launch.

`stop_simulation()`

- Stops simulation loop.

`reset_simulation()`

- Reinitializes simulation state (agents/zones/runtime state).

`simulation_results()`

- Returns current aggregated mission and metric summaries.

`simulate_step()`

- Advances one simulation tick manually (useful for script-driven control loops).

## Algorithm and motion control

`set_algorithm(formation=None, path_algorithm=None, default_obstacle_type=None)`

- Updates active algorithm settings.
- Provide one or more fields; empty calls return an error payload.

`move_agent(agent, x, y, z=0.0)`

- Directly moves one agent target position through the API.

`run_script_control_loop(controller, steps=100, step_interval_s=0.0, auto_simulate_step=True)`

- Runs a user-defined Python control callback each step and optionally calls
  `simulate_step()` after each command batch.
- Intended for custom, script-owned algorithm development with live GUI visualization.

## Custom path algorithm plugin methods

`register_custom_algorithm(name, import_path, description="", replace=False, mode="waypoint")`

- Registers a custom path algorithm callable by import path.
- `import_path` format: `"module:function"` (or `"module.function"`).
- The module can come from your own project code, not only Swarm Squad sources.

`list_custom_algorithms()`

- Lists registered custom path algorithms.

`remove_custom_algorithm(name)`

- Removes one custom path algorithm.

## Jamming zone controls

`add_jamming_zone(center, radius, jam_type="low_jam", intensity=1.0)`

- `center`: `(x, y, z)`
- `jam_type`: `"physical"`, `"low_jam"`, or `"high_jam"`

`list_jamming_zones()`

- Returns all active jamming zones.

`delete_jamming_zone(zone_id)`

- Removes one jamming zone.

`clear_jamming_zones()`

- Removes all jamming zones.

## Spoofing zone controls

`add_spoofing_zone(center, radius, spoof_type="phantom", phantom_count=2, falsification_magnitude=8.0, coordinate_vector=(10.0, 10.0, 0.0))`

- `spoof_type`: `"phantom"`, `"position_falsification"`, or `"coordinate"`

`list_spoofing_zones()`

- Returns all active spoofing zones.

`delete_spoofing_zone(zone_id)`

- Removes one spoofing zone.

`clear_spoofing_zones()`

- Removes all spoofing zones.

## Defense and communication toggles

`set_crypto_auth(enabled, algorithm="hmac_sha256")`

- Enables/disables cryptographic auth on communications.

`crypto_auth_status()`

- Returns current crypto auth state.

`set_v2v_channel(enabled, params=None)`

- Enables/disables the V2V communication model and optionally updates channel params.

`v2v_channel_status()`

- Returns V2V link summaries and current channel parameters.

`set_comm_model(model)`

- Convenience wrapper for comm model switching:
  - `"v2v_channel"` -> enables V2V channel model
  - `"legacy"` -> disables V2V channel model

`set_llm_assistance(enabled)`

- Enables/disables LLM guidance support.

`llm_assistance_status()`

- Returns LLM assistance state.

`protocol_stats()`

- Returns communication protocol counters (sent/received/dropped/spoof/rejected).

`attack_metrics()`

- Returns spoof-detection metrics (TP/FP/FN/TN + derived rates).

## Visualization and entity pulls

`agents()`

- Returns current per-agent state payload.

`agent(agent_id)`

- Returns one agent state payload.

`add_agent(x, y, z=0.0)`

- Adds a new agent at explicit coordinates.

`remove_agent(agent_id)`

- Removes one agent.

`visualization(trail_length="short")`

- Returns visualization payload used by GUI (`trail_length` supports `"short"` and `"all"`).

## Preset and educational helpers

`apply_preset(preset, seed=0)`

- Resets the live simulation and configures it from a preset.

`apply_education_preset(preset, seed=0)`

- Backward-compatible alias for `apply_preset`.

`list_presets()`

- Static helper returning available preset metadata.

`build_preset_scenario(preset, seed=0)`

- Static helper that constructs a `Scenario` object from a preset key.

## Headless research helpers

`run_headless_scenario(scenario, keep_trace=False, verbose=False)`

- Runs a single scenario without requiring running HTTP services.

`run_headless_preset(preset, seed=0, keep_trace=False, verbose=False)`

- Builds and executes one preset headlessly.

## Example: live script with GUI visualization

```python
from swarm_squad_ep1.client import SwarmSquadClient

client = SwarmSquadClient()

client.reset_simulation()
client.clear_jamming_zones()
client.clear_spoofing_zones()

client.set_algorithm(formation="communication_aware", path_algorithm="theta_star")
client.add_jamming_zone(center=(10, 40, 10), radius=15, jam_type="high_jam")
client.add_spoofing_zone(center=(18, 60, 10), radius=25, spoof_type="phantom")
client.set_crypto_auth(True, algorithm="hmac_sha256")
client.set_llm_assistance(True)

client.start_simulation()
print(client.simulation_state())
print(client.protocol_stats())
```

## Example: custom algorithm + script loop

```python
from swarm_squad_ep1.client import SwarmSquadClient

client = SwarmSquadClient()
client.register_custom_algorithm(
    name="midpoint_demo",
    import_path="examples.custom_algorithms.midpoint_path:midpoint_path",
    description="Midpoint demo path",
    replace=True,
)
client.set_algorithm(path_algorithm="midpoint_demo")

first_agent = next(iter(client.agents()["agents"]))

def controller(state, step):
    pos = state["agents"][first_agent]["position"]
    return [{"agent": first_agent, "x": pos[0] + 1.0, "y": pos[1], "z": pos[2]}]

trace = client.run_script_control_loop(controller, steps=10, step_interval_s=0.1)
print(len(trace))
```

## Error handling guidance

- Transport or timeout failures raise HTTP client exceptions.
- API validation failures return error payloads from server.
- For robust scripts:
  - wrap calls with try/except,
  - check expected keys in responses,
  - call `reset_simulation()` before each new scenario batch.
