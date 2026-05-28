# Script Customization and Live Visualization

This guide shows the canonical way to control Swarm Squad Ep1 from Python while
watching the same run in the web GUI.

## Runtime model

- Your script talks to the simulation backend through `SwarmSquadClient`.
- The GUI reads the same backend state through chat proxy routes.
- Script changes (agents, zones, algorithms, crypto, LLM toggles) appear live in GUI.

## Start once, control from scripts

Terminal A:

```bash
uv run swarm-squad-ep1
```

Browser:

- `http://localhost:5000`

Terminal B (run scripts):

```bash
uv run python your_script.py
```

## Canonical full-control workflow

Use `examples/ep1_custom_control_loop.py` as the reference script. It runs ten phases:

1. reset state + discover config,
2. register custom path + custom crypto,
3. add/update/remove agents,
4. add/update/list/delete jamming and spoofing zones,
5. switch formation/path,
6. toggle communication model,
7. toggle crypto off/on and switch built-in/custom algorithms,
8. toggle LLM assistance,
9. run script control loop with live GUI playback,
10. collect metrics and clean up.

This script is intentionally compact and safe to re-run.

## Knob mapping (goal -> client methods)

Simulation lifecycle:

- `start_simulation(..., destination=..., default_obstacle_type=...)`
- `stop_simulation()`
- `reset_simulation()`
- `simulation_state()`
- `simulation_results()`
- `download_simulation_results(format="json" | "csv")`
- `simulate_step()`

Agent lifecycle:

- `agents()`, `agent(agent_id)`
- `add_agent(x, y, z)`
- `update_agent(agent_id, position=..., jammed=..., communication_quality=...)`
- `remove_agent(agent_id)`
- `move_agent(agent, x, y, z)`

Jamming zone lifecycle:

- `add_jamming_zone(...)`
- `list_jamming_zones()`
- `get_jamming_zone(zone_id)`
- `update_jamming_zone(zone_id, ...)`
- `delete_jamming_zone(zone_id)`
- `clear_jamming_zones()`

Spoofing zone lifecycle:

- `add_spoofing_zone(...)`
- `list_spoofing_zones()`
- `get_spoofing_zone(zone_id)`
- `update_spoofing_zone(zone_id, ...)`
- `delete_spoofing_zone(zone_id)`
- `clear_spoofing_zones()`

Algorithm + comm + crypto controls:

- `set_algorithm(formation=..., path_algorithm=..., default_obstacle_type=...)`
- `set_v2v_channel(enabled, params=...)`
- `v2v_channel_status()`
- `set_comm_model("v2v_channel" | "legacy")`
- `set_crypto_auth(enabled, algorithm=...)`
- `crypto_auth_status()`
- `register_custom_algorithm(...)`
- `list_custom_algorithms()`
- `remove_custom_algorithm(name)`
- `register_custom_crypto_algorithm(...)`
- `list_custom_crypto_algorithms()`
- `remove_custom_crypto_algorithm(name)`

LLM and metrics:

- `set_llm_assistance(enabled)`
- `llm_assistance_status()`
- `llm_targets()`
- `clear_llm_target(agent_id)`
- `clear_all_llm_targets()`
- `protocol_stats()`
- `attack_metrics()`

## Custom plugin contracts

Custom path plugin callable:

```python
def my_path(start, goal, jamming_zones, **kwargs):
    return [start, goal]
```

Custom crypto plugin callables:

```python
def my_crypto_sign(*, key: bytes, data: bytes, sender_id: str, crypto=None) -> bytes:
    ...

def my_crypto_verify(
    *,
    key: bytes,
    data: bytes,
    signature: bytes,
    sender_id: str,
    crypto=None,
) -> bool:
    ...
```

Then register with:

```python
client.register_custom_crypto_algorithm(
    name="my_crypto",
    sign_import_path="my_package.crypto:my_crypto_sign",
    verify_import_path="my_package.crypto:my_crypto_verify",
    description="My custom crypto",
    replace=True,
)
```

## Fast sanity checklist for new scripts

1. call `reset_simulation()` before applying a new scenario,
2. print `simulation_config()` once to inspect available options,
3. start simulation after all setup calls,
4. stop simulation and cleanup in `finally`.

## Headless mode

For non-GUI experiments:

- `SwarmSquadClient.run_headless_preset(...)`
- `SwarmSquadClient.run_headless_scenario(...)`

For larger experiment matrices and plot generation, use `swarm-squad-ep1 research ...`.
