# Script Customization and Live Visualization

This guide shows how to write your own Python control scripts while watching results
live in the Swarm Squad Ep1 GUI.

## Workflow model

One backend, two interfaces:

- Your script uses `SwarmSquadClient` to call simulation endpoints.
- The browser GUI reads the same backend state through the chat proxy.

That means every change in your script (zones, algorithms, toggles) appears in the GUI
without restarting services.

## Step 1: start the runtime

In terminal A:

```bash
uv run swarm-squad-ep1
```

Keep this process running.

## Step 2: open the GUI

In a browser, open:

- `http://localhost:5000`

Keep it open to observe updates as your script runs.

## Step 3: write and run scripts in terminal B

### Template A: baseline live control

```python
from swarm_squad_ep1.client import SwarmSquadClient

client = SwarmSquadClient()

client.reset_simulation()
client.clear_jamming_zones()
client.clear_spoofing_zones()
client.set_algorithm(formation="communication_aware", path_algorithm="astar")
client.set_crypto_auth(False)
client.set_llm_assistance(False)

client.start_simulation()
print(client.simulation_state())
```

### Template B: attack + defense scenario

```python
from swarm_squad_ep1.client import SwarmSquadClient

client = SwarmSquadClient()

client.reset_simulation()
client.clear_jamming_zones()
client.clear_spoofing_zones()

client.add_jamming_zone(center=(15, 50, 10), radius=18, jam_type="high_jam")
client.add_spoofing_zone(
    center=(20, 65, 10),
    radius=22,
    spoof_type="position_falsification",
    falsification_magnitude=10.0,
)
client.set_crypto_auth(True, algorithm="hmac_sha256")
client.set_llm_assistance(True)
client.set_algorithm(path_algorithm="theta_star")

client.start_simulation()
print(client.protocol_stats())
```

### Template C: quick algorithm sweep

```python
from swarm_squad_ep1.client import SwarmSquadClient

client = SwarmSquadClient()

for algo in ["direct", "astar", "theta_star", "dijkstra", "bi_astar"]:
    client.reset_simulation()
    client.set_algorithm(formation="communication_aware", path_algorithm=algo)
    client.start_simulation(path_algorithm=algo)
    state = client.simulation_state()
    print(algo, state.get("running"))
    client.stop_simulation()
```

Run scripts with:

```bash
uv run python your_script.py
```

If you previously ran plain `uv sync` and noticed test tooling missing, reinstall
development dependencies with:

```bash
uv sync --extra dev
```

### Template D: script-defined control loop (GUI updates live)

```python
from swarm_squad_ep1.client import SwarmSquadClient

client = SwarmSquadClient()
client.reset_simulation()

first_agent = next(iter(client.agents()["agents"]))

def policy(state, step_idx):
    # Replace this with your own algorithm logic.
    pos = state["agents"][first_agent]["position"]
    return [{"agent": first_agent, "x": pos[0] + 1.0, "y": pos[1], "z": pos[2]}]

trace = client.run_script_control_loop(policy, steps=20, step_interval_s=0.1)
print("loop steps:", len(trace))
```

### Template E: register a custom path algorithm plugin

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
client.start_simulation(path_algorithm="midpoint_demo")
```

## Core `SwarmSquadClient` method groups

Simulation lifecycle:

- `start_simulation()`
- `stop_simulation()`
- `reset_simulation()`
- `simulation_state()`
- `simulation_results()`
- `simulate_step()`

Algorithms and control:

- `set_algorithm(formation=..., path_algorithm=..., default_obstacle_type=...)`
- `move_agent(agent, x, y, z)`
- `run_script_control_loop(controller, steps=..., step_interval_s=...)`
- `register_custom_algorithm(...)`
- `list_custom_algorithms()`
- `remove_custom_algorithm(name)`

Jamming controls:

- `add_jamming_zone(...)`
- `list_jamming_zones()`
- `delete_jamming_zone(zone_id)`
- `clear_jamming_zones()`

Spoofing controls:

- `add_spoofing_zone(...)`
- `list_spoofing_zones()`
- `delete_spoofing_zone(zone_id)`
- `clear_spoofing_zones()`

Defense and comms toggles:

- `set_crypto_auth(enabled, algorithm=...)`
- `crypto_auth_status()`
- `set_v2v_channel(enabled, params=...)`
- `v2v_channel_status()`
- `set_comm_model("v2v_channel" | "legacy")`
- `set_llm_assistance(enabled)`
- `llm_assistance_status()`
- `protocol_stats()`

Visualization/state pulls:

- `status()`
- `agents()`
- `agent(agent_id)`
- `add_agent(x, y, z)`
- `remove_agent(agent_id)`
- `visualization(trail_length="short" | "all")`
- `simulation_config()`

Preset helpers:

- `apply_preset(preset, seed=0)`
- `list_presets()`
- `build_preset_scenario(preset, seed=0)`

## Recommended script sequence

Use this order to avoid stale state:

1. `reset_simulation()`
2. clear zones (`clear_jamming_zones()`, `clear_spoofing_zones()`)
3. set algorithms/toggles
4. add zones and custom settings
5. `start_simulation()`
6. poll `simulation_state()` or `visualization()`

## Common mistakes and fixes

`Connection refused` from client:

- runtime is not running; start `uv run swarm-squad-ep1`.

Script runs but GUI does not update:

- confirm GUI is at `http://localhost:5000` and runtime was not restarted between calls.

Changes do not appear as expected:

- call `reset_simulation()` before applying a new scenario.
- print `simulation_config()` and `simulation_state()` to inspect active settings.

Crypto appears enabled but spoofing still has impact:

- cryptographic auth mitigates forged/tampered messages, but does not remove pure RF jamming effects.

## Headless mode (no GUI needed)

For purely programmatic experiments:

- `SwarmSquadClient.run_headless_preset(...)`
- `SwarmSquadClient.run_headless_scenario(...)`

For larger matrices and output artifacts, use `swarm-squad-ep1 research ...`.
