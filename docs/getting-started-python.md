# Getting Started (Python + GUI Track)

This quick start is for users who will run custom Python scripts while watching the same simulation in the web GUI.

## 1) Two-terminal workflow

### Terminal A (keep running)

```bash
uv run swarm-squad-ep1
```

### Browser

- `http://localhost:5000`

### Terminal B (run scripts)

```bash
uv run python your_script.py
```

## 2) Minimal script example

Fastest starter:

```bash
uv run python examples/ep1_assignment_starter.py
```

Or build your own script:

```python
from swarm_squad_ep1.client import SwarmSquadClient

client = SwarmSquadClient()
client.reset_simulation()
client.clear_jamming_zones()
client.clear_spoofing_zones()
client.set_algorithm(formation="communication_aware", path_algorithm="astar")
client.start_simulation()
print(client.simulation_state())
```

## 3) Advanced example

Use the full-control reference script:

```bash
uv run python examples/ep1_custom_control_loop.py
```

## 4) Assignment links

- Student handout: `docs/assignment-python-student.md`
- Teacher manual: `docs/assignment-python-teacher.md`

## 5) Next references

- `docs/script-customization.md`
- `docs/client-api-reference.md`
- `docs/troubleshooting.md`
