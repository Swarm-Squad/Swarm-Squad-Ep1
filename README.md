# Swarm Squad Ep1

Swarm Squad Ep1 is an installable swarm simulation platform for:

- live browser visualization,
- script-driven simulation control,
- cyber attack/defense experimentation,
- reproducible headless research runs.

## What You Run

Default launcher (GUI + simulation backend):

```bash
uv run swarm-squad-ep1
```

Access points:

- GUI: `http://localhost:5000`
- Simulation API: `http://localhost:5001`

## Quick Setup

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev
cp .env.example .env
docker compose up -d
ollama serve
```

Then launch:

```python
from swarm_squad_ep1.client import SwarmSquadClient

client = SwarmSquadClient()
client.reset_simulation()
client.set_algorithm(formation="communication_aware", path_algorithm="astar")
client.add_jamming_zone(center=(12, 45, 10), radius=16, jam_type="low_jam")
client.start_simulation()
```

## CLI Summary

```bash
uv run swarm-squad-ep1
uv run swarm-squad-ep1 gui
uv run swarm-squad-ep1 services --simulation-only
uv run swarm-squad-ep1 services --chat-only
uv run swarm-squad-ep1 research list
uv run swarm-squad-ep1 research smoke
```

## Documentation

- Start here: `docs/getting-started.md`
- Runtime and command modes: `docs/runtime-and-cli.md`
- Custom script workflows (student/user): `docs/script-customization.md`
- Troubleshooting: `docs/troubleshooting.md`
- System architecture: `docs/system-overview.md`
- Python client reference: `docs/client-api-reference.md`
- Algorithms and threat model: `docs/algorithms-and-threat-model.md`
- Research harness and E1-E6: `docs/research-harness.md`
- Communication model notes: `docs/communication_model_research.md`

## Development

Install dev/test dependencies first:

```bash
uv sync --extra dev
```

Then run tests:

```bash
uv run pytest -q
```
