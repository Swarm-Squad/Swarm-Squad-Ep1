# Getting Started

This guide is for first-time users who want to run Swarm Squad Ep1.

Before diving in, choose your track:

- GUI-only track (no Python required): `docs/getting-started-gui.md`
- Python + GUI track (custom scripts): `docs/getting-started-python.md`

## What Swarm Squad Ep1 includes

Swarm Squad Ep1 has four major pieces:

- `swarm_squad_ep1.simulation.api`: simulation backend, algorithms, attack/defense logic.
- `swarm_squad_ep1.chat.app`: GUI/chat service and proxy API used by the browser.
- `swarm_squad_ep1.client.SwarmSquadClient`: Python API for script-driven control.
- `swarm_squad_ep1.research`: headless experiment harness (E1-E6) and plotting tools.

When you run `swarm-squad-ep1`, the runtime starts both services:

- Chat/GUI service on `http://localhost:5000`
- Simulation API on `http://localhost:5001`

## Prerequisites

- Python `3.11+`
- [uv](https://github.com/astral-sh/uv) for environment and dependency management
- Docker + Docker Compose (Qdrant dependency)
- Ollama (required for chat/LLM-assisted behavior)

## Setup and install

From the project root:

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev
cp .env.example .env
```

Start service dependencies:

```bash
docker compose up -d
ollama serve
```

Notes:

- Keep `ollama serve` running in a separate terminal.
- If you use remote/HPC Ollama, set `OLLAMA_HOST` in `.env` before launch.

## First launch

Start the full stack:

```bash
uv run swarm-squad-ep1
```

Expected behavior:

- Terminal shows startup banners and service access points.
- Browser UI is available at `http://localhost:5000`.
- Simulation status endpoint responds at `http://localhost:5001/status`.

## Verify health before classroom/demo use

Run these checks in another terminal:

```bash
curl -s http://localhost:5000/health
curl -s http://localhost:5000/status
curl -s http://localhost:5001/status
```

What to confirm:

- Chat health reports simulation API online.
- `status` payload includes `boundaries` and `mission_end`.
- Simulation service responds without timeout.

## First guided mission in the GUI

1. Open `http://localhost:5000`.
2. In the right panel, choose formation/path/comm settings.
3. Press Start simulation.
4. Watch:
   - agent motion in 3D,
   - jamming/spoofing overlays,
   - attack metrics and protocol stats panels.
5. Toggle:
   - crypto authentication,
   - LLM assistance,
   - path/formation algorithms.

This gives a baseline feel for how attacks and countermeasures affect mission success.

## First custom script (live with GUI)

Keep `swarm-squad-ep1` running, then run a script in a second terminal.

Create `student_live_demo.py`:

```python
from swarm_squad_ep1.client import SwarmSquadClient

client = SwarmSquadClient()

client.reset_simulation()
client.set_algorithm(formation="communication_aware", path_algorithm="theta_star")
client.clear_jamming_zones()
client.clear_spoofing_zones()
client.add_jamming_zone(center=(10, 45, 10), radius=16, jam_type="high_jam")
client.set_crypto_auth(True, algorithm="hmac_sha256")
client.set_llm_assistance(True)
client.start_simulation()

print(client.simulation_state())
```

Run it:

```bash
uv run python student_live_demo.py
```

Expected result:

- Your script updates the running backend.
- GUI immediately reflects changed zones/algorithm/state.

For the full phased script that covers all major knobs (agents, zones, comm model,
custom path, custom crypto, LLM toggle, metrics, cleanup), run:

```bash
uv run python examples/ep1_custom_control_loop.py
```

Related docs:

- `docs/script-customization.md`
- `docs/client-api-reference.md`

## Fast validation commands

```bash
uv run pytest -q
uv run swarm-squad-ep1 -h
uv run swarm-squad-ep1 research smoke
```

## Where to go next

- GUI track student assignment: `docs/assignment-gui-student.md`
- GUI track teacher manual: `docs/assignment-gui-teacher.md`
- Python track student assignment: `docs/assignment-python-student.md`
- Python track teacher manual: `docs/assignment-python-teacher.md`
- GUI quick start: `docs/getting-started-gui.md`
- Python quick start: `docs/getting-started-python.md`
- Runtime and command modes: `docs/runtime-and-cli.md`
- Script cookbook and API usage patterns: `docs/script-customization.md`
- Troubleshooting by symptoms: `docs/troubleshooting.md`
- Full architecture and references:
  - `docs/system-overview.md`
  - `docs/client-api-reference.md`
  - `docs/algorithms-and-threat-model.md`
  - `docs/research-harness.md`
