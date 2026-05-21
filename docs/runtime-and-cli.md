# Runtime and CLI

Swarm Squad Ep1 runs as a dual-service system by default.
This page explains service roles, launch modes, and how to choose the right command.

## Runtime model

The default launcher starts:

- `swarm_squad_ep1.simulation.api` on port `5001`
- `swarm_squad_ep1.chat.app` on port `5000`

The browser talks to the chat service, and the chat service forwards simulation calls.
Your Python scripts can call the simulation service directly through `SwarmSquadClient`.

## Core commands

### Start full stack (recommended)

```bash
uv run swarm-squad-ep1
```

This is the default classroom/development mode:

- starts both services,
- keeps the process in monitor mode,
- prints endpoints and status messages.

### Explicit GUI stack launch

```bash
uv run swarm-squad-ep1 gui
```

Equivalent to the default command above.

### Start without monitor loop

```bash
uv run swarm-squad-ep1 gui --no-monitor
```

Starts services and returns terminal control immediately.
Use this when launching from scripts or another supervisor.

## Service-only launch modes

Start only simulation backend:

```bash
uv run swarm-squad-ep1 services --simulation-only
```

Start only chat/GUI service:

```bash
uv run swarm-squad-ep1 services --chat-only
```

Notes:

- `--simulation-only` and `--chat-only` are mutually exclusive.
- Omit both flags to start both services through `services`.
- `--no-monitor` also works with `services`.

## Research harness commands

List available experiments:

```bash
uv run swarm-squad-ep1 research list
```

Run one experiment:

```bash
uv run swarm-squad-ep1 research run --experiment=E1 --seeds=3
```

Run all experiments:

```bash
uv run swarm-squad-ep1 research run --experiment=all --seeds=5
```

Smoke test:

```bash
uv run swarm-squad-ep1 research smoke
```

Render plots from a CSV:

```bash
uv run swarm-squad-ep1 research plot --csv results/E1/<timestamp>.csv
```

## Choosing the right mode

- Use `swarm-squad-ep1` for normal GUI + script workflows.
- Use `services --simulation-only` when integrating your own UI or automation.
- Use `services --chat-only` only if a simulation backend is already running elsewhere.
- Use `research ...` for reproducible headless evaluations.

## Ports and health endpoints

Default ports:

- Chat/GUI: `5000`
- Simulation API: `5001`

Useful checks:

```bash
curl -s http://localhost:5000/health
curl -s http://localhost:5000/status
curl -s http://localhost:5001/status
```

## Environment variables that affect runtime

Common runtime knobs in `.env`:

- `CHAT_API_PORT`, `SIM_API_PORT`
- `OLLAMA_HOST`
- `QDRANT_HOST`, `QDRANT_PORT`
- `EDU_BEGINNER_MODE`, `EDU_DEFAULT_PRESET`
- `LLM_ASSISTANCE_ENABLED`, `CRYPTO_AUTH_ENABLED`

Detailed variable explanations are in `.env.example`.
