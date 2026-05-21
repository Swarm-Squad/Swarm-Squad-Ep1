# Troubleshooting

This page maps common symptoms to likely causes and concrete recovery steps.

## Quick diagnostic bundle

Run this first when behavior looks inconsistent:

```bash
curl -s http://localhost:5000/health
curl -s http://localhost:5000/status
curl -s http://localhost:5001/status
```

Interpretation:

- `5000/health` checks GUI/chat service plus simulation reachability.
- `5000/status` is the chat-side status contract used by frontend bootstrap.
- `5001/status` confirms direct simulation availability.

## Symptom: ports are already in use

Typical error:

- service startup fails or one service exits immediately.

Check:

```bash
lsof -i :5000
lsof -i :5001
```

Fix:

- stop stale processes on those ports,
- restart with `uv run swarm-squad-ep1`.

## Symptom: GUI opens but scene does not update

Likely causes:

- simulation service is not running,
- startup race or prior crashed process.

Check:

```bash
curl -s http://localhost:5001/status
curl -s http://localhost:5000/health
```

Fix:

1. stop runtime (`Ctrl+C` where it was launched),
2. relaunch `uv run swarm-squad-ep1`,
3. hard-refresh browser tab.

## Symptom: chat/LLM commands hang on "Thinking..."

Likely causes:

- Ollama not running,
- wrong `OLLAMA_HOST`,
- remote tunnel mismatch.

Check:

```bash
curl -s http://localhost:11434/api/tags
```

Fix:

- start Ollama: `ollama serve`
- if using a non-default tunnel port, update `OLLAMA_HOST` in `.env`
- restart runtime after changing `.env`.

## Symptom: Qdrant warnings at startup

Likely cause:

- Docker dependency not up yet.

Fix:

```bash
docker compose up -d
docker compose logs qdrant
```

If logs show healthy startup, relaunch Swarm Squad.

## Symptom: script gets connection errors

Typical message:

- `Connection refused` from `SwarmSquadClient`.

Likely causes:

- `swarm-squad-ep1` is not running,
- wrong client base URL.

Fix:

- launch runtime first,
- if needed, set `SwarmSquadClient(base_url="http://localhost:5001")` explicitly.

## Symptom: simulation behavior seems stale after script changes

Likely cause:

- previous zones/state still active.

Fix sequence:

```python
client.reset_simulation()
client.clear_jamming_zones()
client.clear_spoofing_zones()
```

Then re-apply algorithms/toggles and start again.

## Symptom: crypto enabled but mission still degrades

Expected behavior:

- crypto helps against spoofed/tampered packets,
- crypto does not eliminate jamming-induced delivery loss.

What to do:

- compare metrics with/without high-jam zones,
- keep LLM assistance enabled when testing under jamming.

## Symptom: research smoke test fails

Check commands:

```bash
uv run swarm-squad-ep1 research smoke
uv run pytest -q
```

If smoke fails:

- rerun with `--quiet` removed to inspect scenario names and failures,
- confirm dependencies (`ollama`, `docker compose up -d`) are available when required.

## Clean restart procedure

When in doubt:

1. Stop current runtime process.
2. Ensure dependencies are healthy (`docker compose up -d`, `ollama serve`).
3. Start fresh: `uv run swarm-squad-ep1`.
4. Re-run your script and verify in GUI.
