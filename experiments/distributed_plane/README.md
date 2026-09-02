# Distributed reasoning plane — one reasoning agent per vehicle

A double multi-agent layer for Swarm-Squad-Ep1: each vehicle carries its own reasoning agent, the
reasoning agents exchange compact proposal records over the simulator's MAVLink bus, and a
deterministic override gate on each vehicle decides whether the aggregated proposal or the inherited
controller drives that vehicle in the next step.

**This directory adds files only.** No file outside it is modified, so the inherited control law
(`algo/controller.py`) is provably untouched: the plane presents the simulator's own
`LLMAssistanceController` interface and is consumed through the existing hook in
`src/swarm_squad_ep1/research/runner.py`.

## Layout

| path | contents |
|---|---|
| `swarm_bridge.py` | the plane: per-vehicle reasoners, proposal bus, wire-format validation, override gate, experiment runner |
| `override_gate.py` | record encode/decode, safety envelope, `verify`, circular fusion, canonicalisation |
| `aggregation.py` | consistency sweep, quorum rule, admission gate, W-MSR |
| `redteam.py` | Byzantine failure-class harness |
| `test_gate.py` | gate property tests |
| `demo_stage6.py` | runnable demo (see below) |
| `build_stage6_spec.py` | regenerates `docs/integration_spec.md` from `results/*.csv` |
| `docs/` | design specifications, each with a dated corrections section |
| `results/` | measured results the specification renders from |

## Running

```bash
# from the repository root
export PYTHONPATH=src
python experiments/distributed_plane/demo_stage6.py --n 5 --arm distributed --byzantine agent3 --steps 400

# arms: baseline (inherited controller only) | centralised (one ungated supervisor) | distributed
# other flags: --no-jamming --crypto --sep-mode {static,worstcase,reported} --probe-g1 --enforce-g1 --json
python experiments/distributed_plane/test_gate.py     # gate property tests
```

The demo constructs scenarios exactly as the experiment harness does, so a single-seed run reproduces
the corresponding row of `results/stage6_grid.csv`.

## Extra dependencies

Beyond the simulator's own: `pathfinding3d`, `python-dotenv`, `fastapi`; `tabulate` and `pandas` only
for regenerating the specification.

## Where the numbers come from

`docs/integration_spec.md` is generated, not written — every table and every in-prose quantity is
rendered from `results/*.csv` at build time. Regenerate with:

```bash
python experiments/distributed_plane/build_stage6_spec.py   # runs from any cwd
```

Section 9 of that document states explicitly what the measurements do not establish.
