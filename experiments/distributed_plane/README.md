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
| `build_report.py` | regenerates `docs/literature_matrix.md` from `results/byzantine_llm_swarm_matrix.csv` and `results/screening/` |
| `docs/` | design specifications and the literature matrix (below) |
| `results/` | measured results and figures the documents render from |

### Documents

| document | what it fixes | generated from |
|---|---|---|
| `docs/architecture_spec.md` | the five architecture invariants, the layer split, the A0–A4 command-authority hierarchy | written; figure `results/architecture.png` |
| `docs/message_interface.md` | the proposal-record wire format, bandwidth budget, graph-robustness requirement | written; `results/bandwidth_budget.csv`, `results/topology_robustness.csv` |
| `docs/override_spec.md` | the deterministic gate: envelope, evidence rules, arbitration, property tests | written; `results/gate_tests.csv` |
| `docs/aggregation_spec.md` | consistency sweep, quorum rule, bounded-influence proposition, W-MSR | written; `results/aggregation_scaling.csv`, `results/failure_cases.csv`, `results/kappa_tradeoff.csv`, `results/tau_tradeoff.csv` |
| `docs/integration_spec.md` | what running the plane on the simulator falsified in the four documents above | **generated** from `results/stage6_*.csv` |
| `docs/literature_matrix.md` | 22 prior works scored on nine capability axes, each verified against its primary source | **generated** from `results/byzantine_llm_swarm_matrix.csv` |

Each written document carries a dated corrections section recording what a later stage falsified in it;
the corrections are appended rather than edited in silently, so the original claim and its refutation
both stay readable.

## Running

```bash
# from the repository root, in the project environment
uv sync
uv run python experiments/distributed_plane/demo_stage6.py --n 5 --arm distributed --byzantine agent3 --steps 400

# arms: baseline (inherited controller only) | centralised (one ungated supervisor) | distributed
# other flags: --no-jamming --crypto --sep-mode {static,worstcase,reported} --probe-g1 --enforce-g1 --json
uv run python experiments/distributed_plane/test_gate.py     # gate property tests
```

`PYTHONPATH=src` works instead of `uv run` only if the package and its dependencies
(`numpy`, `pathfinding3d`, `python-dotenv`, `matplotlib`) are already importable in the active
interpreter; all four are declared in the repository's `pyproject.toml`, so `uv sync` is the
shorter path.

The demo constructs scenarios exactly as the experiment harness does, so a single-seed run reproduces
the corresponding row of `results/stage6_grid.csv`.

## Extra dependencies

Beyond the simulator's own: `pathfinding3d`, `python-dotenv`, `fastapi`; `tabulate` and `pandas` only
for regenerating the specification.

## Where the numbers come from

`docs/integration_spec.md` is generated, not written — every table and every in-prose quantity is
rendered from `results/*.csv` at build time. Regenerate with:

```bash
uv run --extra research --with tabulate python experiments/distributed_plane/build_stage6_spec.py
uv run --extra research python experiments/distributed_plane/build_report.py   # docs/literature_matrix.md
```

Both generators resolve their inputs and outputs relative to their own location, so they run from any
working directory and rewrite the committed document in place. Re-running them on an unmodified
checkout reproduces the committed file byte for byte, which is the check that a document has not
drifted from its data.

Section 9 of `docs/integration_spec.md` states explicitly what the measurements do not establish.
