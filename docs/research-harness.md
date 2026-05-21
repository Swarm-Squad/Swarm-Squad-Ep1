# Research Harness Guide

Swarm Squad Ep1 includes a built-in headless harness for repeatable experiments,
ablation studies, and figure generation workflows.

Entrypoint:

```bash
uv run swarm-squad-ep1 research <command> [args...]
```

## Available commands

List experiment IDs:

```bash
uv run swarm-squad-ep1 research list
```

Run one experiment:

```bash
uv run swarm-squad-ep1 research run --experiment=E1 --seeds=3
```

Run multiple experiments:

```bash
uv run swarm-squad-ep1 research run --experiment=E1,E2,E3 --seeds=5
```

Run all experiments:

```bash
uv run swarm-squad-ep1 research run --experiment=all --seeds=5
```

Smoke test:

```bash
uv run swarm-squad-ep1 research smoke
```

Plot from one CSV:

```bash
uv run swarm-squad-ep1 research plot --csv results/E1/<timestamp>.csv
```

## Experiment set (E1-E6)

- `E1`: single-vs-dual attack susceptibility.
- `E2`: LLM assistance under combined attacks.
- `E3`: path planning algorithm comparison under attack.
- `E4`: crypto authentication algorithm comparison.
- `E5`: full factorial (`attack x llm x crypto`) evidence matrix.
- `E6`: communication model comparison (`v2v` vs `legacy`).

## Output artifacts

By default, outputs are written under `results/`:

- per-experiment CSV (row-wise scenario results),
- summary JSON,
- generated plot PNG files (when plotting is invoked).

You can override output location:

```bash
uv run swarm-squad-ep1 research run --experiment=E5 --out-dir my_results
```

## Useful run options

- `--seeds <N>`: repeats each scenario for N random seeds.
- `--max-steps <N>`: cap scenario simulation steps.
- `--keep-trace`: preserve detailed traces for deeper analysis.
- `--quiet`: reduce console verbosity.

## Reproducibility tips

- Always report:
  - experiment ID,
  - seed count,
  - code revision,
  - key environment toggles (LLM/Crypto/ports if changed).
- Keep output folders per run timestamp or git branch.
- Use fixed seed counts when comparing algorithm variants.

## Suggested analysis flow

1. Run smoke to validate environment.
2. Run one target experiment with small seed count.
3. Inspect CSV and summary JSON for expected schema.
4. Increase seed count for robust comparison.
5. Generate plots and include command provenance in notes.

## Relationship to live runtime

Research commands run headless scenarios and do not require the GUI stack.
Use the main runtime (`swarm-squad-ep1`) when you need live visualization.

For scenario definitions and experiment matrices, see:

- `src/swarm_squad_ep1/research/scenarios.py`
- `src/swarm_squad_ep1/research/experiments.py`
- `src/swarm_squad_ep1/research/runner.py`
