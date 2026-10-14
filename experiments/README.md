# Seeded supervisor experiments

Headless, repeated runs of the 2D simulator for the ICC revision of
"LLM-Guided Communication-Aware Control for Connectivity Recovery in Jammed
UAV Swarm Networks". The runs compare four supervisors on the layouts of the
CCNC paper:

| Supervisor | What it does |
| --- | --- |
| `none` | Reactive formation and behavior control only |
| `rule` | Applies the evasion hints in the telemetry; moves an unhinted jammed agent 10 units away from the field center; when J_n < P_T, moves an endpoint of the weakest link 5 units to fix its spacing |
| `mpc` | Rolls each candidate command (3 distances x 32 bearings) forward over the command lifetime with the link model, and keeps the one that maximizes predicted J_n with a small progress term and an obstacle penalty |
| `llm` | The paper's supervisor: `llama3.2:3b-instruct-q4_K_M` through Ollama, with the prompts in `src/swarm_squad_ep1/config.py` |

All supervisors see the same condensed telemetry, issue commands in the same
"Agent-i: Move d units bearing" form, and go through the same parser, blend,
and command lifetime as the LLM.

## Scenarios

`scenarios/*.json` hold the obstacle layouts of the CCNC runs:

- `obstacles_1`: seven obstacles, physical mode
- `obstacles_2`: nine obstacles, physical mode
- `jamming`: one low-power jamming field

Each seed draws uniform jitter of +/-2 units on the initial positions and, for
`jamming`, +/-3 units on the field center. With zero jitter, `none` on
`obstacles_1` reproduces the GUI run of the paper (489 steps, J_n = 0.9723).

`--c2` adds jammed command-and-control (C2) links. A jammed agent whose
penetration depth is at least `--c2-depth` (default 0: any jammed agent) does
not receive commands, and the supervisor sees its last telemetry from before
the link was lost.

## Running

```bash
# non-LLM supervisors, unpaced with a virtual clock
uv run python -m swarm_squad_ep1.experiments.runner \
    --scenarios obstacles_1 obstacles_2 jamming --supervisors none rule mpc \
    --seeds 0-9 --out experiments/results/local
uv run python -m swarm_squad_ep1.experiments.runner \
    --scenarios jamming --supervisors none rule mpc --seeds 0-9 --c2 \
    --out experiments/results/local

# LLM, paced at 0.25 s per step, with GPU energy (NVIDIA machine)
experiments/run_llm_batch.sh check
experiments/run_llm_batch.sh start
experiments/run_llm_batch.sh status
```

Runs with an existing `summary.json` are skipped unless `--force` is given.

LLM requests pass the decoding options explicitly (the Ollama 0.6.2 sampling
defaults, `num_ctx` 4096) and a per-request seed of `1000 * run seed + query
index`. The 0.6.2 default context of 2048 tokens truncates the jamming prompt,
which is 2144 tokens when four agents are jammed.

## Outputs

Each run writes `<out>/<scenario>[_c2|_c2dNNN]/<supervisor>/seed_NN/`:

- `timeseries.npz`: per step time, J_n, r_n, positions, jammed flags,
  penetration depth, overridden agents, unreachable agents
- `decisions.jsonl`: one line per supervisor decision (text, parsed and
  delivered commands, and for the LLM latency, tokens, and Ollama durations)
- `summary.json`: the metrics below

`<out>/runs.jsonl` collects the summaries, and `<out>/environment.jsonl`
records the commit, Python, Ollama version, model digest, GPU, and arguments
of each batch.

Main metrics (mission phase = after formation convergence):

- `steps`, `success`: steps to reach the destination, within `MAX_ITER`
- `jn_min_mission`, `jn_mean_mission`: minimum and mean J_n
- `steps_below_pt`: steps with J_n below P_T = 0.94
- `recovery_steps`: steps from the first drop below P_T until J_n stays above
  P_T for 20 steps
- `min_clearance`: smallest distance from an agent to an obstacle edge
- `jammed_agent_steps`, `unreachable_agent_steps`: summed over agents
- `llm_latency_s`, `energy_above_idle_j`: per answered LLM query; energy is
  GPU board power from `nvidia-smi` at 100 ms, integrated over the request
  and minus the idle power measured before the batch

## Changes to the simulator

Relative to `2d-sim` at `d35e103`:

- `SwarmState.apply_obstacle_effects` recomputes the jamming penetration
  depth every step. Before, it kept the deepest value an agent ever reached, so
  link degradation did not recover after the agent left the field center.
- `LLMController` clears its commands when their lifetime expires, applies
  the avoidance commands it computes for nearby agents, and parses multi-word
  bearings such as "west by southwest" (before, they were read as "west").
- `LLMController` exposes hooks the harness needs: a replaceable clock, a
  result callback, and explicit Ollama options.
- `ControllerFactory` applies LLM overrides through one method instead of four
  copies of the same block.

Tests: `uv run pytest src/swarm_squad_ep1/tests/test_experiments.py`.
