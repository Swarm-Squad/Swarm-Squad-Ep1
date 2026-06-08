# Swarm Squad Ep1 Assignment (Python + GUI Student Version)

## Assignment Title

**AAM Swarm Resilience Lab: Scripted Attack/Defense Experimentation**

## Audience

This track is for students who can write and run Python scripts.

## Learning Goals

By the end of this assignment, you should be able to:

1. run GUI and Python script workflows at the same time,
2. configure attacks/defenses from code using `SwarmSquadClient`,
3. compare outcomes across scripted scenario variants,
4. justify findings using both metrics and visuals.

## Two-Terminal Workflow (Required)

### Terminal A (keep running)

```bash
uv run swarm-squad-ep1
```

### Browser

- `http://localhost:5000`

### Terminal B (run your script)

```bash
uv run python student_assignment.py
```

Starter option:

```bash
uv run python examples/ep1_assignment_starter.py
```

Then copy/adapt it into your own `student_assignment.py`.

## Required Tasks

### Task 1 - Baseline Script

Create `student_assignment.py` that:

1. resets simulation,
2. clears zones,
3. sets formation/path algorithm,
4. starts simulation,
5. prints `simulation_state()`.

### Task 2 - Attack Matrix (Scripted)

Implement at least 4 scripted runs:

1. baseline,
2. low-power jamming,
3. spoofing (`phantom` or `coordinate` or `position_falsification`),
4. spoofing + encryption (`hmac_sha256`).

For each run, collect:

- `attack_metrics()`,
- `protocol_stats()`,
- one GUI screenshot.

### Task 3 - Knob Coverage

In your script, demonstrate at least 6 controls:

- add/remove/update agent,
- add/update/delete jamming zone,
- add/update/delete spoofing zone,
- formation/path switch,
- comm model switch,
- encryption toggle/algorithm select,
- LLM assistance toggle.

### Task 4 - Custom Algorithm Extension

Choose one:

- register + use a custom path algorithm, or
- register + use a custom crypto algorithm.

You must show it appears in runtime options and is actively used.

### Task 5 - Analysis Write-up

Write 300-500 words:

- compare at least two attack methods,
- explain the effect of defense settings,
- connect metrics to observed GUI behavior.

## Submission Checklist

1. `student_assignment.py`
2. scenario results artifact (CSV/JSON or table)
3. screenshots (minimum 4)
4. analysis write-up

## References

- `examples/ep1_assignment_starter.py` (starter template)
- `docs/script-customization.md`
- `docs/client-api-reference.md`
- `examples/ep1_custom_control_loop.py` (advanced reference)
