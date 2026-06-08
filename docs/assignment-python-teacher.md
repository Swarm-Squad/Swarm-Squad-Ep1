# Swarm Squad Ep1 Assignment (Python + GUI Teacher Manual)

This manual supports `docs/assignment-python-student.md`.

## Instructional Intent

Students use scripted control to run reproducible cyber-physical experiments and interpret outcomes with both telemetry and visualization.

## Target Competencies

Students should demonstrate:

1. two-terminal runtime proficiency (GUI + script),
2. correct use of `SwarmSquadClient` control surface,
3. controlled scenario comparisons with metric evidence,
4. custom extension usage (path or crypto plugin).

## Suggested Timeline (75-100 minutes)

- 10 min: architecture + API review
- 15 min: baseline scripting setup
- 20 min: attack matrix implementation
- 20 min: defense/custom extension
- 10-35 min: analysis and debrief

## Key / Expected Results

### Pattern A - Scripted Repeatability

- repeated runs with same settings should produce similar qualitative outcomes.

### Pattern B - Attack Method Contrast

- jamming-heavy runs should emphasize communication degradation,
- spoofing-heavy runs should more strongly impact spoof-detection metrics.

### Pattern C - Encryption Effects

- enabling crypto should improve spoofing detection trends relative to crypto-off baselines.

### Pattern D - Custom Algorithm Integration

- custom path/crypto registration should succeed,
- selected custom algorithm should appear in runtime config and active status.

## Evidence Checklist

Required evidence:

1. runnable script with clear scenario sections,
2. extracted metrics per scenario,
3. screenshots matching scenario log,
4. written comparison tied to evidence.

Suggested scaffold:

- `examples/ep1_assignment_starter.py` for baseline classroom submissions.
- `examples/ep1_custom_control_loop.py` as advanced extension reference.

## Rubric (100 points)

- **Runtime and code execution (20 pts)**: script runs with GUI concurrently.
- **API/control correctness (25 pts)**: proper use of required knobs/methods.
- **Experiment design (20 pts)**: meaningful scenario comparisons.
- **Analysis and interpretation (25 pts)**: evidence-backed conclusions.
- **Code/report quality (10 pts)**: clarity and reproducibility.

## Common Issues

- Script modifies too many variables at once:
  - require scenario helper function with explicit parameter set.
- Missing cleanup between scenarios:
  - require `reset_simulation()` and zone clears.
- Custom registration fails:
  - check import path format `module:function` and callable signatures.

## References

- `docs/assignment-python-student.md`
- `examples/ep1_assignment_starter.py`
- `docs/script-customization.md`
- `docs/client-api-reference.md`
- `examples/ep1_custom_control_loop.py`
