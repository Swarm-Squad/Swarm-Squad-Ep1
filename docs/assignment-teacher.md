# Swarm Squad Ep1 Classroom Assignment (Teacher Version)

This file contains instructor materials, expected outcomes, and assessment guidance
for the student handout in `docs/assignment-student.md`.

## Instructor Intent

This lab is designed for introductory engineering students to connect:

- swarm autonomy,
- cybersecurity concepts (spoofing/jamming),
- AI-assisted control,
- software interface skills (GUI + Python API).

## Core Concepts Students Should Understand

By completion, students should understand:

1. **System architecture**: one backend, two interfaces (GUI + script),
2. **Threat models**: spoofing/falsification vs communication degradation,
3. **Defense mechanisms**: cryptographic auth and communication model choices,
4. **Experiment method**: controlled scenario comparison using metrics.

## Instructor Setup Checklist

Before class:

1. verify environment setup:
   - `uv sync --extra dev`
   - dependencies up (`docker compose up -d`, `ollama serve` if needed),
2. smoke check:
   - `uv run swarm-squad-ep1`
   - open `http://localhost:5000`,
3. quick script test:
   - `uv run python examples/ep1_custom_control_loop.py`.

## Suggested Class Timeline (75-90 min)

- 10 min: architecture and controls walkthrough,
- 15 min: launch/runtime setup support,
- 20 min: baseline + scenario matrix runs,
- 20 min: scripted custom controls,
- 10-15 min: debrief and reflection discussion.

## Key / Expected Patterns

Use these as interpretation anchors (not strict numeric grading targets):

### Scenario Pattern A - Spoofing active, crypto off

- expected trend: `detection_rate` near zero,
- expected trend: false negatives (`fn`) increase,
- GUI may show phantom/falsified influence while traffic remains accepted.

### Scenario Pattern B - Spoofing active, crypto on (built-in algo)

- expected trend: `tp` increases relative to Pattern A,
- expected trend: `detection_rate` improves vs crypto off,
- expected trend: `fn` decreases vs crypto off.

### Scenario Pattern C - Jamming emphasis

- expected trend: communication quality degrades,
- protocol/mission quality may degrade even with crypto on,
- students should explain that crypto mitigates forged/tampered messages but does
  not remove RF jamming effects.

### Scenario Pattern D - Custom algorithm registration

- students should show successful runtime registration and selection,
- algorithm should appear in script-driven config and in runtime option lists.

## Evidence Key (What to Look For)

Minimum evidence of mastery:

1. two-terminal workflow correctly used (`swarm-squad-ep1` + script),
2. script demonstrates control of multiple simulator knobs,
3. reported metrics correspond to scenario differences,
4. reflection links observed GUI behavior with metric changes.

## Evaluation Rubric (100 points)

- **Environment and execution (20 pts)**
  - runtime launches, GUI reachable, script runs concurrently.
- **Scenario design and controls (25 pts)**
  - baseline + required comparisons, correct API usage.
- **Metrics and analysis (25 pts)**
  - correct capture of attack/protocol metrics and reasoned interpretation.
- **Custom extension (15 pts)**
  - working custom path or custom crypto registration/use.
- **Communication quality (15 pts)**
  - clear reflection, reproducible commands, organized submission.

## Common Pitfalls and Fixes

- `Connection refused` in script:
  - runtime is not running; start `uv run swarm-squad-ep1` first.
- GUI not updating during script run:
  - verify script points to active backend and GUI is open at `http://localhost:5000`.
- No algorithm appears after custom registration:
  - check import path format (`module:function`), then re-run registration call.
- Confusing attack results:
  - remind students to compare trends across scenarios, not exact one-run values.

## Optional Advanced Extension

For advanced students:

- compare `legacy` vs `v2v_channel` comm models under identical attack setup,
- run a small repeatability loop with different seeds and summarize variance.

## Related Docs

- `docs/assignment-student.md`
- `docs/getting-started.md`
- `docs/script-customization.md`
- `docs/client-api-reference.md`
- `examples/ep1_custom_control_loop.py`
