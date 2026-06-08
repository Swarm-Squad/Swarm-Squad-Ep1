# Swarm Squad Ep1 Assignment (GUI-Only Teacher Manual)

This manual supports `docs/assignment-gui-student.md`.

## Instructional Intent

Students learn core cyber-physical concepts through observation and controlled GUI experiments, without requiring code.

## Target Concepts

Students should demonstrate understanding of:

1. mission control settings (formation/path/comm model),
2. jamming severity differences (`low_jam` vs `high_jam`),
3. spoofing modes (`phantom`, `coordinate`, `position_falsification`),
4. encryption impact on spoof-detection metrics.

## Suggested Timeline (50-70 minutes)

- 10 min: launch + panel walkthrough
- 15 min: baseline + jamming comparison
- 15 min: spoofing comparison
- 10 min: encryption comparison
- 5-20 min: debrief/reflection

## Key / Expected Patterns

### Pattern A - Low vs High Jamming

- higher jamming power usually degrades communication quality and mission coherence more strongly.

### Pattern B - Spoofing Mode Differences

- `phantom`: introduces fake agents/signals and perception confusion,
- `coordinate`: biases perceived coordinate frames,
- `position_falsification`: shifts perceived positions of real agents.

### Pattern C - Encryption ON vs OFF

- with encryption ON, spoofed/tampered messages should be rejected more effectively,
- `detection_rate` generally improves relative to encryption OFF.

## Evidence Checklist

Minimum acceptable evidence:

1. correct launch and GUI use,
2. completed scenario matrix with fixed-variable comparisons,
3. screenshots tied to each run,
4. reflection that connects observed behavior to metrics.

## Rubric (100 points)

- **Execution and workflow (20 pts)**: runs completed correctly in GUI.
- **Scenario quality (25 pts)**: valid comparisons with controlled changes.
- **Evidence quality (25 pts)**: clear screenshots and metric extraction.
- **Analysis depth (20 pts)**: correct interpretation of attack/defense effects.
- **Communication (10 pts)**: concise and organized submission.

## Common Issues

- Not resetting between scenarios:
  - require explicit reset before each run.
- Changing too many variables at once:
  - enforce one-variable-at-a-time comparisons.
- Overreliance on visuals:
  - require metric values in addition to screenshots.

## References

- `docs/assignment-gui-student.md`
- `docs/getting-started.md`
- `docs/algorithms-and-threat-model.md`
