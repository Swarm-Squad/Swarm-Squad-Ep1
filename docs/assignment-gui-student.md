# Swarm Squad Ep1 Assignment (GUI-Only Student Version)

## Assignment Title

**AAM Swarm Resilience Lab: Visual Cyber Defense Analysis**

## Audience

This track is for students who are not using Python scripting yet.

## Learning Goals

By the end of this assignment, you should be able to:

1. launch and use the Swarm Squad Ep1 web GUI,
2. configure attacks and defenses from the right-side control panels,
3. compare mission outcomes using GUI metrics,
4. explain how settings affect swarm behavior.

## Setup and Launch

From the project root:

```bash
uv run swarm-squad-ep1
```

Open:

- `http://localhost:5000`

## Required Tasks

### Task 1 - Baseline Mission

In the GUI:

1. choose formation/path/comm model,
2. keep jamming and spoofing zones off,
3. start simulation and observe mission behavior.

Record:

- one screenshot,
- high-level notes on formation quality and movement.

### Task 2 - Jamming Comparison

Run two scenarios:

1. **low-power jamming** (`low_jam`),
2. **high-power jamming** (`high_jam`).

Keep all other settings the same.

Record for each run:

- communication quality trend (from panel data),
- protocol stats summary,
- one screenshot.

### Task 3 - Spoofing Comparison

Run three spoofing scenarios:

1. `phantom`,
2. `coordinate`,
3. `position_falsification`.

Keep all other settings fixed.

Record:

- visible behavior changes,
- attack metrics panel values (`tp`, `fp`, `fn`, `tn`, `detection_rate`),
- one screenshot per scenario.

### Task 4 - Defense Comparison

For one spoofing scenario, compare:

1. encryption OFF,
2. encryption ON (`hmac_sha256`).

Record:

- what changed in attack metrics,
- what changed visually in the swarm behavior.

### Task 5 - Reflection

Write a short response (200-350 words):

- Which attack type caused the biggest mission impact?
- Which defense setting helped most?
- What did the metrics show that the visualization alone did not?

## Submission Checklist

Submit:

1. screenshot set (at least 6),
2. a short scenario table (settings + observed outcomes),
3. reflection write-up.

## Useful References

- `docs/getting-started.md`
- `docs/algorithms-and-threat-model.md`
