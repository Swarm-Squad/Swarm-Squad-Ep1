# Swarm Squad Ep1 Classroom Assignment (Student Version)

## Assignment Title

**AAM Swarm Resilience Lab: Cyber Attacks, Defenses, and Scripted Control**

## Context

In this lab, you will use `swarm_squad_ep1` to explore advanced air mobility (AAM)
swarm behavior under communication attacks and defensive controls. You will operate
the simulator from both:

- the web GUI (`http://localhost:5000`),
- your own Python script (`SwarmSquadClient`).

## Learning Goals

By the end of this assignment, you should be able to:

1. run the simulator runtime and a custom script at the same time,
2. control agents, zones, and algorithms from Python,
3. compare attack/defense outcomes using simulator metrics,
4. explain how crypto and comm-model choices affect swarm behavior.

## Launch Workflow (GUI + Script at the Same Time)

From the project root, use two terminals.

### Terminal A (keep running)

```bash
uv run swarm-squad-ep1
```

This starts:

- GUI/chat service on `http://localhost:5000`,
- simulation API on `http://localhost:5001`.

### Browser

Open:

- `http://localhost:5000`

### Terminal B (run your script while Terminal A is still running)

```bash
uv run python student_assignment.py
```

Your script updates should appear live in the GUI.

## Required Tasks

### Task 1 - Baseline Run

Create `student_assignment.py` and run a baseline mission:

- reset simulation,
- clear jamming/spoofing zones,
- choose one formation and one path algorithm,
- start simulation and print `simulation_state()`.

### Task 2 - Attack vs Defense Comparison

Run at least **3 scenarios** and record metrics:

1. baseline (no spoofing zone),
2. spoofing active + crypto off,
3. spoofing active + crypto on (`hmac_sha256`).

For each scenario, collect:

- `attack_metrics()` (`tp`, `fp`, `fn`, `tn`, `detection_rate`),
- `protocol_stats()` (message counters),
- one screenshot from the GUI.

### Task 3 - Scripted Knob Control

In your script, demonstrate at least **5** of these controls:

- add/remove agent,
- add/update/delete jamming zone,
- add/update/delete spoofing zone,
- switch formation/path algorithm,
- toggle comm model (`legacy` vs `v2v_channel`),
- toggle crypto on/off,
- toggle LLM assistance,
- run `run_script_control_loop(...)`.

### Task 4 - Custom Extension (Choose One)

Choose one extension:

- register and use a custom path algorithm, or
- register and use a custom crypto algorithm.

Use runtime registration methods from `SwarmSquadClient` and show that the algorithm
is selectable/active during your run.

### Task 5 - Reflection

Write a short reflection (250-400 words):

- Which setting had the largest impact on mission behavior?
- What changed when crypto was enabled?
- What did you observe in detection metrics vs visible GUI behavior?

## Submission Checklist

Submit:

1. `student_assignment.py`,
2. one CSV or JSON metric export (optional helper: `download_simulation_results(...)`),
3. 3 scenario screenshots from GUI,
4. reflection write-up.

## Related Docs

- `docs/getting-started.md`
- `docs/script-customization.md`
- `docs/client-api-reference.md`
- `examples/ep1_custom_control_loop.py`
