"""
Aggregate seeded runs into per-configuration statistics.

Usage:
    uv run python -m swarm_squad_ep1.experiments.aggregate \
        experiments/results/local experiments/results/llm --out experiments/results
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from swarm_squad_ep1.experiments.runner import group_name

SUPERVISOR_ORDER = ["none", "rule", "mpc", "llm"]
GROUP_ORDER = ["obstacles_1", "obstacles_2", "jamming", "jamming_c2", "jamming_c2d050"]

RUN_FIELDS = [
    "group",
    "scenario",
    "supervisor",
    "seed",
    "success",
    "steps",
    "conv_step",
    "mission_steps",
    "wall_s",
    "pacing_overruns",
    "jn_min_mission",
    "jn_mean_mission",
    "steps_below_pt",
    "recovery_steps",
    "min_clearance",
    "jammed_agent_steps",
    "max_depth",
    "unreachable_agent_steps",
    "override_steps",
    "override_agent_steps",
    "n_decisions",
    "n_decisions_with_commands",
    "n_commands",
    "n_dropped_c2",
    "llm_queries",
    "llm_answered",
    "llm_errors",
    "llm_parse_success",
    "llm_valid_per_instruction",
]


def load_runs(dirs):
    runs = {}
    for d in dirs:
        index = Path(d) / "runs.jsonl"
        if not index.exists():
            continue
        for line in index.read_text().splitlines():
            run = json.loads(line)
            run["group"] = group_name(
                run["scenario"], run["c2_gate"], run.get("c2_depth") or 0.0
            )
            # A re-run with --force appends a newer line for the same run
            runs[(run["group"], run["supervisor"], run["seed"])] = run
    return list(runs.values())


def _values(runs, key):
    return np.array([r[key] for r in runs if r.get(key) is not None], dtype=float)


def mean_std(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": None, "std": None, "n": 0}
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return {"mean": float(values.mean()), "std": std, "n": int(values.size)}


def per_query(runs, key):
    """Pool a per-query statistic over runs from the decision logs."""
    values = []
    for r in runs:
        path = Path(r["run_dir"]) / "decisions.jsonl" if "run_dir" in r else None
        if path is None or not path.exists():
            continue
        for line in path.read_text().splitlines():
            d = json.loads(line)
            if d.get("kind") == "llm" and d.get(key) is not None and not d.get("error"):
                values.append(d[key])
    return np.array(values, dtype=float)


def summarize_group(runs):
    out = {"n": len(runs), "success": sum(bool(r["success"]) for r in runs)}
    for key in [
        "steps",
        "mission_steps",
        "jn_min_mission",
        "jn_mean_mission",
        "steps_below_pt",
        "min_clearance",
        "jammed_agent_steps",
        "override_agent_steps",
        "n_commands",
        "n_dropped_c2",
    ]:
        out[key] = mean_std(_values(runs, key))
    recovered = [r for r in runs if r.get("first_below") is not None]
    out["runs_below_pt"] = len(recovered)
    out["recovery_steps"] = mean_std(_values(recovered, "recovery_steps"))
    if runs and runs[0]["supervisor"] == "llm":
        out["llm_parse_success"] = mean_std(_values(runs, "llm_parse_success"))
        out["llm_answered"] = mean_std(_values(runs, "llm_answered"))
        for key, field in [
            ("latency_s", "latency_s"),
            ("prompt_tokens", "prompt_eval_count"),
            ("output_tokens", "eval_count"),
            ("energy_j", "energy_j"),
            ("energy_above_idle_j", "energy_above_idle_j"),
        ]:
            values = per_query(runs, field)
            out[f"query_{key}"] = {
                **mean_std(values),
                "median": float(np.median(values)) if values.size else None,
                "p95": float(np.percentile(values, 95)) if values.size else None,
            }
        steps = []
        for r in runs:
            path = Path(r["run_dir"]) / "decisions.jsonl"
            for line in path.read_text().splitlines():
                d = json.loads(line)
                if d.get("kind") == "llm" and d.get("step_response") is not None:
                    steps.append(d["step_response"] - d["step_request"])
        out["query_latency_steps"] = mean_std(steps)
        out["run_energy_j"] = mean_std(_values(runs, "gpu_energy_run_j"))
    else:
        compute = []
        for r in runs:
            stats = r.get("decision_compute_s") or {}
            if stats.get("mean") is not None:
                compute.append(stats["mean"])
        out["decision_compute_s"] = mean_std(compute)
    return out


def paired_difference(runs, group, a, b, key):
    """Per-seed difference a - b of a metric, over the seeds both have."""
    by_seed = defaultdict(dict)
    for r in runs:
        if r["group"] == group and r["supervisor"] in (a, b):
            by_seed[r["seed"]][r["supervisor"]] = r.get(key)
    diffs = [
        v[a] - v[b]
        for v in by_seed.values()
        if v.get(a) is not None and v.get(b) is not None
    ]
    out = mean_std(diffs)
    out["a_lower"] = int(sum(d < 0 for d in diffs))
    out["ties"] = int(sum(d == 0 for d in diffs))
    return out


def fmt(stat, digits=0):
    if stat["mean"] is None:
        return "--"
    return f"{stat['mean']:.{digits}f}$\\pm${stat['std']:.{digits}f}"


def latex_rows(summary):
    lines = []
    for group in GROUP_ORDER:
        if group not in summary:
            continue
        for sup in SUPERVISOR_ORDER:
            s = summary[group].get(sup)
            if s is None:
                continue
            lines.append(
                f"{group} & {sup} & {fmt(s['steps'])} & {fmt(s['jn_min_mission'], 3)}"
                f" & {fmt(s['steps_below_pt'])} & {fmt(s['n_commands'])} \\\\"
            )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    runs = load_runs(args.dirs)
    groups = defaultdict(lambda: defaultdict(list))
    for r in runs:
        groups[r["group"]][r["supervisor"]].append(r)

    summary = {
        g: {s: summarize_group(rs) for s, rs in sups.items()}
        for g, sups in groups.items()
    }
    comparisons = {}
    for g, sups in groups.items():
        for a in sups:
            for b in sups:
                if SUPERVISOR_ORDER.index(a) <= SUPERVISOR_ORDER.index(b):
                    continue
                for key in ("steps", "steps_below_pt", "jn_min_mission"):
                    comparisons[f"{g}/{a}-{b}/{key}"] = paired_difference(
                        runs, g, a, b, key
                    )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps({"groups": summary, "paired": comparisons}, indent=2)
    )
    with open(args.out / "runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(
            runs,
            key=lambda r: (
                GROUP_ORDER.index(r["group"]) if r["group"] in GROUP_ORDER else 99,
                SUPERVISOR_ORDER.index(r["supervisor"]),
                r["seed"],
            ),
        ):
            writer.writerow(r)
    (args.out / "table_rows.tex").write_text(latex_rows(summary) + "\n")

    for group in GROUP_ORDER:
        for sup in SUPERVISOR_ORDER:
            s = summary.get(group, {}).get(sup)
            if s is None:
                continue
            line = (
                f"{group:15s} {sup:5s} n={s['n']:2d} ok={s['success']:2d} "
                f"steps {fmt(s['steps'], 1)} Jn_min {fmt(s['jn_min_mission'], 3)} "
                f"below {fmt(s['steps_below_pt'], 1)} "
                f"recov {fmt(s['recovery_steps'], 1)} ({s['runs_below_pt']} runs) "
                f"cmds {fmt(s['n_commands'], 1)}"
            )
            if sup == "llm":
                lat = s["query_latency_s"]
                line += (
                    f" lat {lat['mean']:.2f}s" if lat["mean"] is not None else " lat --"
                )
            print(line.replace("$\\pm$", "+-"))


if __name__ == "__main__":
    main()
