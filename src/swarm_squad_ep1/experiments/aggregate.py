"""
Aggregate seeded runs into per-configuration statistics.

Usage:
    uv run python -m swarm_squad_ep1.experiments.aggregate \
        experiments/results/local experiments/results/llm --out experiments/results
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

from swarm_squad_ep1.experiments import compass
from swarm_squad_ep1.experiments.runner import group_name
from swarm_squad_ep1.experiments.supervisors import HINT_PATTERN, INSTRUCTION_PATTERN

SUPERVISOR_ORDER = ["none", "rule", "mpc", "llm"]
GROUP_ORDER = [
    "obstacles_1",
    "obstacles_2",
    "jamming",
    "jamming_c2",
    "jamming_c2d050",
    "jamming_blocking",
]

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
            folder = group_name(
                run["scenario"], run["c2_gate"], run.get("c2_depth") or 0.0
            )
            run["run_dir"] = str(
                Path(d) / folder / run["supervisor"] / f"seed_{run['seed']:02d}"
            )
            run["group"] = folder + ("_blocking" if run.get("blocking") else "")
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


STATE_HEADER = re.compile(r" - STATE DESCRIPTION \(Step (\d+)\):")


def hint_agreement(run_dir):
    """
    Compare delivered LLM commands with the evasion and avoidance hints in
    the telemetry of the same query (read back from supervisor.log).
    """
    hints, step = {}, None
    log = Path(run_dir) / "supervisor.log"
    if not log.exists():
        return None
    for line in log.read_text().splitlines():
        m = STATE_HEADER.search(line)
        if m:
            step = int(m.group(1))
            hints[step] = {}
            continue
        m = HINT_PATTERN.search(line)
        if m and step is not None:
            _, agent, distance, bearing = m.groups()
            hints[step][int(agent)] = (int(distance), bearing.lower())
    out = {"commands": 0, "to_hinted": 0, "hint_bearing": 0, "within_45": 0}
    out["hints_answered"] = 0
    for line in (Path(run_dir) / "decisions.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("kind") != "llm" or "step_parsed" not in d:
            continue
        query_hints = hints.get(d["step_request"], {})
        out["hints_answered"] += len(query_hints)
        if not d.get("delivered"):
            continue
        for agent, _, bearing in INSTRUCTION_PATTERN.findall(d.get("text") or ""):
            agent = int(agent)
            if agent not in d["delivered"]:
                continue
            out["commands"] += 1
            if agent not in query_hints:
                continue
            out["to_hinted"] += 1
            hinted = compass.BEARINGS[query_hints[agent][1]]
            diff = abs((compass.BEARINGS[bearing.lower()] - hinted + 180) % 360 - 180)
            out["hint_bearing"] += diff == 0
            out["within_45"] += diff <= 45
    out["hinted_queries"] = sum(1 for h in hints.values() if h)
    return out


def collision_agent_steps(run_dir):
    """Agent-steps spent inside a physical obstacle."""
    ts = np.load(Path(run_dir) / "timeseries.npz")
    obstacles = ts["obstacles"]
    if obstacles.size == 0:
        return 0
    pos = ts["positions"][:, :, None, :]
    dist = np.linalg.norm(pos - obstacles[None, None, :, :2], axis=-1)
    return int((dist < obstacles[None, None, :, 2]).any(axis=-1).sum())


def motion_stats(run_dir, conv_step):
    """Minimum inter-agent distance and steps of more than 5 units, after convergence."""
    pos = np.load(Path(run_dir) / "timeseries.npz")["positions"][conv_step or 0 :]
    diff = pos[:, :, None, :] - pos[:, None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    n = pos.shape[1]
    dist[:, np.arange(n), np.arange(n)] = np.inf
    moves = np.linalg.norm(np.diff(pos, axis=0), axis=-1)
    return float(dist.min()), int((moves > 5.0).sum())


def depth_stats(run_dir):
    """Agent-steps with penetration depth >= 0.5, and the maximum depth."""
    depth = np.load(Path(run_dir) / "timeseries.npz")["depth"]
    return int((depth >= 0.5).sum()), float(depth.max()) if depth.size else 0.0


def summarize_group(runs):
    out = {"n": len(runs), "success": sum(bool(r["success"]) for r in runs)}
    if runs and runs[0]["mode"] != "physical":
        depth = [depth_stats(r["run_dir"]) for r in runs]
        out["deep_agent_steps"] = mean_std([d[0] for d in depth])
        out["max_depth"] = mean_std([d[1] for d in depth])
    motion = [motion_stats(r["run_dir"], r["conv_step"]) for r in runs]
    out["min_separation"] = mean_std([m[0] for m in motion])
    out["min_separation_worst"] = float(min(m[0] for m in motion)) if motion else None
    out["jump_runs"] = sum(m[1] > 0 for m in motion)
    out["jump_steps"] = int(sum(m[1] for m in motion))
    if runs and runs[0]["mode"] == "physical":
        collisions = [collision_agent_steps(r["run_dir"]) for r in runs]
        out["collision_runs"] = sum(c > 0 for c in collisions)
        out["collision_agent_steps"] = mean_std(collisions)
        clear = [
            r["min_clearance"]["min"]
            if isinstance(r["min_clearance"], dict)
            else r["min_clearance"]
            for r in runs
        ]
        out["min_clearance_worst"] = float(min(clear))
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
                "max": float(values.max()) if values.size else None,
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
        agreement = [hint_agreement(r["run_dir"]) for r in runs]
        agreement = [a for a in agreement if a is not None]
        out["hint_agreement"] = (
            {k: sum(a[k] for a in agreement) for k in agreement[0]}
            if agreement
            else None
        )
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


def round_half_up(x, digits=0):
    quantum = Decimal(1).scaleb(-digits)
    return str(Decimal(repr(x)).quantize(quantum, rounding=ROUND_HALF_UP))


def fmt(stat, digits=0):
    if stat["mean"] is None:
        return "--"
    mean = round_half_up(stat["mean"], digits)
    std = round_half_up(stat["std"], digits)
    return f"{mean}$\\pm${std}"


GROUP_LABELS = {
    "obstacles_1": "Obstacles~I",
    "obstacles_2": "Obstacles~II",
    "jamming": "Jamming",
    "jamming_c2": "Jamming, jammed C2",
    "jamming_c2d050": "Jamming, jammed C2 for $d_{\\mathrm{pen}}\\geq0.5$",
    "jamming_blocking": "Jamming, LLM without latency",
}
SUPERVISOR_LABELS = {"none": "None", "rule": "Rule", "mpc": "MPC", "llm": "LLM"}


def latex_rows(summary, groups=GROUP_ORDER[:4], columns=5):
    """Body of the results table: one block of rows per scenario."""
    lines = []
    for group in groups:
        if group not in summary:
            continue
        lines.append(
            f"\\multicolumn{{{columns}}}{{|l|}}{{\\emph{{{GROUP_LABELS[group]}}}}} \\\\"
        )
        lines.append("\\hline")
        for sup in SUPERVISOR_ORDER:
            s = summary[group].get(sup)
            if s is None:
                continue
            lines.append(
                f"{SUPERVISOR_LABELS[sup]} & {fmt(s['steps'], 1)}"
                f" & {fmt(s['jn_min_mission'], 3)} & {fmt(s['steps_below_pt'], 1)}"
                f" & {fmt(s['n_commands'], 1)} \\\\"
            )
        lines.append("\\hline")
    return "\n".join(lines)


def latex_wide_rows(summary):
    """Body of the full-width table: one row per supervisor."""
    obstacle_cols = [("steps", 0), ("n_commands", 0)]
    jamming_cols = [
        ("steps", 0),
        ("steps_below_pt", 0),
        ("jn_min_mission", 3),
        ("n_commands", 0),
    ]
    layout = [
        ("obstacles_1", obstacle_cols),
        ("obstacles_2", obstacle_cols),
        ("jamming", jamming_cols),
        ("jamming_c2", jamming_cols),
    ]
    lines = []
    for sup in SUPERVISOR_ORDER:
        cells = [SUPERVISOR_LABELS[sup]]
        for group, cols in layout:
            s = summary.get(group, {}).get(sup)
            for key, digits in cols:
                cells.append(fmt(s[key], digits) if s else "--")
        lines.append(" & ".join(cells) + " \\\\")
    s = summary.get("jamming_blocking", {}).get("llm")
    if s:
        cells = ["LLM, no latency"] + ["--"] * 4
        cells += [fmt(s[key], digits) for key, digits in jamming_cols] + ["--"] * 4
        lines.append(" & ".join(cells) + " \\\\")
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
    (args.out / "table_wide_rows.tex").write_text(latex_wide_rows(summary) + "\n")

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
