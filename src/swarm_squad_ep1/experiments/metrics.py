"""
Per-run metrics computed from the recorded time series and decisions.
"""

import numpy as np

import swarm_squad_ep1.config as config

RECOVERY_HOLD = 20  # steps J_n must stay >= P_T to count as recovered


def threshold_stats(jn, pt=config.PT, hold=RECOVERY_HOLD) -> dict:
    """
    Steps below P_T, first such step, and recovery steps (from the first
    step below P_T to the start of the first stretch of `hold` steps at or
    above P_T, or of the remaining steps if fewer are left).
    """
    jn = np.asarray(jn, dtype=float)
    below = jn < pt
    out = {
        "steps_below_pt": int(below.sum()),
        "first_below": None,
        "recovery_steps": None,
    }
    if not below.any():
        return out
    first = int(np.argmax(below))
    out["first_below"] = first
    for k in range(first, len(jn)):
        window = jn[k : k + hold]
        if (
            window.size
            and np.all(window >= pt)
            and (window.size == hold or k + window.size == len(jn))
        ):
            out["recovery_steps"] = k - first
            break
    return out


def min_clearance(positions, obstacles) -> float:
    """Minimum distance from any agent to any obstacle boundary (negative = inside)."""
    if len(obstacles) == 0:
        return float("nan")
    obstacles = np.asarray(obstacles, dtype=float)
    d = (
        np.linalg.norm(positions[:, :, None, :] - obstacles[None, None, :, :2], axis=-1)
        - obstacles[None, None, :, 2]
    )
    return float(d.min())


def _stats(values) -> dict:
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if values.size == 0:
        return {"mean": None, "median": None, "p95": None, "max": None, "n": 0}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "n": int(values.size),
    }


def summarize(ts: dict, decisions: list, run: dict, power=None, idle_w=None) -> dict:
    """
    Args:
        ts: time series (jn, rn, positions, jammed, depth, applied, unreachable)
        decisions: supervisor decision records
        run: run settings and outcome (success, steps, conv_step, ...)
        power: PowerMonitor or None
        idle_w: idle GPU power for the above-idle energy

    Returns:
        JSON-serializable summary
    """
    conv = run["conv_step"]
    jn = np.asarray(ts["jn"], dtype=float)
    mission = jn[conv:] if conv is not None else np.array([])
    summary = dict(run)

    summary["jn_final"] = float(jn[-1]) if jn.size else None
    summary["rn_final"] = float(ts["rn"][-1]) if len(ts["rn"]) else None
    if mission.size:
        summary["jn_min_mission"] = float(mission.min())
        summary["jn_mean_mission"] = float(mission.mean())
        summary.update(threshold_stats(mission))
    else:
        summary.update(
            {
                "jn_min_mission": None,
                "jn_mean_mission": None,
                "steps_below_pt": None,
                "first_below": None,
                "recovery_steps": None,
            }
        )

    positions = np.asarray(ts["positions"])
    obstacles = np.asarray(run["obstacles"])
    if run["mode"] == config.ObstacleMode.HARD.value:
        summary["min_clearance"] = min_clearance(positions, obstacles)
    else:
        summary["min_clearance"] = None
    jammed = np.asarray(ts["jammed"], dtype=bool)
    depth = np.asarray(ts["depth"], dtype=float)
    summary["jammed_agent_steps"] = int(jammed.sum())
    summary["max_depth"] = float(depth.max()) if depth.size else 0.0
    summary["unreachable_agent_steps"] = int(np.asarray(ts["unreachable"]).sum())

    applied = np.asarray(ts["applied"], dtype=bool)
    summary["override_steps"] = int(applied.any(axis=1).sum()) if applied.size else 0
    summary["override_agent_steps"] = int(applied.sum())

    parsed = [d for d in decisions if "step_parsed" in d]
    summary["n_decisions"] = len(decisions)
    summary["n_decisions_with_commands"] = sum(1 for d in parsed if d["delivered"])
    summary["n_commands"] = sum(len(d["delivered"]) for d in parsed)
    summary["n_dropped_c2"] = sum(len(d["dropped_c2"]) for d in parsed)

    llm = [d for d in decisions if d.get("kind") == "llm"]
    if llm:
        answered = [d for d in llm if d.get("t_end") is not None]
        errors = [d for d in answered if d.get("error")]
        ok = [d for d in answered if not d.get("error")]
        summary["llm_queries"] = len(llm)
        summary["llm_answered"] = len(answered)
        summary["llm_errors"] = len(errors)
        # An answer that arrives as the run ends is never parsed
        parsed_ok = [d for d in ok if "step_parsed" in d]
        summary["llm_parsed"] = len(parsed_ok)
        summary["llm_latency_s"] = _stats([d["latency_s"] for d in ok])
        summary["llm_latency_steps"] = _stats(
            [d["step_response"] - d["step_request"] for d in ok]
        )
        summary["llm_prompt_tokens"] = _stats([d.get("prompt_eval_count") for d in ok])
        summary["llm_output_tokens"] = _stats([d.get("eval_count") for d in ok])
        summary["llm_parse_success"] = (
            sum(1 for d in parsed_ok if d["n_valid"] > 0) / len(parsed_ok)
            if parsed_ok
            else None
        )
        summary["llm_valid_per_instruction"] = (
            sum(d["n_valid"] for d in parsed_ok)
            / max(1, sum(d["n_instructions"] for d in parsed_ok))
            if parsed_ok
            else None
        )
        if power is not None:
            for d in ok:
                d["energy_j"] = power.energy(d["t_start"], d["t_end"])
                if idle_w is not None:
                    d["energy_above_idle_j"] = power.energy(
                        d["t_start"], d["t_end"], idle_w
                    )
            summary["llm_energy_j"] = _stats([d.get("energy_j") for d in ok])
            summary["llm_energy_above_idle_j"] = _stats(
                [d.get("energy_above_idle_j") for d in ok]
            )
    else:
        decide = [d.get("compute_s") for d in decisions if "compute_s" in d]
        summary["decision_compute_s"] = _stats(decide)
    return summary
