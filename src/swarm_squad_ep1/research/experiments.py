"""
Experiment matrices.

Each experiment is a callable that returns a list of ``Scenario`` objects.
``run_experiment(name, out_dir, seeds, max_steps)`` executes them, writes a
CSV + ``summary.json``, and returns the list of ``Result`` objects.

Experiments
-----------
E1  Single-vs-dual attack susceptibility
    Proves jamming + spoofing together is worse than either alone.

E2  LLM assistance under dual attack
    Proves LLM guidance always improves mission success under combined
    jamming + spoofing.

E3  Path planning algorithm comparison
    Compares all grid/direct algorithms under combined attack.

E4  Cryptographic authentication comparison
    Compares HMAC-SHA256 vs ChaCha20-Poly1305 vs AES-256-CTR under
    spoofing, plus a no-crypto baseline.

E5  Full factorial (attack × LLM × crypto)
    The main evidence table: every combination of attack presence,
    LLM on/off, crypto on/off.

E6  Communication model comparison
    V2V channel vs legacy under varying jamming intensity.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Callable

from .runner import Result, run_scenario
from .scenarios import (
    Scenario,
    baseline_scenario,
    combined_scenario,
    jamming_scenario,
    spoofing_scenario,
)


# --------------------------------------------------------------------------
# Experiment builders
# --------------------------------------------------------------------------

def _e1_single_vs_dual(seeds: int) -> list[Scenario]:
    """Jamming only, spoofing only, and combined — no LLM, no crypto.

    This isolates the attack effect: combined should degrade performance
    more than either attack alone.
    """
    out: list[Scenario] = []
    for seed in range(seeds):
        out.append(baseline_scenario(seed=seed, name="E1_baseline"))

        for jam in ("low_jam", "high_jam"):
            sc = jamming_scenario(jam_type=jam, seed=seed, llm=False)
            sc.name = f"E1_jam_{jam}"
            out.append(sc)

        for spoof in ("phantom", "position_falsification", "coordinate"):
            sc = spoofing_scenario(spoof_type=spoof, crypto=False, seed=seed, llm=False)
            sc.name = f"E1_spoof_{spoof}"
            out.append(sc)

        for jam in ("low_jam", "high_jam"):
            for spoof in ("phantom", "position_falsification", "coordinate"):
                sc = combined_scenario(
                    jam_type=jam, spoof_type=spoof,
                    crypto=False, llm=False, seed=seed,
                )
                sc.name = f"E1_combo_{jam}_{spoof}"
                out.append(sc)

    return out


def _e2_llm_under_dual(seeds: int) -> list[Scenario]:
    """LLM on vs off under every combined attack configuration.

    Each pair shares the same seed so improvements are directly comparable.
    """
    out: list[Scenario] = []
    for seed in range(seeds):
        for jam in ("low_jam", "high_jam"):
            for spoof in ("phantom", "position_falsification", "coordinate"):
                for llm in (False, True):
                    sc = combined_scenario(
                        jam_type=jam, spoof_type=spoof,
                        crypto=False, llm=llm, seed=seed,
                    )
                    sc.name = f"E2_{jam}_{spoof}_llm={int(llm)}"
                    out.append(sc)

        for llm in (False, True):
            sc = baseline_scenario(seed=seed, name=f"E2_baseline_llm={int(llm)}")
            sc.llm_assistance_enabled = llm
            out.append(sc)

    return out


def _e3_path_algorithms(seeds: int) -> list[Scenario]:
    """Compare path planning algorithms under combined high-jam + phantom."""
    algos = ["direct", "astar", "theta_star", "dijkstra", "bfs", "bi_astar"]
    out: list[Scenario] = []
    for seed in range(seeds):
        for algo in algos:
            sc = combined_scenario(
                jam_type="high_jam", spoof_type="phantom",
                crypto=True, llm=False, seed=seed,
            )
            sc.path_algorithm = algo
            sc.name = f"E3_path_{algo}"
            out.append(sc)

        for algo in algos:
            sc = baseline_scenario(seed=seed, name=f"E3_baseline_{algo}")
            sc.path_algorithm = algo
            out.append(sc)

    return out


def _e4_crypto_comparison(seeds: int) -> list[Scenario]:
    """Compare authentication methods under each spoofing type."""
    crypto_algos = ["hmac_sha256", "chacha20_poly1305", "aes_256_ctr"]
    out: list[Scenario] = []
    for seed in range(seeds):
        for spoof in ("phantom", "position_falsification", "coordinate"):
            sc = spoofing_scenario(
                spoof_type=spoof, crypto=False, seed=seed, llm=False,
            )
            sc.name = f"E4_spoof_{spoof}_nocrypto"
            out.append(sc)

            for algo in crypto_algos:
                sc = spoofing_scenario(
                    spoof_type=spoof, crypto=True,
                    crypto_algorithm=algo, seed=seed, llm=False,
                )
                sc.name = f"E4_spoof_{spoof}_{algo}"
                out.append(sc)

        for spoof in ("phantom", "position_falsification", "coordinate"):
            for jam in ("low_jam", "high_jam"):
                sc = combined_scenario(
                    jam_type=jam, spoof_type=spoof,
                    crypto=False, llm=False, seed=seed,
                )
                sc.name = f"E4_combo_{jam}_{spoof}_nocrypto"
                out.append(sc)

                for algo in crypto_algos:
                    sc = combined_scenario(
                        jam_type=jam, spoof_type=spoof,
                        crypto=True, crypto_algorithm=algo,
                        llm=False, seed=seed,
                    )
                    sc.name = f"E4_combo_{jam}_{spoof}_{algo}"
                    out.append(sc)

    return out


def _e5_full_factorial(seeds: int) -> list[Scenario]:
    """Full evidence table: attack × LLM × crypto.

    Rows cover: no-attack baseline, jam-only, spoof-only, combined,
    each toggled with LLM and crypto on/off.
    """
    out: list[Scenario] = []
    for seed in range(seeds):
        for llm in (False, True):
            for crypto in (False, True):
                tag = f"l{int(llm)}_c{int(crypto)}"

                sc = baseline_scenario(seed=seed, name=f"E5_baseline_{tag}")
                sc.llm_assistance_enabled = llm
                sc.crypto_enabled = crypto
                out.append(sc)

                for jam in ("low_jam", "high_jam"):
                    sc = jamming_scenario(jam_type=jam, seed=seed, llm=llm)
                    sc.crypto_enabled = crypto
                    sc.name = f"E5_jam_{jam}_{tag}"
                    out.append(sc)

                for spoof in ("phantom", "position_falsification", "coordinate"):
                    sc = spoofing_scenario(
                        spoof_type=spoof, crypto=crypto, seed=seed, llm=llm,
                    )
                    sc.name = f"E5_spoof_{spoof}_{tag}"
                    out.append(sc)

                for jam in ("low_jam", "high_jam"):
                    for spoof in ("phantom", "position_falsification", "coordinate"):
                        sc = combined_scenario(
                            jam_type=jam, spoof_type=spoof,
                            crypto=crypto, llm=llm, seed=seed,
                        )
                        sc.name = f"E5_combo_{jam}_{spoof}_{tag}"
                        out.append(sc)

    return out


def _e6_comm_model(seeds: int) -> list[Scenario]:
    """V2V vs legacy channel model under varying jamming."""
    out: list[Scenario] = []
    for seed in range(seeds):
        for model in ("v2v", "legacy"):
            for jam in ("none", "low_jam", "high_jam"):
                if jam == "none":
                    s = baseline_scenario(seed=seed, name=f"E6_comm_{model}_nojam")
                else:
                    s = jamming_scenario(jam_type=jam, seed=seed, llm=False)
                    s.name = f"E6_comm_{model}_{jam}"
                s.comm_model = model
                out.append(s)
    return out


EXPERIMENTS: dict[str, Callable[[int], list[Scenario]]] = {
    "E1": _e1_single_vs_dual,
    "E2": _e2_llm_under_dual,
    "E3": _e3_path_algorithms,
    "E4": _e4_crypto_comparison,
    "E5": _e5_full_factorial,
    "E6": _e6_comm_model,
}


# --------------------------------------------------------------------------
# Runner + writers
# --------------------------------------------------------------------------

RESULT_COLUMNS = [
    "experiment", "scenario_name", "seed",
    "comm_model", "path_algorithm", "formation_type",
    "llm_enabled", "crypto_enabled", "crypto_algorithm",
    "jamming_types", "spoofing_types",
    "destination_reached", "steps_to_destination", "total_steps",
    "duration_s",
    "final_Jn", "avg_Jn", "avg_comm_quality", "total_path_length",
    "packet_loss_rate",
    "spoof_tp", "spoof_fp", "spoof_fn", "spoof_tn",
    "detection_rate", "false_positive_rate", "precision",
    "llm_calls", "llm_parse_success", "llm_parse_fail",
    "llm_repair_success", "llm_fallback_used",
]


def _row_from_result(experiment: str, scenario: Scenario, res: Result) -> dict:
    jt = ",".join(z.obstacle_type for z in scenario.jamming_zones) or "-"
    st = ",".join(z.spoof_type for z in scenario.spoofing_zones) or "-"
    return {
        "experiment": experiment,
        "scenario_name": scenario.name,
        "seed": scenario.seed,
        "comm_model": scenario.comm_model,
        "path_algorithm": scenario.path_algorithm,
        "formation_type": scenario.formation_type,
        "llm_enabled": scenario.llm_assistance_enabled,
        "crypto_enabled": scenario.crypto_enabled,
        "crypto_algorithm": scenario.crypto_algorithm if scenario.crypto_enabled else "-",
        "jamming_types": jt,
        "spoofing_types": st,
        "destination_reached": res.destination_reached,
        "steps_to_destination": res.steps_to_destination,
        "total_steps": res.total_steps,
        "duration_s": res.duration_s,
        "final_Jn": res.final_Jn,
        "avg_Jn": res.avg_Jn,
        "avg_comm_quality": res.avg_comm_quality,
        "total_path_length": res.total_path_length,
        "packet_loss_rate": res.packet_loss_rate,
        "spoof_tp": res.spoof_tp,
        "spoof_fp": res.spoof_fp,
        "spoof_fn": res.spoof_fn,
        "spoof_tn": res.spoof_tn,
        "detection_rate": res.detection_rate,
        "false_positive_rate": res.false_positive_rate,
        "precision": res.precision,
        "llm_calls": res.llm_calls,
        "llm_parse_success": res.llm_parse_success,
        "llm_parse_fail": res.llm_parse_fail,
        "llm_repair_success": res.llm_repair_success,
        "llm_fallback_used": res.llm_fallback_used,
    }


def run_experiment(
    name: str,
    out_dir: str | Path = "results",
    seeds: int = 3,
    max_steps: int | None = None,
    keep_trace: bool = False,
    verbose: bool = True,
) -> list[Result]:
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}. Options: {list(EXPERIMENTS)}")

    scenarios = EXPERIMENTS[name](seeds)
    if max_steps is not None:
        for s in scenarios:
            s.max_steps = max_steps

    out_root = Path(out_dir) / name
    out_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = out_root / f"{ts}.csv"
    summary_path = out_root / f"{ts}_summary.json"

    results: list[Result] = []
    total = len(scenarios)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for i, sc in enumerate(scenarios):
            if verbose:
                print(f"[{name}] {i+1}/{total} {sc.name} seed={sc.seed}")
            res = run_scenario(sc, keep_trace=keep_trace, verbose=False)
            results.append(res)
            writer.writerow(_row_from_result(name, sc, res))
            fh.flush()

    bucket: dict[tuple, list[Result]] = {}
    for sc, res in zip(scenarios, results):
        key = (
            sc.name.rsplit("_", 1)[0] if sc.name.startswith("E") else sc.name,
            sc.llm_assistance_enabled,
            sc.crypto_enabled,
            sc.comm_model,
            sc.path_algorithm,
            sc.crypto_algorithm if sc.crypto_enabled else "-",
        )
        bucket.setdefault(key, []).append(res)

    summary = []
    for (prefix, llm, crypto, comm, path_algo, crypto_algo), group in sorted(bucket.items()):
        n = len(group)
        success = sum(1 for r in group if r.destination_reached) / max(1, n)
        summary.append({
            "prefix": prefix,
            "llm_enabled": llm,
            "crypto_enabled": crypto,
            "crypto_algorithm": crypto_algo,
            "comm_model": comm,
            "path_algorithm": path_algo,
            "n": n,
            "success_rate": round(success, 3),
            "avg_final_Jn": round(sum(r.final_Jn for r in group) / n, 4),
            "avg_comm_quality": round(sum(r.avg_comm_quality for r in group) / n, 4),
            "avg_steps": round(sum(r.total_steps for r in group) / n, 1),
            "avg_path_length": round(sum(r.total_path_length for r in group) / n, 2),
            "avg_packet_loss": round(sum(r.packet_loss_rate for r in group) / n, 4),
            "avg_detection_rate": round(sum(r.detection_rate for r in group) / n, 4),
            "avg_fpr": round(sum(r.false_positive_rate for r in group) / n, 4),
            "avg_precision": round(sum(r.precision for r in group) / n, 4),
            "avg_llm_parse_fail_rate": round(
                sum(r.llm_parse_fail / max(1, r.llm_calls) for r in group) / n, 4,
            ),
        })

    with open(summary_path, "w") as fh:
        json.dump({"experiment": name, "timestamp": ts, "summary": summary}, fh, indent=2)

    if verbose:
        print(f"\n[{name}] {total} scenarios complete")
        print(f"  CSV:     {csv_path}")
        print(f"  Summary: {summary_path}")

    return results
