"""
Headless runner for the supervisor comparison.

Examples:
    # Non-LLM supervisors, all scenarios, seeds 0-9 (unpaced)
    python -m swarm_squad_ep1.experiments.runner \
        --scenarios obstacles_1 obstacles_2 jamming --supervisors none rule mpc \
        --seeds 0-9 --out experiments/results/local

    # LLM supervisor, paced at 0.25 s per step, with GPU power sampling
    python -m swarm_squad_ep1.experiments.runner \
        --scenarios jamming --supervisors llm --seeds 0-9 --paced --gpu-power \
        --out experiments/results/llm
"""

import argparse
import contextlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

import swarm_squad_ep1.config as config
from swarm_squad_ep1.controllers.controller_factory import ControllerType
from swarm_squad_ep1.experiments import metrics, power
from swarm_squad_ep1.experiments.scenario import REPO_ROOT, Scenario
from swarm_squad_ep1.experiments.supervisors import SUPERVISORS, ExperimentFactory
from swarm_squad_ep1.models.swarm_state import SwarmState

T_STEP = 0.25  # s per control step, the GUI rate of the CCNC runs
VIRTUAL_T0 = 1.0e6  # origin of the virtual clock for unpaced runs
LLM_MODEL = "llama3.2:3b-instruct-q4_K_M"
LLM_OPTIONS = {
    # Ollama 0.6.2 sampling defaults, passed explicitly so that the server
    # version does not change them
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    # The 0.6.2 default of 2048 truncates the jamming prompt (2144 tokens with
    # four agents jammed)
    "num_ctx": 4096,
    "num_predict": -1,
}
OLLAMA_URL = config.LLM_ENDPOINT.split("/api/")[0]


def parse_seeds(text: str):
    seeds = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def git_info() -> dict:
    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def ollama_info(model: str) -> dict:
    try:
        version = requests.get(f"{OLLAMA_URL}/api/version", timeout=5).json()
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
    except Exception as e:
        return {"error": str(e)}
    digest = next(
        (m.get("digest") for m in tags.get("models", []) if m.get("name") == model),
        None,
    )
    return {"version": version.get("version"), "model": model, "digest": digest}


def environment(model=None) -> dict:
    cpu = ""
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    env = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cpu": cpu,
        "git": git_info(),
        "gpu": power.query_gpu(),
    }
    if model:
        env["ollama"] = ollama_info(model)
    return env


def warm_up(model: str, options: dict):
    """Load the model with the run's options so that loading is not timed."""
    response = requests.post(
        config.LLM_ENDPOINT,
        json={
            "model": model,
            "prompt": "Reply with OK.",
            "stream": False,
            "options": {**options, "num_predict": 1},
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def run_one(
    scenario: Scenario,
    supervisor: str,
    seed: int,
    run_dir: Path,
    c2: bool = False,
    c2_depth: float = 0.0,
    paced: bool = False,
    t_step: float = T_STEP,
    max_iter: int = config.MAX_ITER,
    model: str = LLM_MODEL,
    options: dict = None,
    monitor=None,
    idle_w=None,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    instance = scenario.instance(seed)

    config.OBSTACLE_MODE = scenario.mode
    state = SwarmState()
    state.swarm_position = instance["positions"].copy()
    state.initial_positions = instance["positions"].copy()
    state.swarm_destination = instance["destination"].copy()
    state.obstacles = [tuple(map(float, o)) for o in instance["obstacles"]]
    state.obstacle_mode = scenario.mode

    kwargs = {
        "run_dir": str(run_dir),
        "c2_gate": c2,
        "c2_depth": c2_depth,
        "llm_model": model,
    }
    if supervisor == "llm":
        kwargs.update({"options": options or LLM_OPTIONS, "seed": seed})

    def make_supervisor(swarm_state):
        return SUPERVISORS[supervisor](swarm_state, **kwargs)

    factory = ExperimentFactory(state, make_supervisor)
    factory.set_active_controller(ControllerType.COMBINED)
    sup = factory.controllers[ControllerType.LLM]
    if not paced:
        sup._clock = lambda: VIRTUAL_T0 + state.iteration * t_step

    n = state.swarm_size
    ts = {k: [] for k in ("t", "jn", "rn", "positions", "jammed", "depth")}
    ts.update({"applied": [], "unreachable": [], "pending": []})
    conv_step = None
    success = False
    overruns = 0

    started = time.time()
    clock0 = time.perf_counter()
    for k in range(max_iter):
        factory.applied_agents = []
        factory.step()

        applied = np.zeros(n, dtype=bool)
        applied[factory.applied_agents] = True
        ts["t"].append(time.perf_counter() - clock0)
        ts["jn"].append(state.Jn[-1])
        ts["rn"].append(state.rn[-1])
        ts["positions"].append(state.swarm_position.copy())
        ts["jammed"].append(state.jamming_affected.copy())
        ts["depth"].append(state.jamming_depth.copy())
        ts["applied"].append(applied)
        ts["unreachable"].append(~sup.gamma)
        ts["pending"].append(bool(sup.is_llm_request_pending))

        if not state.Jn_converged and state.check_convergence():
            # The GUI pauses here until the user presses Continue; the
            # headless run continues at once
            state.Jn_converged = True
            conv_step = state.iteration
        elif state.check_destination_reached():
            success = True
            break

        if paced:
            delay = clock0 + (k + 1) * t_step - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                overruns += 1
    wall_s = time.time() - started

    if sup.feedback_thread is not None:
        sup.feedback_thread.join(timeout=30)
    sup.close()

    arrays = {
        "t": np.array(ts["t"]),
        "jn": np.array(ts["jn"]),
        "rn": np.array(ts["rn"]),
        "positions": np.array(ts["positions"]),
        "jammed": np.array(ts["jammed"]),
        "depth": np.array(ts["depth"]),
        "applied": np.array(ts["applied"]),
        "unreachable": np.array(ts["unreachable"]),
        "pending": np.array(ts["pending"]),
        "initial_positions": instance["positions"],
        "obstacles": instance["obstacles"],
    }
    np.savez_compressed(run_dir / "timeseries.npz", **arrays)

    run = {
        "scenario": scenario.name,
        "mode": scenario.mode.value,
        "supervisor": supervisor,
        "c2_gate": c2,
        "c2_depth": c2_depth if c2 else None,
        "seed": seed,
        "paced": paced,
        "t_step": t_step,
        "max_iter": max_iter,
        "success": success,
        "steps": state.iteration,
        "conv_step": conv_step,
        "mission_steps": state.iteration - conv_step if conv_step is not None else None,
        "wall_s": wall_s,
        "pacing_overruns": overruns,
        "obstacles": instance["obstacles"].tolist(),
        "initial_positions": instance["positions"].tolist(),
    }
    if supervisor == "llm":
        run["llm_model"] = model
        run["llm_options"] = {
            k: v for k, v in sup.ollama_options.items() if k != "seed"
        }
    if idle_w is not None:
        run["gpu_idle_w"] = idle_w
    if monitor is not None:
        run["gpu_power_synced"] = monitor.sync(time.time())
    summary = metrics.summarize(arrays, sup.decisions, run, monitor, idle_w)
    if monitor is not None:
        summary["gpu_energy_run_j"] = monitor.energy(started, started + wall_s)
        summary["gpu_power_field"] = monitor.field

    with open(run_dir / "decisions.jsonl", "w") as f:
        for d in sup.decisions:
            f.write(json.dumps(d, default=_json_default) + "\n")
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default)
    )
    return summary


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"not serializable: {type(obj)}")


def group_name(scenario: str, c2: bool, c2_depth: float) -> str:
    if not c2:
        return scenario
    if c2_depth == 0.0:
        return f"{scenario}_c2"
    return f"{scenario}_c2d{round(c2_depth * 100):03d}"


def run_dir_for(
    out: Path, scenario: str, supervisor: str, seed: int, c2: bool, c2_depth: float
) -> Path:
    return out / group_name(scenario, c2, c2_depth) / supervisor / f"seed_{seed:02d}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--supervisors", nargs="+", required=True, choices=SUPERVISORS)
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--c2", action="store_true", help="jammed C2 links")
    parser.add_argument(
        "--c2-depth",
        type=float,
        default=0.0,
        help="a jammed agent loses its C2 link at this penetration depth "
        "(0: as soon as it enters the field)",
    )
    parser.add_argument("--paced", action="store_true", help="real-time steps")
    parser.add_argument("--t-step", type=float, default=T_STEP)
    parser.add_argument("--max-iter", type=int, default=config.MAX_ITER)
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("--num-ctx", type=int, default=LLM_OPTIONS["num_ctx"])
    parser.add_argument("--gpu-power", action="store_true")
    parser.add_argument("--idle-seconds", type=float, default=10.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="re-run finished runs")
    args = parser.parse_args(argv)

    logging.getLogger().setLevel(logging.WARNING)
    for name in ("LLMController", "LLMResponseFile", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    options = {**LLM_OPTIONS, "num_ctx": args.num_ctx}
    uses_llm = "llm" in args.supervisors
    args.out.mkdir(parents=True, exist_ok=True)

    env = environment(args.model if uses_llm else None)
    env["argv"] = sys.argv
    env["llm_options"] = options if uses_llm else None
    with open(args.out / "environment.jsonl", "a") as f:
        f.write(json.dumps({"time": time.time(), **env}) + "\n")
    if env["git"]["dirty"]:
        print("WARNING: tracked files have uncommitted changes", file=sys.stderr)

    monitor, idle_w = None, None
    if args.gpu_power:
        monitor = power.PowerMonitor().start()
    if uses_llm:
        warm_up(args.model, options)
    if monitor is not None:
        time.sleep(1.0)
        t0 = time.time()
        time.sleep(args.idle_seconds)
        t1 = time.time()
        monitor.sync(t1)
        idle_w = monitor.mean_power(t0, t1)
        print(f"idle GPU power {idle_w:.1f} W ({monitor.field})", file=sys.stderr)

    seeds = parse_seeds(args.seeds)
    index = args.out / "runs.jsonl"
    try:
        for name in args.scenarios:
            scenario = Scenario.load(name)
            for supervisor in args.supervisors:
                for seed in seeds:
                    run_dir = run_dir_for(
                        args.out, name, supervisor, seed, args.c2, args.c2_depth
                    )
                    if (run_dir / "summary.json").exists() and not args.force:
                        continue
                    if supervisor == "llm":
                        warm_up(args.model, options)
                    with (
                        open(os.devnull, "w") as devnull,
                        contextlib.redirect_stdout(devnull),
                    ):
                        summary = run_one(
                            scenario,
                            supervisor,
                            seed,
                            run_dir,
                            c2=args.c2,
                            c2_depth=args.c2_depth,
                            paced=args.paced,
                            t_step=args.t_step,
                            max_iter=args.max_iter,
                            model=args.model,
                            options=options,
                            monitor=monitor,
                            idle_w=idle_w,
                        )
                    with open(index, "a") as f:
                        f.write(json.dumps(summary, default=_json_default) + "\n")
                    print(
                        f"{group_name(name, args.c2, args.c2_depth)} {supervisor} "
                        f"seed {seed}: "
                        f"success={summary['success']} steps={summary['steps']} "
                        f"Jn_min={summary['jn_min_mission']} "
                        f"commands={summary['n_commands']} wall={summary['wall_s']:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
    finally:
        if monitor is not None:
            monitor.stop()


if __name__ == "__main__":
    main()
