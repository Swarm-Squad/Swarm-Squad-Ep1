"""
CLI entry point for the research harness.

    python -m swarm_squad_ep1.research run --experiment=E1 --seeds=3
    python -m swarm_squad_ep1.research run --experiment=all --seeds=5 --max-steps=600
    python -m swarm_squad_ep1.research list
    python -m swarm_squad_ep1.research smoke
    python -m swarm_squad_ep1.research plot --csv=results/E1/<timestamp>.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPERIMENT_DESCRIPTIONS = {
    "E1": "Single-vs-dual attack susceptibility (jam/spoof alone vs combined)",
    "E2": "LLM assistance under dual attack (proves LLM always helps)",
    "E3": "Path planning algorithm comparison under attack",
    "E4": "Cryptographic authentication comparison (3 algos + no-crypto)",
    "E5": "Full factorial: attack × LLM × crypto (main evidence table)",
    "E6": "Communication model comparison (V2V vs legacy)",
}


def _cmd_list(args) -> int:
    print("\nAvailable experiments:\n")
    for eid, desc in EXPERIMENT_DESCRIPTIONS.items():
        print(f"  {eid:4s}  {desc}")
    print("\n  all   Run E1 through E6 sequentially\n")
    return 0


def _cmd_run(args) -> int:
    from swarm_squad_ep1.research.experiments import EXPERIMENTS, run_experiment

    if args.experiment == "all":
        targets = list(EXPERIMENTS.keys())
    elif "," in args.experiment:
        targets = [e.strip() for e in args.experiment.split(",")]
    else:
        targets = [args.experiment]

    for name in targets:
        if name not in EXPERIMENTS:
            print(
                f"unknown experiment {name!r}; options: {list(EXPERIMENTS)}",
                file=sys.stderr,
            )
            return 2

    for name in targets:
        print(f"\n{'=' * 60}")
        print(f"  EXPERIMENT {name}: {EXPERIMENT_DESCRIPTIONS.get(name, '')}")
        print(f"{'=' * 60}\n")
        run_experiment(
            name,
            out_dir=args.out_dir,
            seeds=args.seeds,
            max_steps=args.max_steps,
            keep_trace=args.keep_trace,
            verbose=not args.quiet,
        )

    return 0


def _cmd_smoke(args) -> int:
    from swarm_squad_ep1.research.smoke_test import run_smoke

    return run_smoke(verbose=not args.quiet)


def _cmd_plot(args) -> int:
    from swarm_squad_ep1.research.plot import generate_all_plots

    p = Path(args.csv)
    if not p.exists():
        print(f"not found: {p}", file=sys.stderr)
        return 2
    paths = generate_all_plots(p, out_dir=args.out_dir)
    for x in paths:
        print(f"wrote {x}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m swarm_squad_ep1.research",
        description="Research harness for Swarm Squad Ep1 experiments",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="show available experiments")
    p_list.set_defaults(func=_cmd_list)

    p_run = sub.add_parser("run", help="run an experiment matrix")
    p_run.add_argument(
        "--experiment",
        default="E1",
        help="experiment ID (E1-E6), comma-separated list, or 'all'",
    )
    p_run.add_argument("--seeds", type=int, default=3)
    p_run.add_argument("--max-steps", type=int, default=None, dest="max_steps")
    p_run.add_argument("--out-dir", default="results", dest="out_dir")
    p_run.add_argument("--keep-trace", action="store_true", dest="keep_trace")
    p_run.add_argument("--quiet", action="store_true")
    p_run.set_defaults(func=_cmd_run)

    p_smoke = sub.add_parser("smoke", help="run a fast smoke test (4 short scenarios)")
    p_smoke.add_argument("--quiet", action="store_true")
    p_smoke.set_defaults(func=_cmd_smoke)

    p_plot = sub.add_parser("plot", help="render plots from a result CSV")
    p_plot.add_argument("--csv", required=True)
    p_plot.add_argument(
        "--out-dir",
        default=None,
        dest="out_dir",
        help="directory for PNGs (default: next to CSV)",
    )
    p_plot.set_defaults(func=_cmd_plot)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
