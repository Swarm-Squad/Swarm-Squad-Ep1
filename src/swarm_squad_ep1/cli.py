"""Command-line entrypoint for Swarm Squad Ep1."""

from __future__ import annotations

import argparse
from typing import Sequence

from . import runtime
from .research.__main__ import main as research_main


def _run_gui(args: argparse.Namespace) -> int:
    """Launch the default dual-service GUI stack."""
    return int(runtime.launch(mode="dual", monitor=not args.no_monitor))


def _run_services(args: argparse.Namespace) -> int:
    """Launch explicit service modes for advanced workflows."""
    if args.simulation_only and args.chat_only:
        raise SystemExit("--simulation-only and --chat-only are mutually exclusive")
    mode: runtime.ServiceMode = "dual"
    if args.simulation_only:
        mode = "simulation"
    elif args.chat_only:
        mode = "chat"
    return int(runtime.launch(mode=mode, monitor=not args.no_monitor))


def _run_research(args: argparse.Namespace) -> int:
    """Forward to the research harness CLI."""
    return int(research_main(args.research_args))


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="swarm-squad-ep1",
        description="Swarm Squad Ep1 launcher.",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    gui_parser = sub.add_parser(
        "gui",
        help="Launch chat dashboard and simulation API.",
    )
    gui_parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Start services and exit without blocking the terminal.",
    )
    gui_parser.set_defaults(func=_run_gui)

    services_parser = sub.add_parser(
        "services",
        help="Launch only selected service(s): simulation, chat, or both.",
    )
    services_parser.add_argument(
        "--simulation-only",
        action="store_true",
        help="Launch only the simulation API service.",
    )
    services_parser.add_argument(
        "--chat-only",
        action="store_true",
        help="Launch only the chat GUI service.",
    )
    services_parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Start services and exit without blocking the terminal.",
    )
    services_parser.set_defaults(func=_run_services)

    research_parser = sub.add_parser(
        "research",
        help="Run research harness commands (list/run/smoke/plot).",
    )
    research_parser.add_argument(
        "research_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to `python -m swarm_squad_ep1.research`.",
    )
    research_parser.set_defaults(func=_run_research)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by `[project.scripts]`.

    With no subcommand, defaults to launching the GUI stack.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        return int(runtime.launch(mode="dual", monitor=True))
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
