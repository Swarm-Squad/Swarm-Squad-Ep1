"""
Paper figures from the recorded runs, with control steps on the x-axis.

Usage:
    uv run python -m swarm_squad_ep1.experiments.figures \
        experiments/results/local experiments/results/llm --seed 0 --out fig
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

import swarm_squad_ep1.config as config  # noqa: E402
from swarm_squad_ep1.experiments import compass, linkmodel  # noqa: E402
from swarm_squad_ep1.experiments.supervisors import INSTRUCTION_PATTERN  # noqa: E402

AGENT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]
SUPERVISOR_STYLE = {
    "none": {"color": "0.35", "ls": "-", "label": "Reactive only"},
    "rule": {"color": "#1f77b4", "ls": "--", "label": "Rule"},
    "mpc": {"color": "#2ca02c", "ls": "-.", "label": "MPC"},
    "llm": {"color": "#d62728", "ls": "-", "label": "LLM"},
}
COMMAND_LIFETIME = 10
DESTINATION = config.DEFAULT_DESTINATION.astype(float)

plt.rcParams.update(
    {
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "lines.linewidth": 0.9,
        "axes.linewidth": 0.5,
        "pdf.fonttype": 42,
    }
)


def find_run(dirs, group, supervisor, seed):
    for d in dirs:
        run_dir = Path(d) / group / supervisor / f"seed_{seed:02d}"
        if (run_dir / "timeseries.npz").exists():
            return run_dir
    raise FileNotFoundError(f"{group}/{supervisor}/seed_{seed:02d}")


def load(run_dir):
    ts = dict(np.load(run_dir / "timeseries.npz"))
    summary = json.loads((run_dir / "summary.json").read_text())
    decisions = [
        json.loads(line)
        for line in (run_dir / "decisions.jsonl").read_text().splitlines()
    ]
    return ts, summary, decisions


def active_commands(decisions, step):
    """Commands in force at a step, as {agent: (distance, bearing)}."""
    active = {}
    for d in decisions:
        start = d.get("step_parsed")
        if start is None or not (start <= step < start + COMMAND_LIFETIME):
            continue
        delivered = set(d.get("delivered") or [])
        for agent, distance, bearing in INSTRUCTION_PATTERN.findall(
            d.get("text") or ""
        ):
            agent = int(agent)
            if agent in delivered:
                active[agent] = (int(distance), bearing.lower())
    return active


def draw_trajectories(ax, ts, summary):
    for x, y, r in ts["obstacles"]:
        ax.add_patch(Circle((x, y), r, color="0.75", lw=0))
    pos = ts["positions"]
    for i in range(pos.shape[1]):
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        ax.plot(pos[:, i, 0], pos[:, i, 1], color=c, lw=0.7)
        ax.plot(*pos[0, i], "o", color=c, ms=2.5)
    ax.add_patch(Rectangle(DESTINATION - 4, 8, 8, fill=False, ec="k", lw=0.6))
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


def fig_obstacles(dirs, seed, out, group="obstacles_2"):
    fig, axes = plt.subplots(
        1, 3, figsize=(3.5, 1.95), gridspec_kw={"width_ratios": [1, 1, 1.3]}
    )
    jn_ax = axes[2]
    for col, (sup, tag) in enumerate([("none", "a"), ("llm", "b")]):
        ts, summary, _ = load(find_run(dirs, group, sup, seed))
        ax = axes[col]
        draw_trajectories(ax, ts, summary)
        ax.set_xlim(-30, 100)
        ax.set_ylim(-30, 175)
        ax.set_title(
            f"({tag}) {SUPERVISOR_STYLE[sup]['label']},\n{summary['steps']} steps"
        )
        if col:
            ax.set_ylabel("")
        style = SUPERVISOR_STYLE[sup]
        jn_ax.plot(
            ts["jn"], color=style["color"], ls=style["ls"], lw=0.9, label=style["label"]
        )
    jn_ax.axhline(config.PT, color="k", lw=0.5, ls=":")
    jn_ax.set_ylim(0.9, 0.98)
    jn_ax.set_xlabel("Step")
    jn_ax.set_ylabel("$J_n$", labelpad=1)
    jn_ax.set_title("(c) Link quality")
    jn_ax.legend(loc="lower right", frameon=False, handlelength=1.5)
    fig.tight_layout(pad=0.2, w_pad=0.3)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def frame_steps(ts, conv_step):
    jammed = ts["jammed"].any(axis=1)
    idx = np.flatnonzero(jammed)
    if idx.size == 0:
        raise ValueError("no agent was jammed")
    first, last = int(idx[0]), int(idx[-1])
    mission = ts["jn"][conv_step:]
    worst = conv_step + int(np.argmin(mission))
    return [first, worst, last, min(last + 20, len(ts["jn"]) - 1)]


def draw_frame(ax, ts, decisions, step):
    jammers = [tuple(o) for o in ts["obstacles"]]
    for x, y, r in jammers:
        ax.add_patch(
            Circle((x, y), r * config.JAMMING_RADIUS_MULTIPLIER, color="#fff7bc", lw=0)
        )
        ax.add_patch(Circle((x, y), r, color="#fee391", lw=0))
    pos = ts["positions"][step]
    q, a = linkmodel.link_quality(pos, jammers)
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            if a[i, j] <= config.PT:
                continue
            good = q[i, j] >= config.PT
            ax.plot(
                *zip(pos[i], pos[j]),
                color="#2ca02c" if good else "#d62728",
                lw=0.5,
                ls="-" if good else "--",
                zorder=2,
            )
    commands = active_commands(decisions, step)
    for i in range(n):
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        edge = "k" if ts["jammed"][step, i] else c
        ax.plot(*pos[i], "o", color=c, mec=edge, mew=0.6, ms=3.5, zorder=3)
        if i in commands:
            distance, bearing = commands[i]
            v = distance * compass.unit_vector(bearing)
            ax.annotate(
                "",
                xy=pos[i] + v,
                xytext=pos[i],
                arrowprops={"arrowstyle": "-|>", "color": "#3f37c9", "lw": 0.8},
                zorder=4,
            )
    center = pos.mean(axis=0)
    ax.set_xlim(center[0] - 45, center[0] + 45)
    ax.set_ylim(center[1] - 40, center[1] + 40)
    ax.set_aspect("equal")
    ax.set_title(f"Step {step}, $J_n$ = {ts['jn'][step]:.3f}")
    ax.tick_params(length=2)


def fig_jamming_frames(dirs, seed, out, group="jamming", supervisor="llm"):
    ts, summary, decisions = load(find_run(dirs, group, supervisor, seed))
    steps = frame_steps(ts, summary["conv_step"])
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.3))
    for ax, step, tag in zip(axes.flat, steps, "abcd"):
        draw_frame(ax, ts, decisions, step)
        ax.text(
            0.03, 0.95, f"{tag})", transform=ax.transAxes, va="top", fontweight="bold"
        )
    fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.5)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return steps


def fig_performance(
    dirs, seed, out, group="jamming", supervisors=("none", "rule", "mpc", "llm")
):
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.55))
    for sup in supervisors:
        try:
            ts, summary, _ = load(find_run(dirs, group, sup, seed))
        except FileNotFoundError:
            continue
        style = SUPERVISOR_STYLE[sup]
        kw = {"color": style["color"], "ls": style["ls"], "label": style["label"]}
        axes[0].plot(ts["jn"], **kw)
        axes[1].plot(ts["rn"], **kw)
    axes[0].axhline(config.PT, color="k", lw=0.5, ls=":")
    axes[0].set_ylim(0.84, 0.98)
    axes[0].set_ylabel("$J_n$")
    axes[1].set_ylim(14, 42)
    axes[1].set_ylabel("$r_n$")
    for ax in axes:
        ax.set_xlabel("Step")
        ax.tick_params(length=2)
    axes[1].legend(
        loc="lower left",
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=0.8,
        borderaxespad=0.2,
        fontsize=6,
    )
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--supervisors", nargs="+", default=["none", "rule", "mpc", "llm"]
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    made = []
    try:
        fig_obstacles(args.dirs, args.seed, args.out / "obstacles2_compare.pdf")
        made.append("obstacles2_compare.pdf")
    except FileNotFoundError as e:
        print(f"skipping Fig. 1: missing {e}")
    try:
        steps = fig_jamming_frames(
            args.dirs, args.seed, args.out / "jamming_frames.pdf"
        )
        made.append(f"jamming_frames.pdf (steps {steps})")
    except FileNotFoundError as e:
        print(f"skipping Fig. 2: missing {e}")
    fig_performance(
        args.dirs,
        args.seed,
        args.out / "jamming_performance.pdf",
        supervisors=args.supervisors,
    )
    made.append("jamming_performance.pdf")
    print("\n".join(made))


if __name__ == "__main__":
    main()
