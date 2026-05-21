"""
Plotting utilities for experiment CSVs.

Uses matplotlib when available (soft dependency). CSV is the source of
truth; plots are secondary.

Each ``plot_*`` function reads a CSV, produces one figure, and returns the
output path (or ``None`` if matplotlib is unavailable).

``generate_all_plots(csv_path)`` runs every applicable plot for the data.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _load_csv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _to_bool(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _group_by(rows: list[dict], key_fn) -> dict:
    out: dict = {}
    for r in rows:
        k = key_fn(r)
        out.setdefault(k, []).append(r)
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _ensure_mpl():
    if not HAS_MPL:
        print("[plot] matplotlib not installed; skipping plot")
        return False
    return True


COLORS = {
    "blue": "#4c9aff",
    "grey": "#b0b7c3",
    "red": "#ff6b6b",
    "green": "#51cf66",
    "orange": "#ffa94d",
    "purple": "#b197fc",
    "teal": "#38d9a9",
    "pink": "#f06595",
}


# --------------------------------------------------------------------------
# E1: Single vs dual attack
# --------------------------------------------------------------------------


def plot_single_vs_dual(
    csv_path: str | Path, out_png: Optional[str | Path] = None
) -> Optional[str]:
    """Grouped bar chart: success rate for jam-only, spoof-only, combined."""
    if not _ensure_mpl():
        return None

    rows = _load_csv(Path(csv_path))
    rows = [r for r in rows if r.get("scenario_name", "").startswith("E1_")]

    if not rows:
        return None

    categories = {"baseline": [], "jam_only": [], "spoof_only": [], "combined": []}
    for r in rows:
        name = r["scenario_name"]
        reached = _to_bool(r["destination_reached"])
        if "baseline" in name:
            categories["baseline"].append(reached)
        elif "combo" in name:
            categories["combined"].append(reached)
        elif "jam_" in name and "spoof" not in name:
            categories["jam_only"].append(reached)
        elif "spoof_" in name:
            categories["spoof_only"].append(reached)

    labels = ["Baseline", "Jamming\nOnly", "Spoofing\nOnly", "Jamming +\nSpoofing"]
    keys = ["baseline", "jam_only", "spoof_only", "combined"]
    values = [
        _mean([float(v) for v in categories[k]]) if categories[k] else 0 for k in keys
    ]
    counts = [len(categories[k]) for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        labels,
        values,
        color=[COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["red"]],
    )
    for bar, val, n in zip(bars, values, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.0%}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Mission Success Rate")
    ax.set_ylim(0, 1.15)
    ax.set_title("E1: Single Attack vs Combined Attack Susceptibility")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    fig.tight_layout()
    out = (
        Path(out_png) if out_png else Path(csv_path).with_name("E1_single_vs_dual.png")
    )
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------
# E1 supplement: comm quality degradation
# --------------------------------------------------------------------------


def plot_comm_quality_by_attack(
    csv_path: str | Path, out_png: Optional[str | Path] = None
) -> Optional[str]:
    """Box-style bar chart of avg comm quality per attack category."""
    if not _ensure_mpl():
        return None

    rows = _load_csv(Path(csv_path))
    rows = [r for r in rows if r.get("scenario_name", "").startswith("E1_")]
    if not rows:
        return None

    groups: dict[str, list[float]] = {}
    for r in rows:
        name = r["scenario_name"]
        val = _to_float(r.get("avg_comm_quality"))
        if "baseline" in name:
            groups.setdefault("Baseline", []).append(val)
        elif "combo" in name:
            groups.setdefault("Combined", []).append(val)
        elif "jam_" in name and "spoof" not in name:
            groups.setdefault("Jam Only", []).append(val)
        elif "spoof_" in name:
            groups.setdefault("Spoof Only", []).append(val)

    order = ["Baseline", "Jam Only", "Spoof Only", "Combined"]
    labels = [k for k in order if k in groups]
    data = [groups[k] for k in labels]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = [COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["red"]]
    for patch, c in zip(bp["boxes"], colors[: len(labels)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel("Avg Communication Quality")
    ax.set_title("E1: Communication Quality Degradation by Attack Type")
    fig.tight_layout()
    out = Path(out_png) if out_png else Path(csv_path).with_name("E1_comm_quality.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------
# E2: LLM assistance under dual attack
# --------------------------------------------------------------------------


def plot_llm_improvement(
    csv_path: str | Path, out_png: Optional[str | Path] = None
) -> Optional[str]:
    """Paired bar chart: LLM off vs on for each combined-attack config."""
    if not _ensure_mpl():
        return None

    rows = _load_csv(Path(csv_path))
    rows = [r for r in rows if r.get("scenario_name", "").startswith("E2_")]
    if not rows:
        return None

    groups: dict[str, dict[bool, list]] = {}
    for r in rows:
        name = r["scenario_name"]
        llm = _to_bool(r["llm_enabled"])
        reached = _to_bool(r["destination_reached"])
        base = name.replace("_llm=0", "").replace("_llm=1", "").replace("E2_", "")
        groups.setdefault(base, {}).setdefault(llm, []).append(reached)

    labels = sorted(groups.keys())
    off_vals = [
        _mean([float(v) for v in groups[label].get(False, [])]) for label in labels
    ]
    on_vals = [
        _mean([float(v) for v in groups[label].get(True, [])]) for label in labels
    ]

    import numpy as _np

    x = _np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    ax.bar(x - w / 2, off_vals, w, label="LLM OFF", color=COLORS["grey"])
    ax.bar(x + w / 2, on_vals, w, label="LLM ON", color=COLORS["blue"])
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace("_", "\n") for label in labels], fontsize=8)
    ax.set_ylabel("Mission Success Rate")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("E2: LLM Assistance Improvement Under Combined Attack")
    ax.legend()
    fig.tight_layout()
    out = (
        Path(out_png) if out_png else Path(csv_path).with_name("E2_llm_improvement.png")
    )
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def plot_llm_steps_saved(
    csv_path: str | Path, out_png: Optional[str | Path] = None
) -> Optional[str]:
    """Paired bar chart: avg steps to destination, LLM off vs on."""
    if not _ensure_mpl():
        return None

    rows = _load_csv(Path(csv_path))
    rows = [
        r
        for r in rows
        if r.get("scenario_name", "").startswith("E2_")
        and _to_bool(r.get("destination_reached", "false"))
    ]
    if not rows:
        return None

    groups: dict[str, dict[bool, list[float]]] = {}
    for r in rows:
        name = r["scenario_name"]
        llm = _to_bool(r["llm_enabled"])
        steps = _to_float(r.get("total_steps"))
        base = name.replace("_llm=0", "").replace("_llm=1", "").replace("E2_", "")
        groups.setdefault(base, {}).setdefault(llm, []).append(steps)

    labels = sorted(groups.keys())
    off_vals = [_mean(groups[label].get(False, [0])) for label in labels]
    on_vals = [_mean(groups[label].get(True, [0])) for label in labels]

    import numpy as _np

    x = _np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    ax.bar(x - w / 2, off_vals, w, label="LLM OFF", color=COLORS["grey"])
    ax.bar(x + w / 2, on_vals, w, label="LLM ON", color=COLORS["blue"])
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace("_", "\n") for label in labels], fontsize=8)
    ax.set_ylabel("Avg Steps to Destination")
    ax.set_title("E2: Steps to Destination — LLM OFF vs ON")
    ax.legend()
    fig.tight_layout()
    out = Path(out_png) if out_png else Path(csv_path).with_name("E2_steps_saved.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------
# E3: Path planning comparison
# --------------------------------------------------------------------------


def plot_path_algorithms(
    csv_path: str | Path, out_png: Optional[str | Path] = None
) -> Optional[str]:
    """Compare path algorithms: success rate + avg path length."""
    if not _ensure_mpl():
        return None

    rows = _load_csv(Path(csv_path))
    rows = [r for r in rows if r.get("scenario_name", "").startswith("E3_")]
    if not rows:
        return None

    algo_data: dict[str, dict] = {}
    for r in rows:
        algo = r.get("path_algorithm", "?")
        d = algo_data.setdefault(algo, {"reached": [], "path_len": [], "steps": []})
        d["reached"].append(_to_bool(r["destination_reached"]))
        d["path_len"].append(_to_float(r.get("total_path_length")))
        if _to_bool(r["destination_reached"]):
            d["steps"].append(_to_float(r.get("total_steps")))

    algos = sorted(algo_data.keys())
    success = [_mean([float(v) for v in algo_data[a]["reached"]]) for a in algos]
    path_lens = [_mean(algo_data[a]["path_len"]) for a in algos]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars = ax1.bar(algos, success, color=COLORS["blue"])
    for bar, val in zip(bars, success):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax1.set_ylabel("Mission Success Rate")
    ax1.set_ylim(0, 1.15)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.set_title("Success Rate by Algorithm")
    ax1.tick_params(axis="x", rotation=30)

    ax2.bar(algos, path_lens, color=COLORS["teal"])
    ax2.set_ylabel("Avg Total Path Length (m)")
    ax2.set_title("Path Length by Algorithm")
    ax2.tick_params(axis="x", rotation=30)

    fig.suptitle("E3: Path Planning Algorithm Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    out = (
        Path(out_png) if out_png else Path(csv_path).with_name("E3_path_algorithms.png")
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------
# E4: Crypto comparison
# --------------------------------------------------------------------------


def plot_crypto_comparison(
    csv_path: str | Path, out_png: Optional[str | Path] = None
) -> Optional[str]:
    """Detection rate + FPR by crypto algorithm under spoofing."""
    if not _ensure_mpl():
        return None

    rows = _load_csv(Path(csv_path))
    rows = [r for r in rows if r.get("scenario_name", "").startswith("E4_")]
    if not rows:
        return None

    groups: dict[str, dict] = {}
    for r in rows:
        algo = r.get("crypto_algorithm", "-")
        if algo == "-":
            algo = "none"
        d = groups.setdefault(algo, {"det": [], "fpr": [], "prec": [], "reached": []})
        d["det"].append(_to_float(r.get("detection_rate")))
        d["fpr"].append(_to_float(r.get("false_positive_rate")))
        d["prec"].append(_to_float(r.get("precision")))
        d["reached"].append(_to_bool(r["destination_reached"]))

    algos = sorted(groups.keys())
    det_vals = [_mean(groups[a]["det"]) for a in algos]
    fpr_vals = [_mean(groups[a]["fpr"]) for a in algos]
    prec_vals = [_mean(groups[a]["prec"]) for a in algos]
    success_vals = [_mean([float(v) for v in groups[a]["reached"]]) for a in algos]

    import numpy as _np

    x = _np.arange(len(algos))
    w = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * w, success_vals, w, label="Success Rate", color=COLORS["green"])
    ax.bar(x - 0.5 * w, det_vals, w, label="Detection Rate", color=COLORS["blue"])
    ax.bar(x + 0.5 * w, prec_vals, w, label="Precision", color=COLORS["teal"])
    ax.bar(x + 1.5 * w, fpr_vals, w, label="False Positive Rate", color=COLORS["red"])
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", "\n") for a in algos], fontsize=9)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("E4: Cryptographic Authentication Comparison")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = (
        Path(out_png)
        if out_png
        else Path(csv_path).with_name("E4_crypto_comparison.png")
    )
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------
# E5: Full factorial heatmap
# --------------------------------------------------------------------------


def plot_full_factorial(
    csv_path: str | Path, out_png: Optional[str | Path] = None
) -> Optional[str]:
    """Heatmap-style table: attack scenario × (LLM, crypto) → success rate."""
    if not _ensure_mpl():
        return None

    rows = _load_csv(Path(csv_path))
    rows = [r for r in rows if r.get("scenario_name", "").startswith("E5_")]
    if not rows:
        return None

    cells: dict[tuple[str, str], list[bool]] = {}
    for r in rows:
        name = r["scenario_name"]
        llm = _to_bool(r["llm_enabled"])
        crypto = _to_bool(r["crypto_enabled"])
        reached = _to_bool(r["destination_reached"])
        parts = name.replace("E5_", "").rsplit("_l", 1)[0]
        col_label = f"LLM={'ON' if llm else 'OFF'}\nCrypto={'ON' if crypto else 'OFF'}"
        cells.setdefault((parts, col_label), []).append(reached)

    row_labels = sorted(set(k[0] for k in cells))
    col_labels = sorted(set(k[1] for k in cells))

    import numpy as _np

    data = _np.zeros((len(row_labels), len(col_labels)))
    for (r, c), vals in cells.items():
        ri = row_labels.index(r)
        ci = col_labels.index(c)
        data[ri, ci] = _mean([float(v) for v in vals])

    fig, ax = plt.subplots(
        figsize=(max(8, len(col_labels) * 2), max(6, len(row_labels) * 0.6))
    )
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([label.replace("_", " ") for label in row_labels], fontsize=8)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(
                j,
                i,
                f"{data[i, j]:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if data[i, j] > 0.4 else "white",
            )

    fig.colorbar(im, ax=ax, label="Mission Success Rate", shrink=0.8)
    ax.set_title("E5: Full Factorial — Attack × LLM × Crypto", fontsize=12)
    fig.tight_layout()
    out = (
        Path(out_png) if out_png else Path(csv_path).with_name("E5_full_factorial.png")
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------
# E6: Comm model comparison
# --------------------------------------------------------------------------


def plot_comm_model(
    csv_path: str | Path, out_png: Optional[str | Path] = None
) -> Optional[str]:
    """V2V vs legacy: success rate + comm quality by jamming level."""
    if not _ensure_mpl():
        return None

    rows = _load_csv(Path(csv_path))
    rows = [r for r in rows if r.get("scenario_name", "").startswith("E6_")]
    if not rows:
        return None

    groups: dict[tuple[str, str], dict] = {}
    for r in rows:
        model = r.get("comm_model", "?")
        jt = r.get("jamming_types", "-")
        d = groups.setdefault((model, jt), {"reached": [], "comm": []})
        d["reached"].append(_to_bool(r["destination_reached"]))
        d["comm"].append(_to_float(r.get("avg_comm_quality")))

    import numpy as _np

    jam_levels = sorted(set(k[1] for k in groups))
    models = sorted(set(k[0] for k in groups))

    x = _np.arange(len(jam_levels))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for idx, model in enumerate(models):
        vals = [
            _mean([float(v) for v in groups.get((model, j), {}).get("reached", [])])
            for j in jam_levels
        ]
        ax1.bar(x + (idx - 0.5) * w, vals, w, label=model)
    ax1.set_xticks(x)
    ax1.set_xticklabels(jam_levels)
    ax1.set_ylabel("Mission Success Rate")
    ax1.set_ylim(0, 1.15)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.set_title("Success Rate")
    ax1.legend()

    for idx, model in enumerate(models):
        vals = [_mean(groups.get((model, j), {}).get("comm", [0])) for j in jam_levels]
        ax2.bar(x + (idx - 0.5) * w, vals, w, label=model)
    ax2.set_xticks(x)
    ax2.set_xticklabels(jam_levels)
    ax2.set_ylabel("Avg Communication Quality")
    ax2.set_title("Comm Quality")
    ax2.legend()

    fig.suptitle("E6: V2V Channel vs Legacy Model", fontsize=13, y=1.02)
    fig.tight_layout()
    out = Path(out_png) if out_png else Path(csv_path).with_name("E6_comm_model.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------
# General-purpose plots (work across experiments)
# --------------------------------------------------------------------------


def plot_success_by_llm(
    csv_path: str | Path, out_png: Optional[str | Path] = None
) -> Optional[str]:
    """Bar chart of mission-success rate with LLM on vs off."""
    if not _ensure_mpl():
        return None

    rows = _load_csv(Path(csv_path))
    for r in rows:
        r["llm_enabled"] = _to_bool(r["llm_enabled"])
        r["destination_reached"] = _to_bool(r["destination_reached"])

    groups = _group_by(rows, lambda r: r["llm_enabled"])
    labels, values = [], []
    for llm in (False, True):
        grp = groups.get(llm, [])
        reached = sum(1 for r in grp if r["destination_reached"])
        labels.append(f"LLM={'on' if llm else 'off'} (n={len(grp)})")
        values.append(reached / max(1, len(grp)))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(labels, values, color=[COLORS["grey"], COLORS["blue"]])
    ax.set_ylabel("Mission Success Rate")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Success Rate: LLM-Assisted vs Baseline")
    fig.tight_layout()
    out = Path(out_png) if out_png else Path(csv_path).with_name("success_by_llm.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def plot_detection_roc(
    csv_path: str | Path, out_png: Optional[str | Path] = None
) -> Optional[str]:
    """Scatter of (false_positive_rate, detection_rate) points per scenario."""
    if not _ensure_mpl():
        return None

    rows = _load_csv(Path(csv_path))
    for r in rows:
        r["crypto_enabled"] = _to_bool(r["crypto_enabled"])
        r["detection_rate"] = _to_float(r.get("detection_rate"))
        r["false_positive_rate"] = _to_float(r.get("false_positive_rate"))

    has_spoof = [r for r in rows if r.get("spoofing_types", "-") != "-"]
    if not has_spoof:
        return None

    fig, ax = plt.subplots(figsize=(5, 5))
    for crypto in (False, True):
        grp = [r for r in has_spoof if r["crypto_enabled"] == crypto]
        if grp:
            ax.scatter(
                [r["false_positive_rate"] for r in grp],
                [r["detection_rate"] for r in grp],
                label=f"crypto={'on' if crypto else 'off'} (n={len(grp)})",
                alpha=0.6,
            )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("False-Positive Rate")
    ax.set_ylabel("Detection Rate")
    ax.set_title("Spoof Detection: Crypto On vs Off")
    ax.legend()
    fig.tight_layout()
    out = Path(out_png) if out_png else Path(csv_path).with_name("detection_roc.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------
# Master dispatcher
# --------------------------------------------------------------------------

ALL_PLOTS = [
    plot_single_vs_dual,
    plot_comm_quality_by_attack,
    plot_llm_improvement,
    plot_llm_steps_saved,
    plot_path_algorithms,
    plot_crypto_comparison,
    plot_full_factorial,
    plot_comm_model,
    plot_success_by_llm,
    plot_detection_roc,
]


def generate_all_plots(
    csv_path: str | Path,
    out_dir: Optional[str | Path] = None,
) -> list[str]:
    """Run every applicable plot function. Returns list of written PNGs."""
    csv_path = Path(csv_path)
    written: list[str] = []

    for fn in ALL_PLOTS:
        try:
            out_name = fn.__name__.replace("plot_", "") + ".png"
            if out_dir:
                out_png = Path(out_dir) / out_name
                Path(out_dir).mkdir(parents=True, exist_ok=True)
            else:
                out_png = csv_path.with_name(out_name)

            result = fn(csv_path, out_png=out_png)
            if result:
                written.append(result)
        except Exception as e:
            print(f"[plot] {fn.__name__} failed: {e}")

    return written
