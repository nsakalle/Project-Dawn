"""Gantt charts of a night's schedule: one row per node, one bar per job, coloured by application.

Late jobs get a thick dark outline and 'LATE' in their label. Grey hatched bands are maintenance
or outages. The dashed line marks when the disruption became known.

Used by run_pipeline.py; can also be called directly:
    from gantt import compare
    compare(spec, [("Shift-right", sched_a), ("Hybrid repair", sched_b)], "results/gantt.png")
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from common import duration, job_map, node_map, to_hhmm

# Reference categorical palette, fixed order (one slot per application, never cycled)
APP_ORDER = ["Finance close", "Risk", "Retail sales", "Data warehouse", "Machine learning",
             "Supply chain", "Payroll", "IT operations"]
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID, LATE = "#1a1a19", "#6b6a63", "#e4e3dc", "#1a1a19"


def _lanes(spec, schedule, n):
    """Split overlapping jobs on one node into lanes so every bar stays visible."""
    jobs, nodes = job_map(spec), node_map(spec)
    items = sorted((s, s + duration(spec, jobs[j], nodes[n]), j) for j, (nn, s) in schedule.items()
                   if nn == n and j in jobs)
    ends, placed = [], []
    for s, e, j in items:
        k = next((i for i, end in enumerate(ends) if end <= s), None)
        if k is None:
            ends.append(e)
            k = len(ends) - 1
        else:
            ends[k] = e
        placed.append((s, e, j, k))
    return placed, max(1, len(ends))


LANE = 0.42  # height of one lane; a node row grows with the number of jobs running side by side


def _panel(ax, spec, schedule, title, focus=()):
    jobs, nodes = job_map(spec), node_map(spec)
    order = [n["id"] for n in spec["nodes"]]
    color = {a: SLOTS[i] for i, a in enumerate(APP_ORDER)}
    rows = {n: _lanes(spec, schedule, n) for n in order}
    top, base = 0.0, {}
    for n in order:                       # first node at the top
        height = max(1, rows[n][1]) * LANE + 0.3
        base[n] = (top - height, height)
        top -= height
    for n in order:
        y0, height = base[n]
        for b0, b1, why in nodes[n]["blackouts"]:
            if b1 <= 0 or b0 >= 720:
                continue
            ax.barh(y0 + height / 2, b1 - b0, left=b0, height=height - 0.2, color="#d9d8d0", hatch="////",
                    edgecolor="#a9a89f", linewidth=0)
        placed, nl = rows[n]
        for s, e, j, k in placed:
            job = jobs[j]
            late = e > job["deadline"]
            yy = y0 + 0.15 + LANE * (nl - k - 0.5)
            ax.barh(yy, e - s - 2, left=s + 1, height=LANE - 0.06, color=color.get(job["app"], "#888888"),
                    edgecolor=LATE if late else "white", linewidth=2.2 if late else 0.6)
            label = j + (" LATE" if late else "")
            if e - s >= 25:
                ax.text(s + 3, yy, label, va="center", ha="left", fontsize=7, color="white",
                        fontweight="bold" if (late or j in focus) else "normal")
        ax.axhline(y0, color=GRID, linewidth=0.6)
    if spec.get("as_of"):
        ax.axvline(spec["as_of"], color=INK, linestyle="--", linewidth=1)
        ax.text(spec["as_of"] + 4, 0.05, f"disruption known {to_hhmm(spec['as_of'])}", fontsize=8, color=INK,
                va="bottom")
    ax.set_yticks([base[n][0] + base[n][1] / 2 for n in order])
    ax.set_yticklabels(order, fontsize=8, color=INK)
    ticks = list(range(0, 721, 60))
    ax.set_xticks(ticks)
    ax.set_xticklabels([to_hhmm(t) for t in ticks], fontsize=8, color=MUTED)
    ax.set_xlim(0, 720)
    ax.set_ylim(top, 0.5)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    return -top


def compare(spec, panels, path, focus=()):
    """panels: list of (title, schedule). Writes one PNG with the panels stacked."""
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 6.2 * len(panels)), squeeze=False)
    for ax, (title, sched) in zip(axes[:, 0], panels):
        _panel(ax, spec, sched, title, focus)
    handles = [Patch(color=SLOTS[i], label=a) for i, a in enumerate(APP_ORDER)]
    handles.append(Patch(facecolor="#d9d8d0", hatch="////", edgecolor="#a9a89f", label="Maintenance / outage"))
    handles.append(Patch(facecolor="white", edgecolor=LATE, linewidth=2.2, label="Misses its deadline"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
