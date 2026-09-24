#!/usr/bin/env python3
"""Regenerate every figure of the task-11 straggler-profiler RFC.

    source /ytech_m2v4_hdd/xiatianze/envs/relax_env.sh
    python relax_work/task11/evidence/figures/make_figures.py [--expil PATH]

Writes fig_*.png (dpi 200) and fig_*.svg next to this script. Diagrams are drawn
with matplotlib patches only (no graphviz); all text is English (no CJK font).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402


HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
LOGS = TASK / "logs"

# ----------------------------------------------------------------------------- style
BLUE, BLUE_FILL, BLUE_BAND, BLUE_TXT = "#1F5FAD", "#DCE8F7", "#F2F6FC", "#153F75"
ORANGE, ORANGE_FILL, ORANGE_BAND, ORANGE_TXT = "#D9730D", "#FBE3CB", "#FDF4EA", "#8A4306"
GREEN, GREEN_FILL, GREEN_TXT = "#2E7D46", "#DDF0E2", "#1D5530"
RED, RED_FILL, RED_TXT = "#C0392B", "#F8DCD8", "#8E2A20"
GRAY, GRAY_FILL, GRAY_TXT = "#7A7A7A", "#ECECEC", "#3A3A3A"
PURPLE, PURPLE_FILL, PURPLE_TXT = "#6A4C93", "#ECE6F5", "#4A2F6E"
TEXT, MUTED = "#1B1B1B", "#555555"
OFF_C, ON_C = "#8C8C8C", BLUE
STRIP_COLORS = ("#3C78C3", "#8FB3E2", "#C9DBF2")  # GPU segments, CPU fields, counters

KIND = {
    "hot": (ORANGE_FILL, ORANGE, ORANGE_TXT),
    "cold": (BLUE_FILL, BLUE, BLUE_TXT),
    "out": (GREEN_FILL, GREEN, GREEN_TXT),
    "alert": (RED_FILL, RED, RED_TXT),
    "up": (GRAY_FILL, GRAY, GRAY_TXT),
    "plain": ("white", GRAY, TEXT),
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "text.color": TEXT,
        "axes.edgecolor": "#444444",
        "axes.labelcolor": TEXT,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "svg.fonttype": "path",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    print(f"wrote {name}.png / .svg")


# ----------------------------------------------------------------------------- diagram helpers
def canvas(w: float, h: float) -> tuple[plt.Figure, plt.Axes]:
    """Full-bleed axes in inch units, so layout coordinates are physical sizes."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, fc, ec, lw=1.3, ls="-", r=0.07, z=2):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
    )


def node(ax, x, y, w, h, kind, title, lines=(), ts=10.5, ls=9.5, lw=1.3, ls_style="-", dy=0.0):
    """Rounded box with a bold title and optional sub-lines.

    ``lines`` items are ``str`` or ``(str, color, weight)``.
    """
    fc, ec, tc = KIND[kind]
    rbox(ax, x, y, w, h, fc, ec, lw=lw, ls=ls_style)
    items = [(title, ts, "bold", tc)]
    for line in lines:
        text, color, weight = (line, MUTED, "normal") if isinstance(line, str) else line
        items.append((text, ls, weight, color))
    heights = [size / 72 * 1.38 for _, size, _, _ in items]
    cy = y + h / 2 + sum(heights) / 2 + dy
    for (text, size, weight, color), hh in zip(items, heights):
        ax.text(x + w / 2, cy - hh / 2, text, ha="center", va="center", fontsize=size, fontweight=weight,
                color=color, zorder=6)
        cy -= hh


def arrow(ax, p0, p1, color, lw=1.5, ms=11, conn="arc3", ls="-", z=4, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, connectionstyle=conn,
                        linestyle=ls, zorder=z, shrinkA=0, shrinkB=0)
    )


def strip(ax, x, y, cell_w, cell_h, z=6):
    """18-cell WindowStats vector: 10 GPU segments, 3 CPU fields, 5 counters."""
    i = 0
    for n, color in zip((10, 3, 5), STRIP_COLORS):
        for _ in range(n):
            ax.add_patch(Rectangle((x + i * cell_w, y), cell_w, cell_h, fc=color, ec="white", lw=0.6, zorder=z))
            i += 1


def legend_row(ax, x, y, items, size=9.5, sw=0.16, gap=0.28):
    for label, fc, ec in items:
        rbox(ax, x, y - sw / 2, sw, sw, fc, ec, lw=1.2, r=0.03)
        t = ax.text(x + sw + 0.07, y, label, ha="left", va="center", fontsize=size, color=MUTED)
        x += sw + 0.07 + len(label) * size / 72 * 0.58 + gap
    return t


# ----------------------------------------------------------------------------- data helpers
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def t975(df: int) -> float:
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df))
    except Exception:  # pragma: no cover - scipy is in relax_env, table is a fallback
        table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 8: 2.306, 10: 2.228, 20: 2.086}
        return next((v for k, v in sorted(table.items()) if df <= k), 1.96)


def pct(v: float, nd: int = 2) -> str:
    return f"{v:+.{nd}f}%".replace("-", "−")


def mean_ci(xs) -> tuple[float, float]:
    xs = list(xs)
    m = st.mean(xs)
    if len(xs) < 2:
        return m, float("nan")
    return m, t975(len(xs) - 1) * st.stdev(xs) / math.sqrt(len(xs))


def load_steps(path: Path) -> dict[int, dict[str, np.ndarray]]:
    rows: dict[int, list[dict]] = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[int(r["pair"])].append(r)
    out = {}
    for pair, rs in sorted(rows.items()):
        rs.sort(key=lambda r: int(r["step"]))
        out[pair] = {k: np.array([float(r[k]) for r in rs]) for k in rs[0]}
    return out


def parse_rank_table(log: Path, step: int) -> dict[str, list[float]] | None:
    """Per-rank INFO table printed by relax.utils.straggler.reporter after ``step``."""
    if not log.exists():
        return None
    lines = ANSI.sub("", log.read_text(errors="replace")).splitlines()
    marker = f"[straggler] step={step} per-rank window"
    for i, line in enumerate(lines):
        if marker not in line:
            continue
        header = [c.strip() for c in lines[i + 1].split(")", 1)[-1].split("|")]
        rows = []
        for row in lines[i + 2:]:
            cells = [c.strip() for c in row.split(")", 1)[-1].split("|")]
            if len(cells) != len(header) or not cells[0].isdigit():
                break
            rows.append(dict(zip(header, cells)))
        if rows:
            return {k: [float(r[k]) for r in rows] for k in ("rank", "self_ms", "wait_ms", "late_ms")} | {
                "reason": [r["reason"] for r in rows]
            }
    return None


# ============================================================================= fig_mechanism
def fig_mechanism() -> None:
    W, H = 11.9, 6.25
    fig, ax = canvas(W, H)
    yc = 2.78  # height of the per-window data path (WindowStats -> gather -> detector)

    # column headers
    for x, label in ((2.62, "Each training rank (× N)"), (6.33, "Every K = 10 rollouts"), (9.68, "Primary rank")):
        ax.text(x, 5.98, label, ha="center", va="center", fontsize=12.5, fontweight="bold", color=TEXT)

    # ---- left: one rank, hot path | cold path
    rbox(ax, 0.1, 0.55, 5.05, 5.12, "none", "#9C9C9C", lw=1.1, r=0.1, z=1)
    rbox(ax, 0.2, 0.65, 2.42, 4.92, ORANGE_BAND, "none", r=0.08, z=1)
    rbox(ax, 2.74, 0.65, 2.31, 4.92, BLUE_BAND, "none", r=0.08, z=1)
    ax.text(1.41, 5.36, "HOT PATH", ha="center", fontsize=10.5, fontweight="bold", color=ORANGE)
    ax.text(1.41, 5.13, "every timer start / stop", ha="center", fontsize=9.5, color=ORANGE)
    ax.text(3.895, 5.36, "COLD PATH", ha="center", fontsize=10.5, fontweight="bold", color=BLUE)
    ax.text(3.895, 5.13, "once per rollout end", ha="center", fontsize=9.5, color=BLUE)

    hx, hw = 0.32, 2.18
    hcx = hx + hw / 2
    node(ax, hx, 4.1, hw, 0.66, "up", "Megatron call sites", ["22 timers · unchanged"])
    arrow(ax, (hcx, 4.1), (hcx, 3.64), ORANGE)
    ax.text(hcx + 0.1, 3.87, "start() / stop()", ha="left", va="center", fontsize=9, color=ORANGE_TXT, style="italic")
    node(ax, hx, 2.78, hw, 0.86, "hot", "StragglerTimers",
         ["drop-in config.timers", ("no cudaSynchronize", RED, "bold")], lw=1.6)
    arrow(ax, (hcx, 2.78), (hcx, 2.44), ORANGE)
    node(ax, hx, 1.8, hw, 0.64, "hot", "CUDA event pair", ["+ perf_counter · pool 4096"])
    arrow(ax, (hcx, 1.8), (hcx, 1.56), ORANGE)
    ax.text(hcx + 0.1, 1.68, "push", ha="left", va="center", fontsize=9, color=ORANGE_TXT, style="italic")
    # pending FIFO with a queue glyph: in-flight pairs (hollow) at the back, completed ones (filled) at the front
    fx, fy, fh = hx, 0.78, 0.78
    rbox(ax, fx, fy, hw, fh, ORANGE_FILL, ORANGE, lw=1.3)
    ax.text(hcx, fy + fh - 0.16, "Pending FIFO", ha="center", va="center", fontsize=10.5, fontweight="bold",
            color=ORANGE_TXT, zorder=6)
    cw, n_cells = 0.17, 9
    gx = hcx - n_cells * cw / 2
    for i in range(n_cells):
        done = i >= n_cells - 4
        ax.add_patch(Rectangle((gx + i * cw, fy + 0.3), cw, 0.17, fc=ORANGE if done else "white", ec=ORANGE, lw=0.8,
                               zorder=6))
    ax.text(hcx, fy + 0.15, "≤ 16384 pairs · full → drop oldest", ha="center", va="center", fontsize=8.8,
            color=MUTED, zorder=6)

    cx, cwid = 2.86, 2.07
    ccx = cx + cwid / 2
    node(ax, cx, 0.78, cwid, 0.78, "cold", "drain()", ["event.query() from front", "folds completed pairs"])
    arrow(ax, (hx + hw, fy + 0.385), (cx, fy + 0.385), BLUE, lw=1.7)
    # WindowStats vector
    wy, wh = yc - 0.58, 1.16
    rbox(ax, cx, wy, cwid, wh, BLUE_FILL, BLUE, lw=1.6)
    ax.text(ccx, wy + wh - 0.19, "WindowStats", ha="center", va="center", fontsize=10.5, fontweight="bold",
            color=BLUE_TXT, zorder=6)
    ax.text(ccx, wy + wh - 0.41, "18 × float64 per rank", ha="center", va="center", fontsize=9.5, color=MUTED,
            zorder=6)
    sc = 0.108
    sx = ccx - 18 * sc / 2
    strip(ax, sx, wy + 0.3, sc, 0.2)
    ax.text(sx + 5 * sc, wy + 0.15, "GPU × 10", ha="center", va="center", fontsize=8.8, color=BLUE_TXT, zorder=6)
    ax.text(sx + 14 * sc, wy + 0.15, "CPU · counts", ha="center", va="center", fontsize=8.8, color=BLUE_TXT,
            zorder=6)
    arrow(ax, (ccx, 1.56), (ccx, wy), BLUE)
    node(ax, cx, 4.05, cwid, 0.66, "cold", "gc.callbacks", ["on each GC, inside the step"], lw=1.1)
    arrow(ax, (ccx, 4.05), (ccx, wy + wh), BLUE)

    # ---- middle: Gloo all_gather of N vectors
    mx, mw, my, mh = 5.5, 1.66, yc - 0.9, 1.8
    rbox(ax, mx, my, mw, mh, BLUE_FILL, BLUE, lw=1.6)
    ax.text(mx + mw / 2, my + mh - 0.2, "Gloo all_gather", ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=BLUE_TXT, zorder=6)
    rows = [("r0", 0), ("r1", 1), ("r2", 2), ("⋮", None), ("rN−1", 4)]
    sc2 = 0.06
    lab_x = mx + 0.44
    for k, (lab, _) in enumerate(rows):
        ry = my + mh - 0.55 - k * 0.245
        ax.text(lab_x, ry + 0.065, lab, ha="right", va="center", fontsize=9, color=BLUE_TXT, zorder=6)
        if lab != "⋮":
            strip(ax, lab_x + 0.08, ry, sc2, 0.13)
    arrow(ax, (cx + cwid, yc), (mx, yc), BLUE, lw=1.8)
    for i, line in enumerate(("CPU group, not NCCL", "8 ranks ≈ 1.2 KB", "rank metadata: 1st window")):
        ax.text(mx + mw / 2, my - 0.22 - i * 0.22, line, ha="center", va="center", fontsize=9.3, color=MUTED)

    # ---- right: detector and outputs on the primary rank
    dx, dw, dh = 7.74, 1.24, 1.02
    node(ax, dx, yc - dh / 2, dw, dh, "cold", "Detector", ["numpy", "5 reasons"], lw=1.6)
    arrow(ax, (mx + mw, yc), (dx, yc), BLUE, lw=1.8)
    ax.text((mx + mw + dx) / 2, yc + 0.14, "N × 18", ha="center", va="bottom", fontsize=9, color=BLUE_TXT)
    ox, ow, oh = 9.44, 2.36, 0.72
    outs = [
        (yc + 1.15, "out", "TensorBoard / WandB", ["straggler/* (49 at DP8)"]),
        (yc, "out", "Perfetto timeline", ["one row per rank"]),
        (yc - 1.15, "alert", "WARNING log", ["on reason change", "+ INFO per-rank table"]),
    ]
    for oy, kind, title, lines in outs:
        h = oh + (0.2 if len(lines) > 1 else 0)
        node(ax, ox, oy - h / 2, ow, h, kind, title, lines)
        arrow(ax, (dx + dw, yc), (ox, oy), GREEN if kind == "out" else RED, lw=1.5)

    # ---- footer: off state + legend
    rbox(ax, 0.1, 0.1, 5.05, 0.3, "white", GRAY, lw=1.0, ls=(0, (3, 2)), r=0.05)
    ax.text(2.625, 0.25, "Profiler off: config.timers stays None (identical to upstream)", ha="center",
            va="center", fontsize=9.3, color=GRAY_TXT)
    legend_row(ax, 5.55, 0.25, [
        ("hot path", ORANGE_FILL, ORANGE),
        ("cold path", BLUE_FILL, BLUE),
        ("output", GREEN_FILL, GREEN),
        ("alert", RED_FILL, RED),
        ("unchanged Megatron", GRAY_FILL, GRAY),
    ])
    save(fig, "fig_mechanism")


# ============================================================================= fig_detector
def fig_detector() -> None:
    W, H = 10.8, 7.3
    fig, ax = canvas(W, H)

    # ---- inputs: gathered table -> peer groups -> robust score (with the definitions the ladder uses)
    top_y, top_h = 5.72, 1.48
    node(ax, 0.15, top_y, 2.4, top_h, "cold", "Gathered window",
         ["N ranks × 18", "÷ rollouts → per rollout", ("self = fwd + bwd + optim", TEXT, "bold")])
    node(ax, 2.95, top_y, 3.6, top_h, "cold", "Peer groups",
         ["self, tokens, pp_recv:", ("same PP stage", TEXT, "bold"), "dp_grad_sync:",
          ("same (pp, tp) = DP × CP group", TEXT, "bold")])
    node(ax, 6.95, top_y, 3.7, top_h, "cold", "Robust score vs. peers",
         ["leave-one-out median m and MAD", "z = (x − m) / (1.4826 · MAD)",
          ("↑ significant: z ≥ 3.0 and ≥ 10% above m", TEXT, "bold"),
          ("↓ significant: z ≤ −3.0 and ≥ 10% below m", TEXT, "bold")])
    arrow(ax, (2.55, top_y + top_h / 2), (2.95, top_y + top_h / 2), BLUE)
    arrow(ax, (6.55, top_y + top_h / 2), (6.95, top_y + top_h / 2), BLUE)

    # ---- persistence gate
    gy, gh = 4.82, 0.5
    rbox(ax, 0.15, gy, 10.5, gh, BLUE_BAND, BLUE, lw=1.4, ls=(0, (4, 2)))
    ax.text(5.4, gy + gh / 2, "Persistence gate: a rank must be a candidate for 3 consecutive windows, "
            "otherwise → none", ha="center", va="center", fontsize=10.2, fontweight="bold", color=BLUE_TXT)
    arrow(ax, (8.8, top_y), (8.8, gy + gh), BLUE)
    ax.text(8.7, (top_y + gy + gh) / 2, "every window, every rank", ha="right", va="center", fontsize=9,
            color=MUTED, style="italic")

    # ---- priority ladder
    ax.text(0.15, 4.5, "Priority ladder — first match wins", ha="left", va="center", fontsize=10.8,
            fontweight="bold", color=TEXT)
    ladder = [
        ("self ↑  and  tokens ≥ 10% above peers  and  ms / ktok < 10% above peers", "data_imbalance", None,
         "alert"),
        ("self ↑  and  in-step GC ≥ 5% of self", "cpu_bound", None, "alert"),
        ("self ↑  (neither of the above)", "slow_device", None, "alert"),
        ("dp_grad_sync ↓ in its grad-sync group  and  lateness ≥ 5% of stage-median self",
         "late_arrival", "host-side stall", "alert"),
        ("not a candidate  and  pp_recv ↑  and  excess ≥ 5% of stage-median self",
         "upstream_wait", "victim · not in flagged", "cold"),
    ]
    row_h, row_gap = 0.52, 0.14
    y0 = 4.22
    bx, bw = 1.1, 6.5
    px, pw = 8.0, 2.65
    rail_x = 0.74
    ys = []
    for i, (cond, reason, note, kind) in enumerate(ladder):
        ry = y0 - i * (row_h + row_gap) - row_h
        ys.append(ry)
        root = i < 3
        rbox(ax, bx, ry, bw, row_h, PURPLE_FILL if root else "#F4F4F4", PURPLE if root else GRAY, lw=1.1)
        ax.text(bx + 0.18, ry + row_h / 2, cond, ha="left", va="center", fontsize=9.6, color=TEXT, zorder=6)
        # priority badge
        ax.add_patch(matplotlib.patches.Circle((rail_x, ry + row_h / 2), 0.17, fc="white",
                                               ec=PURPLE if root else GRAY, lw=1.4, zorder=6))
        ax.text(rail_x, ry + row_h / 2, str(i + 1), ha="center", va="center", fontsize=10, fontweight="bold",
                color=PURPLE_TXT if root else GRAY_TXT, zorder=7)
        fc, ec, tc = KIND[kind]
        rbox(ax, px, ry, pw, row_h, fc, ec, lw=1.5, r=0.2)
        if note:
            ax.text(px + pw / 2, ry + row_h * 0.66, reason, ha="center", va="center", fontsize=10.5,
                    fontweight="bold", color=tc, zorder=6, family="DejaVu Sans Mono")
            ax.text(px + pw / 2, ry + row_h * 0.27, note, ha="center", va="center", fontsize=8.8, color=tc, zorder=6)
        else:
            ax.text(px + pw / 2, ry + row_h / 2, reason, ha="center", va="center", fontsize=10.5,
                    fontweight="bold", color=tc, zorder=6, family="DejaVu Sans Mono")
        arrow(ax, (bx + bw, ry + row_h / 2), (px, ry + row_h / 2), ec, lw=1.4)
    # rail between badges ("no" → next rule)
    for a, b in zip(ys[:-1], ys[1:]):
        arrow(ax, (rail_x, a + row_h / 2 - 0.17), (rail_x, b + row_h / 2 + 0.17), GRAY, lw=1.1, ms=8)
    # otherwise → none
    ny = ys[-1] - row_gap - 0.4
    arrow(ax, (rail_x, ys[-1] + row_h / 2 - 0.17), (rail_x, ny + 0.2), GRAY, lw=1.1, ms=8)
    ax.text(rail_x + 0.18, ny + 0.2, "otherwise", ha="left", va="center", fontsize=9.6, color=MUTED)
    rbox(ax, px, ny, pw, 0.4, GRAY_FILL, GRAY, lw=1.2, r=0.18)
    ax.text(px + pw / 2, ny + 0.2, "none", ha="center", va="center", fontsize=10.5, fontweight="bold",
            color=GRAY_TXT, family="DejaVu Sans Mono", zorder=6)
    arrow(ax, (rail_x + 1.05, ny + 0.2), (px, ny + 0.2), GRAY, lw=1.1, ls=(0, (3, 2)))

    # group brackets left of the badges
    for top, bot, label, color in ((ys[0] + row_h, ys[2], "root cause", PURPLE), (ys[3] + row_h, ys[4], "symptom",
                                                                               GRAY)):
        ax.plot([0.44, 0.38, 0.38, 0.44], [top, top, bot, bot], color=color, lw=1.3, solid_capstyle="butt")
        ax.text(0.24, (top + bot) / 2, label, rotation=90, ha="center", va="center", fontsize=9.3, color=color,
                style="italic")
    ax.text(0.15, 0.18,
            "A slow rank also reaches grad-sync late; rules 1–3 fire first, so the root cause is reported, "
            "not the symptom.",
            ha="left", va="center", fontsize=9.3, color=MUTED)
    save(fig, "fig_detector")


# ============================================================================= fig_late_arrival
def fig_late_arrival() -> None:
    fig = plt.figure(figsize=(10.6, 4.7))
    ax = fig.add_axes((0.075, 0.13, 0.6, 0.655))
    bx = fig.add_axes((0.755, 0.13, 0.235, 0.655))
    top = 4.25

    ranks = [0, 1, 2, 3]
    late = 2
    start, stall_end = 0.4, 3.1
    compute = {0: 5.9, 1: 5.7, 2: 5.8, 3: 6.0}
    arrive = {r: (stall_end if r == late else start) + compute[r] for r in ranks}
    last = max(arrive.values())
    xfer = 1.3
    end = last + xfer
    bh = 0.52
    ypos = {r: 3 - r for r in ranks}

    for r in ranks:
        y = ypos[r]
        c0 = stall_end if r == late else start
        if r == late:
            ax.add_patch(Rectangle((start, y - bh / 2), stall_end - start, bh, fc=RED_FILL, ec=RED, lw=1.2,
                                   hatch="///", zorder=3))
            ax.text((start + stall_end) / 2, y, "host-side stall", ha="center", va="center", fontsize=9.5,
                    color=RED_TXT, fontweight="bold", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
        ax.add_patch(Rectangle((c0, y - bh / 2), compute[r], bh, fc=BLUE, ec=BLUE, lw=1, zorder=3))
        ax.text(c0 + compute[r] / 2, y, "fwd + bwd", ha="center", va="center", fontsize=9.5, color="white",
                zorder=5)
        if arrive[r] < last - 1e-9:
            ax.add_patch(Rectangle((arrive[r], y - bh / 2), last - arrive[r], bh, fc=ORANGE_BAND, ec=ORANGE,
                                   lw=1.0, hatch="xx", zorder=3))
            ax.text((arrive[r] + last) / 2, y, "idle", ha="center", va="center", fontsize=9.5, color=ORANGE_TXT,
                    zorder=5, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))
        ax.add_patch(Rectangle((last, y - bh / 2), xfer, bh, fc=ORANGE, ec=ORANGE, lw=1, zorder=3))
        # measured dp_grad_sync bracket
        yb = y - bh / 2 - 0.1
        ax.annotate("", (arrive[r], yb), (end, yb), arrowprops=dict(arrowstyle="|-|", lw=1.0, color=ORANGE_TXT,
                                                                     mutation_scale=3), zorder=4)
    ax.text(last + xfer / 2, ypos[0], "transfer", ha="center", va="center", fontsize=9, color="white",
            rotation=0, zorder=5)
    r3 = ranks[-1]
    ax.text((arrive[r3] + end) / 2, ypos[r3] - bh / 2 - 0.34, "measured dp_grad_sync", ha="center", va="center",
            fontsize=9, color=ORANGE_TXT)
    ax.axvline(last, color=RED, lw=1.2, ls=(0, (4, 2)), zorder=2)
    ax.axvline(end, color=GRAY, lw=1.2, ls=(0, (4, 2)), zorder=2)
    white = dict(boxstyle="round,pad=0.1", fc="white", ec="none")
    ax.text(last - 0.08, 3.68, "last rank arrives", ha="right", va="center", fontsize=9.5, color=RED_TXT,
            bbox=white, zorder=5)
    ax.text(end - 0.08, 4.02, "collective ends on all ranks together", ha="right", va="center", fontsize=9.5,
            color=GRAY_TXT, bbox=white, zorder=5)

    ax.set_xlim(0, end + 0.3)
    ax.set_ylim(-0.85, top)
    ax.set_yticks([ypos[r] for r in ranks], [f"rank {r}" for r in ranks])
    ax.set_xticks([])
    ax.set_xlabel("time within one training step →")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.get_yticklabels()[late].set_color(RED_TXT)
    ax.get_yticklabels()[late].set_fontweight("bold")
    ax.set_title("Timeline (schematic)", loc="left")

    # right: measured interval per rank
    dur = {r: end - arrive[r] for r in ranks}
    peer_med = float(np.median([dur[r] for r in ranks if r != late]))
    for r in ranks:
        bx.barh(ypos[r], dur[r], height=bh, color=RED if r == late else ORANGE, zorder=3)
    bx.axvline(peer_med, color=GRAY, lw=1.2, ls=(0, (4, 2)), zorder=2)
    bx.text(peer_med - 0.05, 3.68, "peer median", ha="right", va="center", fontsize=9.5, color=GRAY_TXT,
            bbox=white, zorder=5)
    yl = ypos[late]
    bx.annotate("", (dur[late], yl), (peer_med, yl), arrowprops=dict(arrowstyle="<->", lw=1.3, color=RED_TXT),
                zorder=4)
    bx.text((dur[late] + peer_med) / 2, yl + 0.1, "late_ms", ha="center", va="bottom", fontsize=9.5,
            color=RED_TXT, fontweight="bold")
    bx.text(0.05, yl - bh / 2 - 0.2, "shortest → late_arrival", ha="left", va="center", fontsize=9.3,
            color=RED_TXT)
    bx.set_ylim(-0.85, top)
    bx.set_xlim(0, max(dur.values()) * 1.12)
    bx.set_yticks([ypos[r] for r in ranks], [f"rank {r}" for r in ranks])
    bx.tick_params(axis="y", length=0)
    bx.spines["left"].set_visible(False)
    bx.set_xticks([])
    bx.set_xlabel("duration")
    bx.set_title("Measured dp_grad_sync", loc="left")

    fig.text(0.075, 0.965, "late_arrival: the last rank to reach grad-sync has the shortest dp_grad_sync",
             ha="left", va="top", fontsize=12.5, fontweight="bold")
    fig.text(0.075, 0.9,
             "The collective ends at the same time on every rank, so each interval = waiting for the last "
             "arrival + transfer. No extra instrumentation.",
             ha="left", va="top", fontsize=9.8, color=MUTED)
    handles = [
        Rectangle((0, 0), 1, 1, fc=BLUE, ec=BLUE),
        Rectangle((0, 0), 1, 1, fc=ORANGE_BAND, ec=ORANGE, hatch="xx"),
        Rectangle((0, 0), 1, 1, fc=ORANGE, ec=ORANGE),
        Rectangle((0, 0), 1, 1, fc=RED_FILL, ec=RED, hatch="///"),
    ]
    labels = ["GPU compute", "idle in grad-sync", "grad-sync transfer", "host-side stall"]
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.07, -0.02), ncol=4, fontsize=9.5,
               handlelength=1.6, columnspacing=1.6)
    save(fig, "fig_late_arrival")


# ============================================================================= fig_injection
INJECTION_FALLBACK = {
    "rank": list(range(8)),
    "self_ms": [759.6, 752.7, 755.5, 745.6, 760.5, 1895.4, 737.0, 740.3],
    "wait_ms": [1183.0, 1190.6, 1188.0, 1197.9, 1185.7, 50.3, 1209.9, 1206.0],
}


def per_rank_stack(ax, ranks, self_ms, wait_ms, flagged, red_self=True):
    # red_self=False keeps the flagged rank's compute blue: for late_arrival the compute is normal
    # and only the host-side gap (drawn by the caller) is the problem.
    x = np.arange(len(ranks))
    colors = [RED if (r in flagged and red_self) else BLUE for r in ranks]
    ax.bar(x, self_ms, width=0.64, color=colors, zorder=3, label="self = fwd + bwd + optim")
    ax.bar(x, wait_ms, width=0.64, bottom=self_ms, color=ORANGE_FILL, edgecolor=ORANGE, lw=1.0, zorder=3,
           label="wait = dp_grad_sync + pp_recv + param gather")
    ax.set_xticks(x, [str(int(r)) for r in ranks])
    ax.set_xlabel("rank")
    ax.set_ylabel("ms / step")
    ax.grid(axis="y", color="#E6E6E6", lw=0.8, zorder=0)
    for i, r in enumerate(ranks):
        if r in flagged:
            ax.get_xticklabels()[i].set_color(RED_TXT)
            ax.get_xticklabels()[i].set_fontweight("bold")
    return x


def titled(ax, title: str, sub: str) -> None:
    ax.set_title(title, loc="left", pad=24)
    ax.text(0, 1.025, sub, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.8, color=MUTED)


def warning_numbers(log: Path, step: int, patterns: dict[str, str]) -> dict[str, float]:
    """Numbers quoted in the ``[straggler] step=<step>`` WARNING line."""
    if not log.exists():
        return {}
    for line in ANSI.sub("", log.read_text(errors="replace")).splitlines():
        if "WARNING" in line and f"[straggler] step={step} " in line:
            return {k: float(m.group(1)) for k, p in patterns.items() if (m := re.search(p, line))}
    return {}


def fig_injection() -> None:
    log = LOGS / "exp0923_a100-inj-gpu5.log"
    tab = parse_rank_table(log, 29) or INJECTION_FALLBACK
    warn = warning_numbers(log, 29, {"z": r"\(z=([\d.]+)\)", "ratio": r"= ([\d.]+)x peers"})
    ranks, self_ms, wait_ms = tab["rank"], np.array(tab["self_ms"]), np.array(tab["wait_ms"])
    flagged = next((i for i, r in enumerate(tab.get("reason", [])) if r != "none"), 5)
    peers = np.delete(self_ms, flagged)
    ratio = warn.get("ratio", self_ms[flagged] / np.median(peers))
    z = warn.get("z", 109.3)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = per_rank_stack(ax, ranks, self_ms, wait_ms, {flagged})
    total = self_ms + wait_ms
    ax.axhline(np.median(peers), color=BLUE, lw=1.0, ls=(0, (4, 2)), zorder=4)
    ax.text(-0.45, np.median(peers) + 25, f"peer median self {np.median(peers):.0f}", ha="left", va="bottom",
            fontsize=9, color=BLUE_TXT, zorder=5)
    ax.set_ylim(0, total.max() * 1.32)
    ax.annotate(f"rank {flagged} → slow_device\nself {self_ms[flagged]:.0f} ms = {ratio:.2f}× peers (z = {z:.0f})",
                xy=(x[flagged], total[flagged] + 20), xytext=(x[flagged], total.max() * 1.18), ha="center",
                va="center", fontsize=10, color=RED_TXT, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3))
    wait_peers = np.median(np.delete(wait_ms, flagged))
    ax.text(x[1] + 0.5, total.max() * 1.18, f"the other 7 ranks wait\n≈ {wait_peers / 1e3:.2f} s / step for it",
            ha="center", va="center", fontsize=9.8, color=ORANGE_TXT)
    titled(ax, "SM-contention injection on GPU 5 (8×A100, DP8)",
           "per-rank window ending at step 29 (steps 20–29), ms / step")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2, fontsize=9.5)
    save(fig, "fig_injection")


# ============================================================================= fig_real_case
def fig_real_case() -> bool:
    log = LOGS / "launch_prof2.log"
    tab = parse_rank_table(log, 14)
    if tab is None:
        print("skip fig_real_case: per-rank table at step 14 not found")
        return False
    warn = warning_numbers(log, 14, {"late": r"grad-sync ([\d.]+) ms/step after", "idle": r"peers idle ([\d.]+)",
                                     "compute": r"GPU compute is ([\d.]+)x"})
    interval = re.search(r"report_interval=(\d+)", ANSI.sub("", log.read_text(errors="replace")))
    k = int(interval.group(1)) if interval else 5
    ranks, self_ms, wait_ms = tab["rank"], np.array(tab["self_ms"]), np.array(tab["wait_ms"])
    late_ms = np.array(tab["late_ms"])
    flagged = next((i for i, r in enumerate(tab["reason"]) if r == "late_arrival"), int(np.argmax(late_ms)))
    peers_self = np.median(np.delete(self_ms, flagged))
    late = warn.get("late", late_ms[flagged])
    idle = warn.get("idle")
    compute = warn.get("compute", self_ms[flagged] / peers_self)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = per_rank_stack(ax, ranks, self_ms, wait_ms, {flagged}, red_self=False)
    total = self_ms + wait_ms
    peer_total = float(np.median(np.delete(total, flagged)))
    # the flagged rank's bracketed time falls short of its peers': the gap is outside every GPU bracket
    ax.add_patch(Rectangle((x[flagged] - 0.32, total[flagged]), 0.64, peer_total - total[flagged], fc="none",
                           ec=RED, lw=1.3, ls=(0, (3, 2)), hatch="///", zorder=4))
    ax.annotate(f"rank {flagged} → late_arrival\nreaches grad-sync {late:.0f} ms / step late\n"
                f"own GPU compute {compute:.2f}× peers",
                xy=(x[flagged] + 0.34, total[flagged] + (peer_total - total[flagged]) * 0.6),
                xytext=(x[flagged] + 2.6, peer_total * 1.2), ha="center", va="center", fontsize=10,
                color=RED_TXT, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3, connectionstyle="arc3,rad=0.25"))
    if idle is not None:
        ax.text(x[6], peer_total * 1.2, f"peers idle {idle:.0f} ms / step\nin grad-sync waiting for it",
                ha="center", va="center", fontsize=9.8, color=ORANGE_TXT)
    ax.set_ylim(0, peer_total * 1.42)
    titled(ax, "Real host-side straggler (upstream default config)",
           f"per-rank window ending at step 14 (report interval {k}), ms / step")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Rectangle((0, 0), 1, 1, fc="none", ec=RED, ls=(0, (3, 2)), hatch="///"))
    labels.append("gap outside GPU brackets (host-side)")
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2, fontsize=9.5)
    save(fig, "fig_real_case")
    return True


# ============================================================================= fig_overhead
def load_blocks(path: Path) -> list[dict] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows or None


def fig_overhead(expil: Path) -> None:
    data = load_steps(LOGS / "exp0923_steps.csv")
    pairs = sorted(data)
    off = np.array([data[p]["step_time_off_s"].mean() for p in pairs])
    on = np.array([data[p]["step_time_on_s"].mean() for p in pairs])
    diff = (on / off - 1) * 100
    m, h = mean_ci(diff)
    off_spread = (off.max() / off.min() - 1) * 100
    steps = data[pairs[0]]["step"]
    blocks = load_blocks(expil)

    fig = plt.figure(figsize=(10.2, 8.4 if blocks else 4.3))
    gs = fig.add_gridspec(2 if blocks else 1, 2, wspace=0.32, hspace=0.42,
                          height_ratios=[1, 0.95] if blocks else None, top=0.9 if blocks else None)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # (a) per-run means
    xs = np.arange(len(pairs))
    a.axhspan(off.min(), off.max(), color=GRAY_FILL, zorder=0)
    a.text(len(pairs) - 0.45, off.max() + 0.0006, f"3 off runs span {off_spread:.1f}%", ha="right", va="bottom",
           fontsize=9.5, color=GRAY_TXT)
    for i in xs:
        a.plot([i - 0.14, i + 0.14], [off[i], on[i]], color="#BBBBBB", lw=1.2, zorder=2)
        a.text(i, max(off[i], on[i]) + 0.0012, pct(diff[i]), ha="center", va="bottom", fontsize=9.5, color=TEXT)
    a.scatter(xs - 0.14, off, s=70, color=OFF_C, zorder=3, label="off")
    a.scatter(xs + 0.14, on, s=70, color=ON_C, zorder=3, label="on")
    a.set_xticks(xs, [f"pair {p}" for p in pairs])
    a.set_xlim(-0.6, len(pairs) - 0.4)
    lo, hi = min(off.min(), on.min()), max(off.max(), on.max())
    a.set_ylim(lo - 0.006, hi + 0.008)
    a.set_ylabel(f"mean step_time, steps {int(steps.min())}–{int(steps.max())} (s)")
    a.set_title("Per-run mean step_time", loc="left")
    a.legend(loc="lower left", ncol=2)
    a.grid(axis="y", color="#EEEEEE", lw=0.8, zorder=0)

    # (b) run-level differences vs run-to-run noise
    white = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9)
    b.axhspan(-off_spread, off_spread, color=GRAY_FILL, zorder=0)
    b.text(-0.5, off_spread - 0.1, f"any two off runs: up to {off_spread:.1f}% apart", ha="left", va="top",
           fontsize=9.3, color=GRAY_TXT)
    for yv in (-0.5, 0.5):
        b.axhline(yv, color=GREEN, lw=1.0, ls=(0, (4, 2)), zorder=1)
    b.text(-0.5, 0.56, "±0.5% target", ha="left", va="bottom", fontsize=9, color=GREEN_TXT)
    b.axhline(0, color="#444444", lw=0.8, zorder=1)
    b.scatter(xs, diff, s=70, color=ON_C, zorder=3)
    for i in xs:
        b.text(i + 0.13, diff[i], pct(diff[i]), ha="left", va="center", fontsize=9.3, bbox=white, zorder=5)
    mx = len(pairs) + 0.35
    b.errorbar([mx], [m], yerr=[[h], [h]], fmt="D", ms=8, color=TEXT, capsize=6, lw=1.6, zorder=4)
    b.text(mx + 0.2, m, f"{pct(m)}\n[{pct(m - h, 1)}, {pct(m + h, 1)}]", ha="left", va="center", fontsize=9.3,
           bbox=white, zorder=5)
    b.text(-0.5, m - h, "n = 3 runs: CI too wide\nto confirm < 0.5% yet", ha="left", va="bottom", fontsize=9.3,
           color=MUTED)
    b.set_xticks(list(xs) + [mx], [f"pair {p}" for p in pairs] + ["mean\n95% CI"])
    b.set_xlim(-0.6, mx + 1.25)
    b.set_ylim(min(-3.8, m - h - 0.45), max(2.7, m + h + 0.5))
    b.set_ylabel("on / off − 1 (%)")
    b.set_title("Run-level difference ≈ run-to-run noise", loc="left")

    if blocks:
        c = fig.add_subplot(gs[1, :])
        d = np.array([float(r["on_over_off_minus_1"]) for r in blocks]) * 100
        bp = np.array([int(r["pair"]) for r in blocks])
        bk = np.array([int(r["block"]) for r in blocks])
        cm, ch = mean_ci(d)
        c.axhline(0, color="#444444", lw=0.8, zorder=1)
        for yv in (-0.5, 0.5):
            c.axhline(yv, color=GREEN, lw=1.0, ls=(0, (4, 2)), zorder=1)
        shades = [BLUE, "#6B9BD8", "#163A66", "#9BBBE6", "#2C4F7C"]
        markers = ["o", "s", "^", "D", "v"]
        pair_ids = sorted(set(bp))
        for j, p in enumerate(pair_ids):
            sel = bp == p
            jitter = (j - (len(pair_ids) - 1) / 2) * 0.18
            c.plot(bk[sel] + jitter, d[sel], markers[j % 5], ms=4.5, color=shades[j % 5], label=f"pair {p}",
                   zorder=3)
        if not math.isnan(ch):
            c.axhspan(cm - ch, cm + ch, color=BLUE_FILL, zorder=0, label="mean ± 95% CI")
        c.axhline(cm, color=BLUE_TXT, lw=1.6, zorder=2)
        c.text(bk.max() + 0.6, 0.52, "±0.5% target", ha="right", va="bottom", fontsize=9, color=GREEN_TXT,
               bbox=white, zorder=5)
        ci_txt = f" ± {ch:.2f}%" if not math.isnan(ch) else ""
        n_pairs = len(pair_ids)
        c.set_title(
            f"In-run crossover: each 10-rollout block measured once on, once off "
            f"({n_pairs} pairs, metrics service off; A/A on 3 hosts in table, not plotted)",
            loc="left",
        )
        c.text(0.995, 0.03, f"pooled mean {pct(cm)}{ci_txt} (95% CI, n = {len(d)} blocks; pair-level CI wider)",
               transform=c.transAxes,
               ha="right", va="bottom", fontsize=9.8, color=BLUE_TXT, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC", lw=0.8), zorder=6)
        c.set_xlabel("10-rollout block")
        c.set_ylabel("on / off − 1 (%)")
        c.legend(loc="upper left", ncol=len(pair_ids) + 1, fontsize=9)
        lim = max(np.abs(d).max() * 1.35, 1.0)
        c.set_ylim(-lim, lim)

    fig.suptitle("Profiler overhead on 8×A100, Qwen3-0.6B SFT DP8", x=0.06, ha="left",
                 y=0.965 if blocks else 1.02, fontsize=12.5, fontweight="bold")
    save(fig, "fig_overhead")


# ============================================================================= fig_loss
def fig_loss() -> None:
    data = load_steps(LOGS / "exp0923_steps.csv")
    pairs = sorted(data)
    steps = data[pairs[0]]["step"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.2), gridspec_kw=dict(wspace=0.28))

    for k, p in enumerate(pairs):
        a.plot(steps, data[p]["loss_off"], color=OFF_C, lw=2.6, alpha=0.55, label="off (3 runs)" if k == 0 else None)
        a.plot(steps, data[p]["loss_on"], color=ON_C, lw=1.0, ls=(0, (3, 2)), label="on (3 runs)" if k == 0 else None)
    a.set_xlabel("step")
    a.set_ylabel("train/loss")
    a.set_title("Same-seed loss, off vs on", loc="left")
    a.legend(loc="upper right")
    a.set_xlim(steps.min() - 0.5, steps.max() + 0.5)

    # off-vs-off envelope: the three off runs share seed and data order
    offs = [data[p]["loss_off"] for p in pairs]
    env = np.max([np.abs(x - y) / x for x, y in itertools.combinations(offs, 2)], axis=0) * 100
    b.fill_between(steps, -env, env, color=GRAY_FILL, lw=0, zorder=0, label="off vs off (± per-step max)")
    shades = [BLUE, "#6B9BD8", "#163A66"]
    markers = ["o", "s", "^"]
    rel_max = 0.0
    for k, p in enumerate(pairs):
        rel = (data[p]["loss_on"] - data[p]["loss_off"]) / data[p]["loss_off"] * 100
        rel_max = max(rel_max, float(np.abs(rel).max()))
        b.plot(steps, rel, markers[k], ms=4, color=shades[k], label=f"pair {p}", zorder=3)
    off_max = float(env.max())
    b.axhline(0, color="#444444", lw=0.8, zorder=1)
    b.set_xlabel("step")
    b.set_ylabel("(on − off) / off  (%)")
    b.set_title("Per-step relative difference", loc="left")
    lim = max(rel_max, off_max) * 1.38
    b.set_ylim(-lim, lim)
    b.set_xlim(steps.min() - 0.5, steps.max() + 0.5)
    b.text(0.99, 0.985, f"max |on − off| / off = {rel_max:.3f}%\nmax |off − off| / off = {off_max:.3f}%",
           transform=b.transAxes, ha="right", va="top", fontsize=9.5, color=TEXT,
           bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC", lw=0.8), zorder=6)
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9.3, handletextpad=0.4,
             columnspacing=1.1)
    fig.suptitle(f"Same-seed loss: on-vs-off difference is the size of off-vs-off noise (8×A100, steps "
                 f"{int(steps.min())}–{int(steps.max())})", x=0.06, ha="left", y=1.02, fontsize=12.5,
                 fontweight="bold")
    save(fig, "fig_loss")
    print(f"  loss: max rel on/off {rel_max:.4f}%, off/off {off_max:.4f}%")


# ============================================================================= main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expil", type=Path, default=LOGS / "expil_blocks.csv",
                    help="in-run crossover blocks CSV (panel skipped if missing or empty)")
    args = ap.parse_args()
    fig_mechanism()
    fig_detector()
    fig_late_arrival()
    fig_injection()
    fig_real_case()
    fig_overhead(args.expil)
    fig_loss()


if __name__ == "__main__":
    main()
