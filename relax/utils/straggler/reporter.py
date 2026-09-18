# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Push one analyzed straggler window to the existing reporting outlets.

* scalars -> ``tracking_utils.log`` (MetricsService / TensorBoard / WandB / ClearML)
* per-rank segment bars -> ``Timer().records`` so they ride along with the
  existing timeline trace (one Perfetto row per rank, ``pid = TIMELINE_PID_BASE + rank``)
* alerts -> logger (WARNING on state change, INFO table while any alert is active)
"""

from __future__ import annotations

from typing import Any

from relax.utils import tracking_utils
from relax.utils.logging_utils import get_logger
from relax.utils.metrics.metric_utils import compute_rollout_step
from relax.utils.straggler.detector import WindowReport
from relax.utils.straggler.stats import GPU_SEGMENTS
from relax.utils.timer import TimelineEvent, Timer


logger = get_logger(__name__)

TIMELINE_PID_BASE = 100000
_TABLE_COLUMNS: tuple[str, ...] = (
    "rank",
    "tag",
    "host",
    "device",
    "self_ms",
    "wait_ms",
    "late_ms",
    "tokens",
    "ms_per_ktok",
    "gc_ms",
)


def _timeline_events(report: WindowReport, step: int) -> list[TimelineEvent]:
    """Lay each rank's per-step segment averages out as consecutive bars.

    The bars start at the window's wall-clock start so they line up with the
    primary rank's CPU-side ``actor_train`` event in the same trace file.
    """
    events: list[TimelineEvent] = []
    for row in report.rows:
        cursor = report.window_start_wall
        pid = TIMELINE_PID_BASE + int(row["rank"])
        for tid, name in enumerate(GPU_SEGMENTS):
            ms = float(row[name])
            if ms <= 0.0:
                continue
            events.append(
                TimelineEvent(
                    name=f"straggler/{name} {row['tag']}",
                    start_ts=cursor,
                    end_ts=cursor + ms / 1e3,
                    pid=pid,
                    tid=tid,
                    step=step,
                )
            )
            cursor += ms / 1e3
    return events


def _format_table(report: WindowReport) -> str:
    header = " | ".join(f"{column:>11}" for column in _TABLE_COLUMNS) + " | reason"
    lines = [header]
    for row in report.rows:
        cells = []
        for column in _TABLE_COLUMNS:
            value = row[column]
            cells.append(f"{value:>11.1f}" if isinstance(value, float) else f"{str(value):>11}")
        lines.append(" | ".join(cells) + f" | {row['reason']}")
    return "\n".join(lines)


def report_straggler_window(args: Any, rollout_id: int, report: WindowReport) -> None:
    """Emit metrics, timeline events and log lines for one window (primary rank
    only)."""
    step = compute_rollout_step(args, rollout_id)
    metrics = dict(report.metrics)
    metrics["rollout/step"] = step
    tracking_utils.log(args, metrics, step_key="rollout/step")

    if getattr(args, "timeline_dump_dir", None):
        Timer().records.extend(_timeline_events(report, step))

    for alert in report.alerts:
        if alert.new:
            logger.warning("[straggler] step=%d %s", step, alert.message)
    if report.alerts:
        logger.info("[straggler] step=%d per-rank window (ms/step):\n%s", step, _format_table(report))
    else:
        logger.info(
            "[straggler] step=%d no straggler; self median %.1f ms max %.1f ms (rank %d, +%.0f%%), "
            "latest to grad-sync rank %d by %.1f ms (peers idle %.1f ms), pp_stage_imbalance %.2f, "
            "overhead %.2f ms/step",
            step,
            metrics.get("straggler/self/median_ms", 0.0),
            metrics.get("straggler/self/max_ms", 0.0),
            int(metrics.get("straggler/self/max_rank", -1)),
            100.0 * metrics.get("straggler/self/spread", 0.0),
            int(metrics.get("straggler/late/max_rank", -1)),
            metrics.get("straggler/late/max_ms", 0.0),
            metrics.get("straggler/late/peer_idle_ms", 0.0),
            metrics.get("straggler/pp_stage_imbalance", 1.0),
            metrics.get("straggler/self_overhead_ms", 0.0),
        )
