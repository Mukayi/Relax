# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Turn one analyzed straggler window into scalars, timeline events and log
lines.

* scalars -> returned to the actor, which logs them in the same ``tracking_utils.log`` call as
  the step's ``perf/*`` metrics (MetricsService / TensorBoard / WandB / ClearML)
* per-rank segment bars -> ``Timer().records`` so they ride along with the
  existing timeline trace (one Perfetto row per rank, ``pid = TIMELINE_PID_BASE + rank``)
* alerts, recoveries, uncertain ranks -> logger (WARNING on state change, INFO table while any
  alert is active), from the collector's analysis thread via ``log_straggler_window``
"""

from __future__ import annotations

from typing import Any

from relax.utils.logging_utils import get_logger
from relax.utils.metrics.metric_utils import compute_rollout_step
from relax.utils.straggler.detector import WindowReport
from relax.utils.straggler.health import HEALTH_DISABLE
from relax.utils.straggler.stats import GPU_SEGMENTS
from relax.utils.timer import TimelineEvent, Timer


logger = get_logger(__name__)

# Above any real pid (kernel.pid_max <= 2**22) so the straggler rows never merge
# with a live process's track in Perfetto.
TIMELINE_PID_BASE = 1 << 30
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


def _timeline_events(report: WindowReport) -> list[TimelineEvent]:
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
                )
            )
            cursor += ms / 1e3
    return events


def _format_table(report: WindowReport) -> str:
    header = " | ".join(f"{column:>11}" for column in _TABLE_COLUMNS) + " | reason | uncertain"
    lines = [header]
    for row in report.rows:
        cells = []
        for column in _TABLE_COLUMNS:
            value = row[column]
            cells.append(f"{value:>11.1f}" if isinstance(value, float) else f"{str(value):>11}")
        lines.append(" | ".join(cells) + f" | {row['reason']} | {row.get('uncertain', '')}")
    return "\n".join(lines)


def report_straggler_window(args: Any, report: WindowReport) -> dict[str, float]:
    """Emit timeline events for one window and return its scalars (primary
    rank, training thread).

    The scalars are not logged here: with ``--use-metrics-service`` every
    ``tracking_utils.log`` is a synchronous HTTP request on the training
    thread, so the actor hands them to ``log_perf_data`` for the step it is
    finishing instead of paying for a second request. That step can be later
    than the window, hence ``straggler/window/*``.
    """
    metrics = report.metrics
    metrics["straggler/window/first_rollout"] = float(report.first_rollout)
    metrics["straggler/window/last_rollout"] = float(report.last_rollout)
    if getattr(args, "timeline_dump_dir", None) and metrics.get("straggler/health/state", 0.0) < HEALTH_DISABLE:
        # These ride out (and get their step stamped) with the ``log_perf_data``
        # call the actor makes right after this function.
        Timer().records.extend(_timeline_events(report))
    return metrics


def log_straggler_window(args: Any, report: WindowReport) -> None:
    """Log one window's alerts, recoveries, uncertain ranks and summary at the
    window's own step.

    Only touches the logger, so the collector runs it on its analysis thread
    as soon as the analysis ends; the WARNING lines do not wait for the next
    training step.
    """
    step = compute_rollout_step(args, report.last_rollout)
    metrics = report.metrics
    if metrics.get("straggler/health/state", 0.0) >= HEALTH_DISABLE:
        logger.warning("[straggler] step=%d profiler switched off on every rank: %s", step, report.note)
        return

    for alert in report.alerts:
        if alert.new:
            logger.warning("[straggler] step=%d %s", step, alert.message)
    for recovery in report.recovered:
        logger.warning("[straggler] step=%d %s", step, recovery.message)
    for unsure in report.uncertain:
        if unsure.new:
            logger.warning("[straggler] step=%d %s", step, unsure.message)
    if report.alerts:
        logger.info("[straggler] step=%d per-rank window (ms/step):\n%s", step, _format_table(report))
    else:
        logger.info(
            "[straggler] step=%d no straggler (uncertain %d); self median %.1f ms max %.1f ms (rank %d, +%.0f%%), "
            "latest to grad-sync rank %d by %.1f ms (peers idle %.1f ms), pp_stage_imbalance %.2f, "
            "overhead %.2f ms/step",
            step,
            len(report.uncertain),
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
