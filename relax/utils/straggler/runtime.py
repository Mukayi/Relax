# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bindings to torch / Megatron, and the process-wide profiler the Megatron
actor holds.

Everything that touches the accelerator (through ``relax.utils.device``, so
CUDA and NPU alike), the process groups or Megatron's parallel state is here,
so the rest of the package runs (and is tested) on CPU with fakes.
"""

from __future__ import annotations

import os
import pickle
import socket
from dataclasses import replace
from datetime import timedelta
from functools import partial
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist

from relax.utils import device
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger
from relax.utils.straggler.collector import StragglerCollector
from relax.utils.straggler.control import LEVELS, LevelPoller
from relax.utils.straggler.detector import DetectorConfig, RankMeta, WindowReport
from relax.utils.straggler.modules import ModuleTimers
from relax.utils.straggler.reporter import log_straggler_window, report_straggler_window
from relax.utils.straggler.timers import StragglerTimers
from relax.utils.straggler.trace import write_window_trace
from relax.utils.timer import Timer


logger = get_logger(__name__)

# Only the role that runs training steps closes them; critic / reference /
# actor_fwd actors would only fill the pending queue and never drain it.
PROFILED_ROLES: frozenset[str] = frozenset({"actor"})


def timing_event_factory() -> Callable[[], Any]:
    """Build timing events of the active accelerator."""
    return partial(device.device_module.Event, enable_timing=True)


def stream_capture_check() -> Callable[[], bool]:
    """The active accelerator's "is the current stream being captured" check,
    resolved once because the timers call it on every bracket.

    A backend without one cannot be capturing through this API.
    """
    if not device.is_available():
        return _never
    return getattr(device.device_module, "is_current_stream_capturing", None) or _never


def _never() -> bool:
    return False


def device_synchronize() -> Callable[[], None]:
    """The active accelerator's synchronize, which anchors a sampled step."""
    return device.device_module.synchronize if device.is_available() else (lambda: None)


_STALL_GATHER_GROUP: Any = None


def init_straggler_gather_group(timeout_s: float) -> None:
    """Create the profiler's dedicated Gloo group, with a timeout far shorter
    than the training one.

    The generic Gloo group is shared with weight sync and other paths, so its
    timeout cannot be lowered for this; a straggler gather is ``world × 272 B``
    of CPU copies (milliseconds), and a rank that vanished must not park the
    rest of the job for ``--distributed-timeout-minutes``. Every member of the
    default group must call this in the same order, so it runs on all ranks of
    the actor group before any gather.
    """
    global _STALL_GATHER_GROUP
    if _STALL_GATHER_GROUP is None:
        _STALL_GATHER_GROUP = dist.new_group(backend="gloo", timeout=timedelta(seconds=timeout_s))


def get_straggler_gather_group() -> Any:
    if _STALL_GATHER_GROUP is None:
        raise RuntimeError("the straggler Gloo group has not been initialized; call install_straggler_profiler first")
    return _STALL_GATHER_GROUP


def gloo_all_gather(values: list[float]) -> list[list[float]]:
    """All-gather one CPU vector over the straggler Gloo group (ranks == global
    ranks).

    Every rank issues the collective whatever fails locally: a rank that cannot
    build its tensor sends NaN in every slot and raises after the gather, and
    the collector skips a window with such a row on every rank.
    """
    group = get_straggler_gather_group()
    error = None
    try:
        local = torch.tensor(values, dtype=torch.float64)
    except Exception as exc:
        error, local = exc, torch.full((len(values),), float("nan"), dtype=torch.float64)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, local, group=group)
    if error is not None:
        raise error
    return [row.tolist() for row in gathered]


def gloo_primary_rank(is_primary: bool) -> int | None:
    """The primary rank's global rank, the same on every rank: one
    ``all_reduce`` over the straggler Gloo group; ``None`` without a
    primary."""
    value = torch.tensor([dist.get_rank() if is_primary else -1], dtype=torch.int64)
    dist.all_reduce(value, op=dist.ReduceOp.MAX, group=get_straggler_gather_group())
    rank = int(value.item())
    return rank if rank >= 0 else None


# A pickled RankMeta is under 400 B even with a 253-character host name.
_META_ROW_BYTES = 1024


def gloo_all_gather_objects(obj: Any) -> list[Any]:
    """All-gather one small picklable object per rank as a fixed-size byte row.

    ``dist.all_gather_object`` pickles before its first collective, so a rank
    whose pickling fails would skip the collectives its peers issue. Here every
    rank issues the same single collective whatever fails locally: a rank that
    cannot encode its object sends an all-zero row, and every rank raises after
    the gather.
    """
    group = get_straggler_gather_group()
    local = torch.zeros(_META_ROW_BYTES, dtype=torch.uint8)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    error = None
    try:
        data = pickle.dumps(obj)
        if len(data) > _META_ROW_BYTES:
            raise ValueError(f"{len(data)} B of rank metadata does not fit the {_META_ROW_BYTES} B row")
        local[: len(data)] = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    except Exception as exc:
        error = exc
    dist.all_gather(gathered, local, group=group)
    if error is not None:
        raise error
    missing = [rank for rank, row in enumerate(gathered) if not row.any()]
    if missing:
        raise RuntimeError(f"ranks {missing} sent no metadata")
    # pickle.loads ignores the zero padding after the pickled bytes.
    return [pickle.loads(row.numpy().tobytes()) for row in gathered]


def gloo_gather_objects_to(obj: Any, dst: int) -> list[Any] | None:
    """Gather one picklable object per rank on rank ``dst`` (``None``
    elsewhere) over the straggler Gloo group.

    The object is pickled before the collective so that a rank whose pickling
    fails still joins it, with ``None``; it raises after the gather.
    """
    group = get_straggler_gather_group()
    error = None
    try:
        data = pickle.dumps(obj)
    except Exception as exc:
        error, data = exc, pickle.dumps(None)
    gathered = [None] * dist.get_world_size(group) if dist.get_rank() == dst else None
    dist.gather_object(data, gathered, dst=dst, group=group)
    if error is not None:
        raise error
    return [pickle.loads(item) for item in gathered] if gathered is not None else None


def local_rank_meta() -> RankMeta:
    from megatron.core import mpu

    from relax.utils.distributed_utils import get_gloo_group

    return RankMeta(
        # Rank within the Gloo group == global rank; it indexes the gathered table.
        rank=dist.get_rank(get_gloo_group()),
        dp=mpu.get_data_parallel_rank(with_context_parallel=True),
        tp=mpu.get_tensor_model_parallel_rank(),
        pp=mpu.get_pipeline_model_parallel_rank(),
        cp=mpu.get_context_parallel_rank(),
        ep=mpu.get_expert_model_parallel_rank(),
        host=socket.gethostname(),
        device=device.device_module.current_device() if device.is_available() else -1,
    )


class StragglerProfiler:
    """One collector plus the per-step reporting around ``log_perf_data``.

    ``step_metrics`` never raises: a failing reporter is counted by the
    collector's health like any other profiler failure.
    """

    def __init__(self, collector: StragglerCollector, args: Any) -> None:
        self.collector = collector
        self._args = args

    def step_metrics(self, rollout_id: int, tokens: int) -> dict[str, float] | None:
        """Close this step (all ranks; a Gloo collective every window) and
        return the scalars of the latest window whose analysis has finished,
        for this step's ``log_perf_data``.

        If several windows are ready at once, each still reaches the timeline
        and logs, but only the scalars of the latest one whose reporter
        succeeded are returned (one metrics call per step carries one value per
        key). Each window it supersedes is logged as a WARNING and counted as a
        profiler error; a window whose reporter fails is counted once, as that
        failure.
        """
        self.collector.add_tokens(tokens)
        reports = self.collector.end_step(rollout_id, final=rollout_id + 1 == self._args.num_rollout)
        metrics: dict[str, float] | None = None
        returned: WindowReport | None = None
        superseded: list[WindowReport] = []
        for report in reports:
            try:
                scalars = report_straggler_window(self._args, report)
            except Exception as exc:
                self.collector.on_error("report", exc)
                continue
            if returned is not None:
                superseded.append(returned)
            returned, metrics = report, scalars
        if superseded:
            logger.warning(
                "[straggler] rollout %d: %d windows ready at once; only rollouts %d-%d reach the metrics",
                rollout_id,
                len(reports),
                returned.first_rollout,
                returned.last_rollout,
            )
            for dropped in superseded:
                self.collector.on_error(
                    "report",
                    RuntimeError(
                        f"scalars of rollouts {dropped.first_rollout}-{dropped.last_rollout} superseded by a later "
                        "window"
                    ),
                )
        return metrics


def _cpu_events(start: float, end: float) -> list[dict[str, Any]]:
    """This process's finished ``Timer`` events that overlap the wall-clock
    span ``[start, end]``, without their call stacks; timers still running when
    the step ended are not among them."""
    events = []
    for record in reversed(Timer().records):
        if record.end_ts < start:
            break
        if record.start_ts <= end and not record.name.startswith("straggler/"):
            event = record.to_trace_event()
            event.get("args", {}).pop("call_stack", None)
            events.append(event)
    return events


def trace_dir(args: Any) -> str:
    """Where the primary rank writes sampled-step traces (empty: no tracing).

    ``RELAX_STRAGGLER_TRACE_DIR``, else ``straggler/`` under the timeline dump
    directory.
    """
    timeline_dir = getattr(args, "timeline_dump_dir", None)
    return Envs.RELAX_STRAGGLER_TRACE_DIR or (os.path.join(timeline_dir, "straggler") if timeline_dir else "")


_PROFILER: StragglerProfiler | None = None


def install_straggler_profiler(role: str, args: Any) -> StragglerProfiler | None:
    """Create the process-wide profiler if it is enabled and ``role`` runs
    training steps; ``None`` otherwise.

    Must run after Megatron parallel state and Relax's Gloo group exist, and
    before the optimizer is built (its config takes ``straggler_timers``).
    """
    global _PROFILER
    if not Envs.RELAX_STRAGGLER_PROFILER or role not in PROFILED_ROLES:
        return None
    if _PROFILER is not None:
        return _PROFILER
    from relax.utils.training.train_metric_utils import is_metrics_primary_rank

    # Every rank of the actor group reaches this call in the same order, which
    # is what ``dist.new_group`` requires.
    init_straggler_gather_group(Envs.RELAX_STRAGGLER_GATHER_TIMEOUT_S)

    # The window scalars are logged through ``log_perf_data``, which drops them
    # on every other rank.
    is_primary = is_metrics_primary_rank()
    traces_to = trace_dir(args)
    # Only a tracing run gathers samples; every rank takes this branch alike, so the all-reduce is collective.
    primary_rank = gloo_primary_rank(is_primary) if traces_to else None
    poller: LevelPoller | None = None
    collector = StragglerCollector(
        rank_meta=replace(local_rank_meta(), primary=is_primary),
        is_primary=is_primary,
        report_interval=Envs.RELAX_STRAGGLER_REPORT_INTERVAL,
        detector_config=DetectorConfig(
            z_threshold=Envs.RELAX_STRAGGLER_Z_THRESHOLD,
            rel_threshold=Envs.RELAX_STRAGGLER_REL_THRESHOLD,
            persist_windows=Envs.RELAX_STRAGGLER_PERSIST_WINDOWS,
            recover_windows=Envs.RELAX_STRAGGLER_RECOVER_WINDOWS,
        ),
        event_factory=timing_event_factory(),
        gather=gloo_all_gather,
        gather_objects=gloo_all_gather_objects,
        is_capturing=stream_capture_check(),
        on_report=partial(log_straggler_window, args),
        trace=bool(traces_to),
        synchronize=device_synchronize(),
        gather_to=gloo_gather_objects_to,
        primary_rank=primary_rank,
        trace_sink=partial(write_window_trace, traces_to, Envs.RELAX_STRAGGLER_TRACE_KEEP) if traces_to else None,
        cpu_events=_cpu_events,
        level=initial_level(bool(traces_to)),
        # Read at window closes; the poller is created below, once there is a collector to report.
        level_request=lambda: poller.requested if poller is not None else None,
        interleaved=(getattr(args, "virtual_pipeline_model_parallel_size", None) or 1) > 1,
    )
    if is_primary and getattr(args, "use_metrics_service", False):
        from relax.utils.utils import get_serve_url

        poller = LevelPoller(
            get_serve_url(route_prefix="/metrics"), collector.status, Envs.RELAX_STRAGGLER_CONTROL_POLL_S
        )
    _PROFILER = StragglerProfiler(collector, args)
    logger.info(
        "Straggler profiler enabled: report_interval=%d primary=%s level=%s control=%s trace_dir=%s meta=%s",
        collector.report_interval,
        is_primary,
        LEVELS[collector.level],
        "metrics service" if poller is not None else "-",
        traces_to or "-",
        collector.rank_meta,
    )
    return _PROFILER


def register_straggler_model(model_chunks: Sequence[Any]) -> None:
    """Hand the model chunks to the profiler for the ``modules`` level; a no-op
    without a profiler.

    Their layers are hooked only while a sampled step at that level runs.
    """
    if _PROFILER is None:
        return
    collector = _PROFILER.collector
    try:
        timers = ModuleTimers(model_chunks, collector)
    except Exception as exc:
        collector.on_error("modules", exc)
        return
    collector.module_timers = timers
    if collector.is_primary:
        counts = timers.counts()
        logger.info(
            "[straggler] modules level: %d blocks on this rank (%s)",
            counts.total(),
            ", ".join(f"{name} {count}" for name, count in counts.items()),
        )


def initial_level(trace_on: bool) -> int:
    """The level a run starts at: ``RELAX_STRAGGLER_LEVEL``, else detail when
    traces have somewhere to go and coarse otherwise; never ``modules`` unless
    asked for."""
    level = Envs.RELAX_STRAGGLER_LEVEL
    return level if level >= 0 else (2 if trace_on else 1)


def _timers_for(collector: StragglerCollector, phase: str) -> StragglerTimers:
    if phase == "train":
        return collector.train_timers
    if phase == "forward_only":
        return collector.forward_only_timers
    raise ValueError(f"unknown straggler phase {phase!r}; expected 'train' or 'forward_only'")


def straggler_timers(phase: str) -> StragglerTimers | None:
    """The ``config.timers`` value for ``phase``: ``None`` without a profiler,
    which is what Relax assigns anyway."""
    return _timers_for(_PROFILER.collector, phase) if _PROFILER is not None else None


def interleaved_schedule_table(
    model: Sequence[Any], num_microbatches: int, forward_only: bool
) -> Sequence[tuple[int, int]]:
    """Megatron's interleaved schedule table for one forward-backward call,
    from the inputs its schedule reads: the model's own config and chunk count,
    not the launch arguments (Bridge does not copy every argument into the
    model config).

    Empty for training under combined 1F1B, which merges forward and backward
    steps.
    """
    from megatron.core.pipeline_parallel.schedules import get_schedule_table
    from megatron.core.utils import get_model_config

    config = get_model_config(model[0])
    if config.overlap_moe_expert_parallel_comm and not forward_only:
        return ()
    return get_schedule_table(num_microbatches, len(model), config.microbatch_group_size_per_vp_stage)


def note_schedule(model: Sequence[Any], num_microbatches: int, forward_only: bool) -> None:
    """Report the forward-backward call about to run, so that under interleaved
    PP each bracket takes its micro-batch and model chunk; a no-op without a
    profiler."""
    if _PROFILER is not None:
        _PROFILER.collector.note_schedule(
            partial(interleaved_schedule_table, model, num_microbatches, forward_only), forward_only
        )


def note_microbatch_tokens(total_lengths: Any) -> None:
    """Report the sample lengths of the training micro-batch whose forward
    bracket is open; a no-op without a profiler.

    Only a host-side list is summed: summing a device tensor would synchronize.
    """
    if _PROFILER is not None and isinstance(total_lengths, (list, tuple)):
        _PROFILER.collector.note_tokens(int(sum(total_lengths)))
