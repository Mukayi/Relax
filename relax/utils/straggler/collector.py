# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Per-rank event collection, lazy readout and cross-rank aggregation.

Hot path (called from Megatron's timer call sites and module hooks):
``record_event`` + ``push`` only touch Python lists and record events on the
current stream. Cold path (once per training step, from the actor):
``end_step`` reads back completed event pairs with ``event.query()`` -- never
``synchronize`` -- and every ``report_interval`` steps gathers one small vector
per rank over the Gloo group and runs the detector on the primary rank.
"""

from __future__ import annotations

import gc
import socket
from collections import deque
from time import perf_counter, time
from typing import Any, Callable, Sequence

from relax.utils.logging_utils import get_logger
from relax.utils.straggler.detector import DetectorConfig, DetectorState, RankMeta, WindowReport, analyze_window
from relax.utils.straggler.stats import (
    FIELD_INDEX,
    FORWARD_ONLY_TIMER_SEGMENTS,
    MEGATRON_TIMER_SEGMENTS,
    WindowStats,
)
from relax.utils.straggler.timers import EventPool, StragglerTimers, cuda_timing_event


logger = get_logger(__name__)

_FWD_INDEX = FIELD_INDEX["fwd"]
_BWD_INDEX = FIELD_INDEX["bwd"]
_CPU_FWD_INDEX = FIELD_INDEX["cpu_fwd"]
_CPU_BWD_INDEX = FIELD_INDEX["cpu_bwd"]
_NUM_FWD_INDEX = FIELD_INDEX["num_fwd"]

GatherFn = Callable[[list[float]], list[list[float]]]
GatherObjectsFn = Callable[[Any], list[Any]]


def gloo_all_gather(values: list[float]) -> list[list[float]]:
    """All-gather one CPU vector over Relax's Gloo group (ranks == global
    ranks)."""
    import torch
    import torch.distributed as dist

    from relax.utils.distributed_utils import get_gloo_group

    group = get_gloo_group()
    local = torch.tensor(values, dtype=torch.float64)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, local, group=group)
    return [row.tolist() for row in gathered]


def gloo_all_gather_objects(obj: Any) -> list[Any]:
    import torch.distributed as dist

    from relax.utils.distributed_utils import get_gloo_group

    group = get_gloo_group()
    gathered: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, obj, group=group)
    return gathered


def local_rank_meta() -> RankMeta:
    """Identity of this rank from Megatron's parallel state."""
    import torch
    import torch.distributed as dist
    from megatron.core import mpu

    return RankMeta(
        rank=dist.get_rank(),
        dp=mpu.get_data_parallel_rank(with_context_parallel=True),
        tp=mpu.get_tensor_model_parallel_rank(),
        pp=mpu.get_pipeline_model_parallel_rank(),
        cp=mpu.get_context_parallel_rank(),
        ep=mpu.get_expert_model_parallel_rank(),
        host=socket.gethostname(),
        device=torch.cuda.current_device() if torch.cuda.is_available() else -1,
    )


class StragglerCollector:
    """Owns the event pool, the pending queue and the current window."""

    def __init__(
        self,
        *,
        rank_meta: RankMeta,
        is_primary: bool,
        report_interval: int = 10,
        detector_config: DetectorConfig | None = None,
        event_factory: Callable[[], Any] = cuda_timing_event,
        pool_size: int = 4096,
        max_pending: int = 16384,
        gather: GatherFn = gloo_all_gather,
        gather_objects: GatherObjectsFn = gloo_all_gather_objects,
        clock: Callable[[], float] = perf_counter,
        register_gc_callback: bool = True,
    ) -> None:
        if report_interval < 1:
            raise ValueError("report_interval must be >= 1")
        self.rank_meta = rank_meta
        self.is_primary = is_primary
        self.report_interval = report_interval
        self.detector_config = detector_config or DetectorConfig()
        self.detector_state = DetectorState()
        self._pool = EventPool(pool_size, event_factory)
        self._max_pending = max_pending
        self._pending: deque[tuple[int, Any, Any, float]] = deque()
        self._window = WindowStats()
        self._steps_in_window = 0
        self.window_start_wall = time()
        self._gather = gather
        self._gather_objects = gather_objects
        self._clock = clock
        self._all_meta: list[RankMeta] | None = None
        self._gc_start: float | None = None
        self._gc_registered = False
        if register_gc_callback:
            gc.callbacks.append(self._on_gc)
            self._gc_registered = True
        # One timers object per phase so the same Megatron call sites land in
        # different buckets during training vs. forward-only passes.
        self.train_timers = StragglerTimers(self, MEGATRON_TIMER_SEGMENTS)
        self.forward_only_timers = StragglerTimers(self, FORWARD_ONLY_TIMER_SEGMENTS)

    # ---- hot path -----------------------------------------------------------------------------------------------

    def record_event(self) -> Any:
        event = self._pool.acquire()
        event.record()
        return event

    def push(self, segment_index: int, start_event: Any, end_event: Any, cpu_ms: float) -> None:
        if len(self._pending) >= self._max_pending:
            _, old_start, old_end, _ = self._pending.popleft()
            self._pool.release(old_start)
            self._pool.release(old_end)
            self._window.add("dropped", 1.0)
        self._pending.append((segment_index, start_event, end_event, cpu_ms))

    def add_tokens(self, tokens: int) -> None:
        self._window.add("tokens", float(tokens))

    def _on_gc(self, phase: str, info: dict[str, Any]) -> None:
        if phase == "start":
            self._gc_start = self._clock()
        elif self._gc_start is not None:
            self._window.add("gc", (self._clock() - self._gc_start) * 1e3)
            self._gc_start = None

    # ---- cold path ----------------------------------------------------------------------------------------------

    @property
    def pending_events(self) -> int:
        return len(self._pending)

    @property
    def window(self) -> WindowStats:
        return self._window

    def drain(self) -> int:
        """Fold every completed event pair into the window; keep the rest.

        Returns the number of pairs folded.
        """
        started = self._clock()
        folded = 0
        remaining: deque[tuple[int, Any, Any, float]] = deque()
        window = self._window
        for item in self._pending:
            segment_index, start_event, end_event, cpu_ms = item
            if not end_event.query():
                remaining.append(item)
                continue
            window.add_index(segment_index, start_event.elapsed_time(end_event))
            if segment_index == _FWD_INDEX:
                window.add_index(_CPU_FWD_INDEX, cpu_ms)
                window.add_index(_NUM_FWD_INDEX, 1.0)
            elif segment_index == _BWD_INDEX:
                window.add_index(_CPU_BWD_INDEX, cpu_ms)
            self._pool.release(start_event)
            self._pool.release(end_event)
            folded += 1
        self._pending = remaining
        window.add("overhead", (self._clock() - started) * 1e3)
        return folded

    def end_step(self, step: int) -> WindowReport | None:
        """Close one training step; every ``report_interval`` steps gather and
        analyze.

        All ranks must call this at the same logical step (it contains a
        collective). Returns the report on the primary rank when a window
        closes, ``None`` otherwise.
        """
        self.drain()
        self._window.add("num_steps", 1.0)
        self._steps_in_window += 1
        if self._steps_in_window < self.report_interval:
            return None
        started = self._clock()
        if self._all_meta is None:
            self._all_meta = list(self._gather_objects(self.rank_meta))
        table = self._gather(self._window.as_list())
        report = None
        if self.is_primary:
            report = analyze_window(table, self._all_meta, self.detector_config, self.detector_state)
            report.metrics["straggler/gather_ms"] = (self._clock() - started) * 1e3
            report.window_start_wall = self.window_start_wall
            report.window_end_wall = time()
        self._window.reset()
        self._steps_in_window = 0
        self.window_start_wall = time()
        return report

    @property
    def all_meta(self) -> Sequence[RankMeta] | None:
        return self._all_meta

    def close(self) -> None:
        if self._gc_registered:
            try:
                gc.callbacks.remove(self._on_gc)
            except ValueError:
                pass
            self._gc_registered = False


# ---- process-wide instance ----------------------------------------------------------------------------------------

_COLLECTOR: StragglerCollector | None = None


def install_straggler_collector(args: Any) -> StragglerCollector | None:
    """Create the process-wide collector if the profiler is enabled.

    Must run after Megatron parallel state and Relax's Gloo group exist (i.e.
    from ``MegatronTrainRayActor.init``). Returns ``None`` when disabled so the
    caller's hot path stays a plain attribute check.
    """
    global _COLLECTOR
    from relax.utils.env import Envs

    if not Envs.RELAX_STRAGGLER_PROFILER:
        return None
    if _COLLECTOR is not None:
        return _COLLECTOR
    from megatron.core import mpu

    is_primary = (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.is_pipeline_last_stage()
        and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
    )
    _COLLECTOR = StragglerCollector(
        rank_meta=local_rank_meta(),
        is_primary=is_primary,
        report_interval=Envs.RELAX_STRAGGLER_REPORT_INTERVAL,
        detector_config=DetectorConfig(
            z_threshold=Envs.RELAX_STRAGGLER_Z_THRESHOLD,
            rel_threshold=Envs.RELAX_STRAGGLER_REL_THRESHOLD,
            persist_windows=Envs.RELAX_STRAGGLER_PERSIST_WINDOWS,
        ),
    )
    logger.info(
        "Straggler profiler enabled: report_interval=%d primary=%s meta=%s",
        _COLLECTOR.report_interval,
        is_primary,
        _COLLECTOR.rank_meta,
    )
    return _COLLECTOR


def get_straggler_collector() -> StragglerCollector | None:
    return _COLLECTOR


def straggler_timers(phase: str) -> StragglerTimers | None:
    """Value for ``config.timers``: the phase-specific timers, or ``None`` when
    the profiler is off (which is exactly what Relax assigns today)."""
    if _COLLECTOR is None:
        return None
    if phase == "train":
        return _COLLECTOR.train_timers
    if phase == "forward_only":
        return _COLLECTOR.forward_only_timers
    raise ValueError(f"unknown straggler phase {phase!r}")
