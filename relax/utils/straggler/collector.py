# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Per-rank event collection, lazy readout and the per-window gather.

Hot path (Megatron's timer call sites): ``record_event`` / ``push`` record
events on the current stream and append to a queue. Once per step ``end_step``
reads completed pairs with ``event.query()`` (never ``synchronize``); every
``report_interval`` steps all ranks gather one small vector and the primary
rank queues the table for the background analysis, whose reports a later
``end_step`` returns.

A "step" in this package is one ``end_step`` call, i.e. one rollout, which may
contain several optimizer steps.

With tracing on, the first step of every window after the first is also
placed on a real time axis (see ``trace``): it starts with one device
synchronize and an anchor event, its brackets keep their offsets from that
anchor, and the window close gathers every rank's sample to the primary rank.
"""

from __future__ import annotations

import gc
import threading
from array import array
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter, time
from typing import Any, Callable

from relax.utils.logging_utils import get_logger
from relax.utils.straggler.detector import DetectorConfig, DetectorState, RankMeta, WindowReport, analyze_window
from relax.utils.straggler.health import HEALTH_DISABLE, HealthConfig, ProfilerHealth
from relax.utils.straggler.stats import (
    FIELD_INDEX,
    FORWARD_ONLY_TIMER_SEGMENTS,
    MEGATRON_TIMER_SEGMENTS,
    PP_PHASE_TIMERS,
    WindowStats,
    disabled_row,
)
from relax.utils.straggler.timers import EventPool, StragglerTimers
from relax.utils.straggler.trace import RECORD_WIDTH, RankTrace
from relax.utils.straggler.worker import AnalysisWorker


logger = get_logger(__name__)

_FWD_INDEX = FIELD_INDEX["fwd"]
_BWD_INDEX = FIELD_INDEX["bwd"]
_LP_FWD_INDEX = FIELD_INDEX["lp_fwd"]
# Segments whose brackets are numbered within the step (micro-batch order).
_NUMBERED = (_FWD_INDEX, _BWD_INDEX, _LP_FWD_INDEX)
_CPU_FWD_INDEX = FIELD_INDEX["cpu_fwd"]
_CPU_BWD_INDEX = FIELD_INDEX["cpu_bwd"]
_NUM_FWD_INDEX = FIELD_INDEX["num_fwd"]
_GC_INDEX = FIELD_INDEX["gc"]
_DROPPED_INDEX = FIELD_INDEX["dropped"]
_ERRORS_INDEX = FIELD_INDEX["errors"]
_HEALTH_INDEX = FIELD_INDEX["health"]
_MB_FWD_MAX_INDEX = FIELD_INDEX["mb_fwd_max"]
_MB_FWD_MIN_INDEX = FIELD_INDEX["mb_fwd_min"]
_MB_BWD_MAX_INDEX = FIELD_INDEX["mb_bwd_max"]
_MB_BWD_MIN_INDEX = FIELD_INDEX["mb_bwd_min"]
_MB_FWD_ARGMAX_INDEX = FIELD_INDEX["mb_fwd_argmax"]
_MB_FWD_ARGMAX_TOKENS_INDEX = FIELD_INDEX["mb_fwd_argmax_tokens"]
_MB_TOKENS_MAX_INDEX = FIELD_INDEX["mb_tokens_max"]
_MB_TOKENS_MIN_INDEX = FIELD_INDEX["mb_tokens_min"]

# An analysis takes milliseconds (35 ms at 2048 ranks); this only bounds a stuck worker at the end of a run.
_FINAL_FLUSH_TIMEOUT_S = 10.0

GatherFn = Callable[[list[float]], list[list[float]]]
GatherObjectsFn = Callable[[Any], list[Any]]
# (object, destination rank) -> every rank's object on the destination, None elsewhere.
GatherToFn = Callable[[Any, int], list[Any] | None]
# (gathered traces, rank metadata, first rollout of the window) -> trace scalars.
TraceSink = Callable[[list[RankTrace | None], list[RankMeta], int], dict[str, float]]


@dataclass
class _WindowJob:
    """One closed window on its way to the analysis; ``ready`` carries a report
    that needs no analysis (the switch-off notice) through the same queue so
    reports stay in window order."""

    table: list[list[float]] | None
    meta: list[RankMeta]
    window_start_wall: float
    first_rollout: int
    last_rollout: int
    closed_wall: float = 0.0
    gather_ms: float = 0.0
    ready: WindowReport | None = None
    traces: list[RankTrace | None] | None = None


@dataclass
class _TraceCapture:
    """The sampled step being read out: its anchor event and wall-clock time,
    the sequence numbers of its first pair and one past its last (set when the
    step ends), its brackets so far and how many of them were lost."""

    anchor: Any
    wall: float
    first: int
    end: int | None = None
    records: array = field(default_factory=lambda: array("f"))
    lost: int = 0

    def covers(self, seq: int) -> bool:
        return self.first <= seq and (self.end is None or seq < self.end)


class StragglerCollector:
    """Queue this rank's timing events, fold them lazily, and gather one window
    every ``report_interval`` steps.

    Torch-free: CUDA events, stream-capture detection and the gathers are
    injected (``runtime`` passes the real ones, tests pass fakes).
    """

    def __init__(
        self,
        *,
        rank_meta: RankMeta,
        is_primary: bool,
        event_factory: Callable[[], Any],
        gather: GatherFn,
        gather_objects: GatherObjectsFn,
        is_capturing: Callable[[], bool],
        report_interval: int = 10,
        detector_config: DetectorConfig | None = None,
        pool_size: int = 4096,
        max_pending: int = 16384,
        clock: Callable[[], float] = perf_counter,
        register_gc_callback: bool = True,
        health_config: HealthConfig | None = None,
        background: bool = True,
        on_report: Callable[[WindowReport], None] | None = None,
        max_backlog: int = 4,
        wall_clock: Callable[[], float] = time,
        trace: bool = False,
        synchronize: Callable[[], None] = lambda: None,
        gather_to: GatherToFn | None = None,
        trace_sink: TraceSink | None = None,
        max_trace_brackets: int = 4096,
    ) -> None:
        if report_interval < 1:
            raise ValueError(
                f"straggler report_interval must be >= 1, got {report_interval}: a window closes every "
                "report_interval rollouts. Set RELAX_STRAGGLER_REPORT_INTERVAL to a positive integer."
            )
        self.rank_meta = rank_meta
        self.is_primary = is_primary
        self.report_interval = report_interval
        self.detector_config = detector_config or DetectorConfig()
        self.detector_state = DetectorState()
        self._pool = EventPool(pool_size, event_factory)
        self._max_pending = max_pending
        # (segment, PP phase field or -1, start event, end event, CPU ms, micro-batch or -1, tokens or -1);
        # tokens only for a training forward.
        self._pending: deque[tuple[int, int, Any, Any, float, int, float]] = deque()
        # Pairs ever pushed and ever popped (folded or dropped): the sequence number of the queue head.
        self._pushed = 0
        self._popped = 0
        # Brackets of each numbered segment pushed in this step, and the token
        # count reported for the training forward bracket that is open (-1 when none was).
        self._step_numbers = dict.fromkeys(_NUMBERED, 0)
        self._next_tokens = -1.0
        self._trace_enabled = trace
        self._synchronize = synchronize
        self._gather_to = gather_to
        self._trace_sink = trace_sink
        self._max_trace_brackets = max_trace_brackets
        # The first window has no sample: its first step carries the warm-up.
        self._trace_next = False
        self._trace: _TraceCapture | None = None
        self._trace_done: RankTrace | None = None
        self._primary_rank: int | None = None
        # Primary rank only: the wall-clock span of the step traced last, until
        # the runtime hands over the CPU events in it.
        self._cpu_span: tuple[float, float] | None = None
        self._cpu_events: list[dict[str, Any]] = []
        self._window = WindowStats()
        self._wall = wall_clock
        self.window_start_wall = wall_clock()
        self._gather = gather
        self._gather_objects = gather_objects
        self._clock = clock
        self._is_capturing = is_capturing
        self._all_meta: list[RankMeta] | None = None
        self._meta_gathered = False
        # True from the first bracket of a step until ``end_step``; GC pauses are
        # only attributed to the step while it is set (not during train_wait,
        # offload / clear_memory, etc.).
        self._step_open = False
        self._gc_start: float | None = None
        # Thread that first calls record_event; only that thread's GC pauses
        # are attributed to the training window (gc.callbacks fire on the
        # triggering thread, so a background GC must not corrupt _gc_start).
        self._train_thread: int | None = None
        self.health = ProfilerHealth(health_config)
        # Plain attribute (not ``health.disabled``) because ``record_event`` checks it on every bracket.
        self._disabled = False
        # Appended by the worker, drained by ``end_step`` (deque append / popleft are atomic).
        self._reports: deque[WindowReport] = deque()
        self._on_report = on_report
        self._max_backlog = max_backlog
        self._window_first_rollout: int | None = None
        self._window_last_rollout = -1
        self._worker = AnalysisWorker(self._handle_job) if background and is_primary else None
        if register_gc_callback:
            gc.callbacks.append(self._on_gc)
        # One timers object per phase so the same Megatron call sites land in
        # different buckets during training vs. forward-only passes.
        self.train_timers = StragglerTimers(self, MEGATRON_TIMER_SEGMENTS, PP_PHASE_TIMERS)
        self.forward_only_timers = StragglerTimers(self, FORWARD_ONLY_TIMER_SEGMENTS)

    # ---- hot path -----------------------------------------------------------------------------------------------

    def record_event(self) -> Any:
        """Record a pooled event on the current stream and open the step for GC
        attribution.

        Returns ``None`` (the bracket is skipped) once the profiler is
        disabled, and during CUDA graph capture, where ``elapsed_time`` would
        be meaningless on replay.
        """
        if self._disabled or self._is_capturing():
            return None
        if not self._step_open:
            self._step_open = True
            if self._trace_next:
                self._begin_trace()
        if self._train_thread is None:
            self._train_thread = threading.get_ident()
        event = self._pool.acquire()
        event.record()
        return event

    def discard_event(self, event: Any) -> None:
        """Return the start event of a bracket that could not be closed."""
        self._pool.release(event)

    def on_error(self, where: str, exc: BaseException) -> None:
        """Count an exception caught anywhere in the profiler (see
        ``ProfilerHealth``)."""
        self.health.record_error(where, exc)

    def push(self, segment_index: int, phase_index: int, start_event: Any, end_event: Any, cpu_ms: float) -> None:
        """Queue one bracket for ``drain``; when the queue is full drop the
        oldest pair and count it in ``dropped``, so a rank that stops draining
        cannot grow without bound.

        A fwd / bwd / log-prob fwd bracket takes the next micro-batch number of
        its segment in the step; a training forward also takes the token count
        reported while it was open.
        """
        if len(self._pending) >= self._max_pending:
            _, _, old_start, old_end, _, _, _ = self._pending.popleft()
            self._pool.release(old_start)
            self._pool.release(old_end)
            self._window.add_index(_DROPPED_INDEX, 1.0)
            if self._trace is not None and self._trace.covers(self._popped):
                self._trace.lost += 1
            self._popped += 1
        micro_batch = self._step_numbers.get(segment_index, -1)
        tokens = -1.0
        if micro_batch >= 0:
            self._step_numbers[segment_index] = micro_batch + 1
            if segment_index == _FWD_INDEX:
                tokens, self._next_tokens = self._next_tokens, -1.0
        self._pending.append((segment_index, phase_index, start_event, end_event, cpu_ms, micro_batch, tokens))
        self._pushed += 1

    def _begin_trace(self) -> None:
        """Anchor the sampled step: an event recorded on an idle device marks
        the wall-clock time that every bracket of the step is placed
        against."""
        self._trace_next = False
        self._synchronize()
        anchor = self._pool.acquire()
        anchor.record()
        self._trace = _TraceCapture(anchor=anchor, wall=self._wall(), first=self._pushed)

    def note_tokens(self, tokens: int) -> None:
        """Take the token count of the training micro-batch whose forward
        bracket is open; that bracket's ``push`` claims it."""
        self._next_tokens = float(tokens)

    def add_tokens(self, tokens: int) -> None:
        """Count the tokens this rank trained on in the current step."""
        self._window.add("tokens", float(tokens))

    def take_cpu_span(self) -> tuple[float, float] | None:
        """The wall-clock span of the step traced last, once (primary rank)."""
        span, self._cpu_span = self._cpu_span, None
        return span

    def attach_cpu_events(self, events: list[dict[str, Any]]) -> None:
        """CPU timeline events of the traced step, sent with this rank's
        sample."""
        self._cpu_events = events

    def _on_gc(self, phase: str, info: dict[str, Any]) -> None:
        if self._train_thread is not None and threading.get_ident() != self._train_thread:
            return
        if phase == "start":
            self._gc_start = self._clock() if self._step_open else None
        elif self._gc_start is not None:
            self._window.add_index(_GC_INDEX, (self._clock() - self._gc_start) * 1e3)
            self._gc_start = None

    # ---- cold path ----------------------------------------------------------------------------------------------

    @property
    def pending_events(self) -> int:
        return len(self._pending)

    @property
    def window(self) -> WindowStats:
        return self._window

    def drain(self) -> int:
        """Fold completed event pairs from the front of the queue into the
        window; stop at the first pair that is still in flight.

        Pairs are queued in stream order, so an incomplete pair implies every
        later pair is incomplete too. Each pair is popped only after it has
        been folded, so an exception leaves the queue consistent and no event
        is released twice. Returns the number of pairs folded.
        """
        started = self._clock()
        folded = 0
        pending = self._pending
        window = self._window
        trace = self._trace
        while pending:
            segment_index, phase_index, start_event, end_event, cpu_ms, micro_batch, tokens = pending[0]
            if not (end_event.query() and start_event.query()):
                break
            ms = start_event.elapsed_time(end_event)
            # Read before anything is folded, so a failing readout leaves no partial update.
            traced = trace is not None and trace.covers(self._popped)
            if traced:
                offsets = (trace.anchor.elapsed_time(start_event), trace.anchor.elapsed_time(end_event))
            window.add_index(segment_index, ms)
            if phase_index >= 0:
                window.add_index(phase_index, ms)
            if segment_index == _FWD_INDEX:
                window.add_index(_CPU_FWD_INDEX, cpu_ms)
                window.add_index(_NUM_FWD_INDEX, 1.0)
                if window.max_index(_MB_FWD_MAX_INDEX, ms):
                    window.values[_MB_FWD_ARGMAX_INDEX] = float(micro_batch)
                    window.values[_MB_FWD_ARGMAX_TOKENS_INDEX] = tokens
                window.min_index(_MB_FWD_MIN_INDEX, ms)
                if tokens >= 0:
                    window.max_index(_MB_TOKENS_MAX_INDEX, tokens)
                    window.min_index(_MB_TOKENS_MIN_INDEX, tokens)
            elif segment_index == _BWD_INDEX:
                window.add_index(_CPU_BWD_INDEX, cpu_ms)
                window.max_index(_MB_BWD_MAX_INDEX, ms)
                window.min_index(_MB_BWD_MIN_INDEX, ms)
            if traced:
                if len(trace.records) < self._max_trace_brackets * RECORD_WIDTH:
                    trace.records.extend((segment_index, phase_index, micro_batch, tokens, *offsets))
                else:
                    trace.lost += 1
            pending.popleft()
            self._pool.release(start_event)
            self._pool.release(end_event)
            self._popped += 1
            folded += 1
        self._finish_trace()
        window.add("overhead", (self._clock() - started) * 1e3)
        return folded

    def _finish_trace(self) -> None:
        """Hand the sampled step over once every pair of it has been read."""
        trace = self._trace
        if trace is None or trace.end is None or self._popped < trace.end:
            return
        self._pool.release(trace.anchor)
        self._trace = None
        self._trace_done = RankTrace(wall=trace.wall, records=trace.records, lost=trace.lost)

    @property
    def disabled(self) -> bool:
        return self._disabled

    def end_step(self, rollout_id: int = -1, final: bool = False) -> list[WindowReport]:
        """Close one training step; every ``report_interval`` steps gather and
        analyze.

        All ranks must call this at the same logical step (it contains a
        collective), including a switched-off one: it stops recording time but
        still joins every gather with a row carrying its ``health`` code. Never
        raises: profiler failures are counted by ``self.health``. Returns the
        reports that are ready on the primary rank (empty elsewhere and between
        windows). On the ``final`` step of a run it waits (bounded) for the
        pending analyses, since no later call would pick them up.
        """
        try:
            self.drain()
        except Exception as exc:
            self.on_error("drain", exc)
        self._step_open = False
        self._step_numbers = dict.fromkeys(_NUMBERED, 0)
        if self._trace is not None and self._trace.end is None:
            self._trace.end = self._pushed
            if self.is_primary:
                self._cpu_span = (self._trace.wall, self._wall())
            self._finish_trace()
        self._window.add("num_steps", 1.0)
        if self._window_first_rollout is None:
            self._window_first_rollout = rollout_id
        self._window_last_rollout = rollout_id
        if self._window.get("num_steps") >= self.report_interval:
            self._close_window()
        if final and not self.flush(timeout=_FINAL_FLUSH_TIMEOUT_S):
            self.on_error("analyze", TimeoutError(f"analysis still running {_FINAL_FLUSH_TIMEOUT_S} s after the run"))
        return self._take_reports()

    def _close_window(self) -> None:
        closed_wall = self._wall()
        errors, code = self.health.close_window(self._window.get("dropped"))
        code = self._health_code(code)
        self._window.set("unread", float(len(self._pending)))
        self._window.set("errors", errors)
        self._window.set("health", code)
        try:
            started = self._clock()
            table = self._gather_window(code)
            if table is None:
                return
            gathered = self._clock()
            if self._handle_switch_off(table, closed_wall):
                return
            traces = self._gather_traces() if self._trace_enabled and not self._disabled else None
            # A switched-off primary has already stopped its worker, so there
            # is nothing to analyse.
            if self.is_primary and not self._disabled:
                self._submit(
                    self._job(table, closed_wall=closed_wall, gather_ms=(gathered - started) * 1e3, traces=traces)
                )
        finally:
            self._reset_window()

    def _gather_traces(self) -> list[RankTrace | None] | None:
        """Hand this window's sample to the primary rank: every rank joins the
        gather in the same window, with ``None`` when it has no complete
        sample.

        Returns the gathered list on the primary rank, ``None`` elsewhere or on
        failure.
        """
        if self._trace is not None:
            # Still in flight at the close (only with very short windows): dropped.
            self._pool.release(self._trace.anchor)
            self._trace = None
        sample, self._trace_done = self._trace_done, None
        if sample is not None:
            sample.cpu_events = self._cpu_events
        self._cpu_events = []
        if self._primary_rank is None:
            return None
        try:
            return self._gather_to(sample, self._primary_rank)
        except Exception as exc:
            self.on_error("trace", exc)
            return None

    def _health_code(self, code: float) -> float:
        """Return this window's ``health`` code, logging a new local switch-off
        request once.

        A rank that is already off keeps asking in every window, so a peer that
        has not seen the request yet still does.
        """
        if self._disabled:
            return HEALTH_DISABLE
        if code >= HEALTH_DISABLE:
            logger.warning(
                "[straggler] rank %d asks to switch the profiler off: %s",
                self.rank_meta.rank,
                self.health.request_reason(),
            )
        return code

    def _gather_window(self, code: float) -> list[list[float]] | None:
        """Gather this window's row on every rank, or ``None`` if it failed.

        Every rank issues the same collectives whatever failed locally, or the
        gathers of different ranks would pair up wrongly: the metadata gather
        runs once, in the first window, and every window gathers one row. A
        rank left without the metadata cannot retry it (that object gather
        would be its alone), so its row asks every rank to switch off at this
        window. A failed row gather is counted and the next window tries again;
        the dedicated short-timeout group bounds a group that is really broken,
        and once health gives up the rank also stops recording, so a failed
        group cannot keep costing timer time for the rest of the run.
        """
        if not self._meta_gathered:
            self._meta_gathered = True
            try:
                self._all_meta = list(self._gather_objects(self.rank_meta))
                self._primary_rank = next((meta.rank for meta in self._all_meta if meta.primary), None)
            except Exception as exc:
                self.on_error("gather", exc)
                logger.warning(
                    "[straggler] rank %d asks to switch the profiler off: the rank metadata gather failed (%s: %s)",
                    self.rank_meta.rank,
                    type(exc).__name__,
                    exc,
                )
        if self._all_meta is None:
            code = HEALTH_DISABLE
            self._window.set("health", code)
        try:
            return self._gather(disabled_row() if self._disabled else self._window.as_list())
        except Exception as exc:
            self.on_error("gather", exc)
            if code >= HEALTH_DISABLE and not self._disabled:
                self._disable(f"the window gather kept failing ({type(exc).__name__}: {exc})")
            return None

    def _handle_switch_off(self, table: list[list[float]], closed_wall: float) -> bool:
        """Switch off when a gathered row asks for it; ``True`` when it did.

        On the primary rank the notice joins the analysis queue, so it stays in
        window order with the reports around it.
        """
        # Rows are in gather-group rank order, which is the global rank.
        requesting = [i for i, row in enumerate(table) if row[_HEALTH_INDEX] >= HEALTH_DISABLE]
        if not requesting or self._disabled:
            return False
        note = (
            f"rank {requesting[0]} asked to switch the profiler off"
            + (f" (with {len(requesting) - 1} more)" if len(requesting) > 1 else "")
            + "; the reason is in that rank's log"
        )
        if self.is_primary:
            notice = WindowReport(
                metrics={
                    "straggler/health/state": float(HEALTH_DISABLE),
                    "straggler/health/requested_by": float(requesting[0]),
                    "straggler/health/errors": float(sum(row[_ERRORS_INDEX] for row in table)),
                },
                alerts=[],
                rows=[],
                note=note,
            )
            self._submit(self._job(None, closed_wall=closed_wall, ready=notice))
        self._disable(note)
        return True

    def _reset_window(self) -> None:
        """Start the next window; every close path has to reach this, or the
        window would close (and gather) again on every following step."""
        self._window.reset()
        self.window_start_wall = self._wall()
        self._window_first_rollout = None
        self._trace_next = self._trace_enabled and not self._disabled and self._primary_rank is not None

    def _job(self, table: list[list[float]] | None, **kwargs: Any) -> _WindowJob:
        first = self._window_first_rollout
        return _WindowJob(
            table=table,
            meta=list(self._all_meta or []),
            window_start_wall=self.window_start_wall,
            first_rollout=self._window_last_rollout if first is None else first,
            last_rollout=self._window_last_rollout,
            **kwargs,
        )

    def _submit(self, job: _WindowJob) -> None:
        if self._worker is None:
            self._handle_job(job)
            return
        # A ready notice (job.ready is not None) bypasses the backlog check so
        # the health/state=2 report reaches end_step even when the queue is
        # full. There is at most one such notice per profiler lifetime, so the
        # queue stays bounded at max_backlog + 1.
        if job.ready is None and self._worker.backlog >= self._max_backlog:
            self.on_error(
                "analyze",
                RuntimeError(
                    f"{self._worker.backlog} windows are still waiting for analysis; the window ending at rollout "
                    f"{job.last_rollout} is dropped"
                ),
            )
            return
        self._worker.submit(job)

    def _handle_job(self, job: _WindowJob) -> None:
        """Analyze one window (worker thread, or inline without a worker) and
        queue its report for ``end_step``; never raises.

        Only this method touches ``detector_state``, and jobs run one at a time
        in window order.
        """
        report = job.ready
        if report is None:
            started = self._clock()
            try:
                report = analyze_window(job.table, job.meta, self.detector_config, self.detector_state)
            except Exception as exc:
                self.on_error("analyze", exc)
                return
            report.metrics["straggler/gather_ms"] = job.gather_ms
            report.metrics["straggler/analyze_ms"] = (self._clock() - started) * 1e3
            report.window_start_wall = job.window_start_wall
            if job.traces is not None and self._trace_sink is not None:
                try:
                    report.metrics.update(self._trace_sink(job.traces, job.meta, job.first_rollout))
                except Exception as exc:
                    self.on_error("trace", exc)
        report.first_rollout = job.first_rollout
        report.last_rollout = job.last_rollout
        report.closed_wall = job.closed_wall
        report.analyzed_wall = self._wall()
        if self._on_report is not None:
            try:
                self._on_report(report)
            except Exception as exc:
                self.on_error("report", exc)
        self._reports.append(report)

    def _take_reports(self) -> list[WindowReport]:
        reports: list[WindowReport] = []
        while self._reports:
            reports.append(self._reports.popleft())
        return reports

    def flush(self, timeout: float | None = None) -> bool:
        """Wait for queued analyses; True when none is left."""
        return self._worker.flush(timeout) if self._worker is not None else True

    def _disable(self, reason: str) -> None:
        """Switch the profiler off on this rank for the rest of the run."""
        if self._disabled:
            return
        self._disabled = True
        self.health.disable(reason)
        while self._pending:
            _, _, start_event, end_event, _, _, _ = self._pending.popleft()
            self._pool.release(start_event)
            self._pool.release(end_event)
        if self._trace is not None:
            self._pool.release(self._trace.anchor)
            self._trace = None
        self.close()
        logger.warning(
            "[straggler] profiler disabled on rank %d: %s. Training continues; config.timers now records nothing.",
            self.rank_meta.rank,
            reason,
        )

    def close(self) -> None:
        """Unregister the GC callback and stop the worker once its queue is
        empty; safe to call more than once."""
        if self._on_gc in gc.callbacks:
            gc.callbacks.remove(self._on_gc)
        if self._worker is not None:
            self._worker.stop()
