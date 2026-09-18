# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Non-synchronizing replacement for Megatron's ``config.timers``.

Megatron's ``megatron.core.timers.Timer.start/stop`` call
``torch.cuda.synchronize()`` on every bracket, which is why Relax sets
``config.timers = None`` everywhere. ``StragglerTimers`` speaks the same
protocol (``timers(name, log_level).start(barrier) / .stop(barrier)``) but only
records a pair of CUDA timing events plus a CPU timestamp and hands them to a
sink. Nothing here blocks on the device, so DP grad-reduce overlap and PP
communication overlap are untouched.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Mapping, Protocol

from relax.utils.straggler.stats import FIELD_INDEX


class EventSink(Protocol):
    """What a timer needs from its owner."""

    def record_event(self) -> Any:
        """Return a timing event already recorded on the current stream."""

    def push(self, segment_index: int, start_event: Any, end_event: Any, cpu_ms: float) -> None:
        """Hand over one completed bracket for later, lazy readout."""


def cuda_timing_event() -> Any:
    """Default event factory: a CUDA event that supports ``elapsed_time``."""
    import torch

    return torch.cuda.Event(enable_timing=True)


class EventPool:
    """Reuse timing-event objects so the hot path never allocates."""

    __slots__ = ("_free", "_factory", "created")

    def __init__(self, size: int, factory: Callable[[], Any] = cuda_timing_event) -> None:
        self._factory = factory
        self._free: list[Any] = [factory() for _ in range(size)]
        self.created: int = size

    def acquire(self) -> Any:
        if self._free:
            return self._free.pop()
        self.created += 1
        return self._factory()

    def release(self, event: Any) -> None:
        self._free.append(event)

    def __len__(self) -> int:
        return len(self._free)


class _NullTimer:
    """Returned for timer names we do not track; Megatron only calls
    start/stop."""

    __slots__ = ()

    def start(self, barrier: bool = False) -> None:
        return None

    def stop(self, barrier: bool = False) -> None:
        return None


class _SegmentTimer:
    """One Megatron timer name bound to one straggler segment.

    ``barrier`` is accepted for protocol compatibility and ignored: a barrier
    here would reintroduce exactly the synchronization we are avoiding. A
    second ``start`` without ``stop`` is ignored rather than asserted so the
    profiler can never take the training loop down.
    """

    __slots__ = ("_segment_index", "_sink", "_start_event", "_start_cpu")

    def __init__(self, segment_index: int, sink: EventSink) -> None:
        self._segment_index = segment_index
        self._sink = sink
        self._start_event: Any = None
        self._start_cpu: float = 0.0

    def start(self, barrier: bool = False) -> None:
        if self._start_event is not None:
            return
        self._start_event = self._sink.record_event()
        self._start_cpu = perf_counter()

    def stop(self, barrier: bool = False) -> None:
        start_event = self._start_event
        if start_event is None:
            return
        self._start_event = None
        end_event = self._sink.record_event()
        self._sink.push(self._segment_index, start_event, end_event, (perf_counter() - self._start_cpu) * 1e3)

    @property
    def active(self) -> bool:
        return self._start_event is not None


class StragglerTimers:
    """Drop-in object for ``ModelParallelConfig.timers`` /
    ``OptimizerConfig.timers``.

    Only the subset of the ``megatron.core.timers.Timers`` protocol that
    megatron.core itself uses is implemented (``__call__`` returning an object
    with ``start``/``stop``). ``log``/``write`` raise so any new caller is
    noticed immediately instead of silently doing nothing.
    """

    def __init__(self, sink: EventSink, name_to_segment: Mapping[str, str]) -> None:
        self._sink = sink
        self._name_to_segment = dict(name_to_segment)
        self._timers: dict[str, _SegmentTimer | _NullTimer] = {}
        self._null = _NullTimer()

    def __call__(self, name: str, log_level: int | None = None) -> _SegmentTimer | _NullTimer:
        timer = self._timers.get(name)
        if timer is None:
            segment = self._name_to_segment.get(name)
            timer = _SegmentTimer(FIELD_INDEX[segment], self._sink) if segment is not None else self._null
            self._timers[name] = timer
        return timer

    def log(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "StragglerTimers only records events; use the straggler metrics instead of Timers.log"
        )

    def write(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "StragglerTimers only records events; use the straggler metrics instead of Timers.write"
        )
