# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for the ``config.timers`` replacement and the event pool."""

import pytest

from relax.utils.straggler.stats import FIELD_INDEX, FORWARD_ONLY_TIMER_SEGMENTS, MEGATRON_TIMER_SEGMENTS
from relax.utils.straggler.timers import EventPool, StragglerTimers, _NullTimer, _SegmentTimer


class FakeEvent:
    """Stands in for ``torch.cuda.Event``; ``record`` stamps a global
    counter."""

    clock = 0.0

    def __init__(self):
        self.stamp = None
        self.done = True

    def record(self):
        FakeEvent.clock += 1.0
        self.stamp = FakeEvent.clock

    def query(self):
        return self.done

    def elapsed_time(self, other):
        return other.stamp - self.stamp


class RecordingSink:
    def __init__(self):
        self.pool = EventPool(4, FakeEvent)
        self.pushed = []

    def record_event(self):
        event = self.pool.acquire()
        event.record()
        return event

    def push(self, segment_index, start_event, end_event, cpu_ms):
        self.pushed.append((segment_index, start_event, end_event, cpu_ms))


def test_event_pool_reuses_and_grows():
    pool = EventPool(2, FakeEvent)
    first, second = pool.acquire(), pool.acquire()
    assert len(pool) == 0 and pool.created == 2
    third = pool.acquire()
    assert pool.created == 3
    pool.release(first)
    assert pool.acquire() is first
    assert third is not second


def test_straggler_timers_maps_known_names_and_ignores_unknown():
    sink = RecordingSink()
    timers = StragglerTimers(sink, MEGATRON_TIMER_SEGMENTS)
    assert isinstance(timers("forward-compute", log_level=2), _SegmentTimer)
    assert isinstance(timers("forward-backward", log_level=1), _NullTimer)
    assert timers("forward-compute") is timers("forward-compute", log_level=2)


def test_segment_timer_start_stop_pushes_pair_with_segment_index():
    sink = RecordingSink()
    timers = StragglerTimers(sink, MEGATRON_TIMER_SEGMENTS)
    timers("backward-compute").start(barrier=True)
    timers("backward-compute").stop(barrier=True)
    assert len(sink.pushed) == 1
    segment_index, start, end, cpu_ms = sink.pushed[0]
    assert segment_index == FIELD_INDEX["bwd"]
    assert start.elapsed_time(end) == 1.0
    assert cpu_ms >= 0.0


def test_segment_timer_tolerates_double_start_and_stop_without_start():
    sink = RecordingSink()
    timers = StragglerTimers(sink, MEGATRON_TIMER_SEGMENTS)
    timer = timers("optimizer-inner-step")
    timer.stop()
    assert sink.pushed == []
    timer.start()
    timer.start()
    assert timer.active
    timer.stop()
    assert len(sink.pushed) == 1
    assert not timer.active


def test_all_megatron_optimizer_and_comm_names_map_to_buckets():
    assert {MEGATRON_TIMER_SEGMENTS[name] for name in MEGATRON_TIMER_SEGMENTS if name.startswith("optimizer-")} == {
        "optim"
    }
    assert MEGATRON_TIMER_SEGMENTS["params-all-gather"] == "dp_param_gather"
    assert MEGATRON_TIMER_SEGMENTS["all-grads-sync"] == "dp_grad_sync"
    assert MEGATRON_TIMER_SEGMENTS["forward-send-backward-recv"] == "pp_recv"


def test_forward_only_names_land_in_lp_buckets():
    assert FORWARD_ONLY_TIMER_SEGMENTS["forward-compute"] == "lp_fwd"
    assert FORWARD_ONLY_TIMER_SEGMENTS["forward-recv"] == "lp_pp_recv"
    assert "backward-compute" not in FORWARD_ONLY_TIMER_SEGMENTS
    assert "optimizer-inner-step" not in FORWARD_ONLY_TIMER_SEGMENTS


def test_timers_log_and_write_are_explicitly_unsupported():
    timers = StragglerTimers(RecordingSink(), MEGATRON_TIMER_SEGMENTS)
    with pytest.raises(NotImplementedError):
        timers.log(["forward-compute"])
    with pytest.raises(NotImplementedError):
        timers.write(["forward-compute"], None, 0)
