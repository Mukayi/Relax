# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for the ``config.timers`` replacement, the Megatron timer names it
is given and the event pool."""

import pytest

from relax.utils.straggler.core import timers as timers_module
from relax.utils.straggler.core.stats import DP_REDUCTION_PHASE, FIELD_INDEX, PP_PHASE_FIELDS
from relax.utils.straggler.core.timers import EventPool, StragglerTimers, _NullTimer, _SegmentTimer
from relax.utils.straggler.megatron_adapter import (
    FORWARD_ONLY_TIMER_SEGMENTS,
    GRAD_SYNC_TIMERS,
    IGNORED_TIMER_NAMES,
    MEGATRON_TIMER_SEGMENTS,
    PP_PHASE_TIMERS,
    TIMER_PHASES,
    phase_timers,
)
from tests.utils.straggler_helpers import FakeEvent


class _FakeSink:
    def __init__(self, capturing=False):
        self.pool = EventPool(4, FakeEvent)
        self.pushed = []
        self.discarded = []
        self.errors = []
        self.capturing = capturing
        self.fail = False

    def record_event(self):
        if self.fail:
            raise RuntimeError("cudaErrorLaunchFailure")
        if self.capturing:
            return None
        event = self.pool.acquire()
        event.record()
        return event

    def on_error(self, where, exc):
        self.errors.append((where, exc))

    def discard_event(self, event):
        self.discarded.append(event)
        self.pool.release(event)

    def push(self, segment_index, phase_index, start_event, end_event, cpu_ms):
        self.pushed.append((segment_index, phase_index, start_event, end_event, cpu_ms))


def test_event_pool_reuses_released_events_and_grows_when_empty():
    pool = EventPool(2, FakeEvent)
    first, second = pool.acquire(), pool.acquire()
    assert len(pool) == 0 and pool.created == 2
    third = pool.acquire()
    assert pool.created == 3
    pool.release(first)
    assert pool.acquire() is first
    assert third is not second


def test_straggler_timers_maps_known_names_and_ignores_unknown():
    sink = _FakeSink()
    timers = StragglerTimers(sink, MEGATRON_TIMER_SEGMENTS)
    assert isinstance(timers("forward-compute", log_level=2), _SegmentTimer)
    assert isinstance(timers("forward-backward", log_level=1), _NullTimer)
    assert timers("forward-compute") is timers("forward-compute", log_level=2)


def test_segment_timer_start_stop_pushes_pair_with_segment_index():
    sink = _FakeSink()
    timers = StragglerTimers(sink, MEGATRON_TIMER_SEGMENTS)
    timers("backward-compute").start(barrier=True)
    timers("backward-compute").stop(barrier=True)
    assert len(sink.pushed) == 1
    segment_index, phase_index, start, end, cpu_ms = sink.pushed[0]
    assert segment_index == FIELD_INDEX["bwd"]
    assert phase_index == -1
    assert start.elapsed_time(end) == 1.0
    assert cpu_ms >= 0.0


def test_training_pp_recv_names_carry_their_1f1b_phase_and_gradient_syncs_their_timer():
    sink = _FakeSink()
    timers = StragglerTimers(sink, MEGATRON_TIMER_SEGMENTS, TIMER_PHASES)
    names = [*PP_PHASE_TIMERS, "forward-send-forward-recv", *GRAD_SYNC_TIMERS]
    for name in names:
        timers(name).start()
        timers(name).stop()
    pushed = [(segment_index, phase_index) for segment_index, phase_index, *_ in sink.pushed]
    assert pushed == [
        *[(FIELD_INDEX["pp_recv"], FIELD_INDEX[PP_PHASE_TIMERS[name]]) for name in PP_PHASE_TIMERS],
        (FIELD_INDEX["pp_recv"], -1),
        *[(FIELD_INDEX["dp_grad_sync"], position) for position in range(len(GRAD_SYNC_TIMERS))],
    ]


def test_phase_timers_are_training_pp_recv_names_covering_every_phase_once():
    assert {MEGATRON_TIMER_SEGMENTS[name] for name in PP_PHASE_TIMERS} == {"pp_recv"}
    assert sorted(PP_PHASE_TIMERS.values()) == sorted(PP_PHASE_FIELDS)


def test_segment_timer_tolerates_double_start_and_stop_without_start():
    sink = _FakeSink()
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


def test_segment_timer_skips_bracket_when_sink_cannot_record():
    # Sink returns None (stream under CUDA-graph capture): nothing is pushed and
    # no event leaks, whether capture starts before ``start`` or between the two.
    sink = _FakeSink(capturing=True)
    timer = StragglerTimers(sink, MEGATRON_TIMER_SEGMENTS)("forward-compute")
    timer.start()
    assert not timer.active
    timer.stop()
    assert sink.pushed == [] and sink.discarded == []

    sink.capturing = False
    timer.start()
    sink.capturing = True
    timer.stop()
    assert sink.pushed == []
    assert len(sink.discarded) == 1 and len(sink.pool) == 4
    assert not timer.active


def test_sink_errors_skip_the_bracket_and_are_reported_instead_of_raised():
    sink = _FakeSink()
    timer = StragglerTimers(sink, MEGATRON_TIMER_SEGMENTS)("forward-compute")
    sink.fail = True
    timer.start()  # must not raise into Megatron's schedule
    assert not timer.active and [where for where, _ in sink.errors] == ["timer"]

    sink.fail = False
    timer.start()
    sink.fail = True
    timer.stop()
    assert sink.pushed == [] and len(sink.errors) == 2 and not timer.active
    assert len(sink.pool) == 4, "the start event of a bracket that failed to close goes back to the pool"


def test_all_megatron_optimizer_and_comm_names_map_to_buckets():
    assert {MEGATRON_TIMER_SEGMENTS[name] for name in MEGATRON_TIMER_SEGMENTS if name.startswith("optimizer-")} == {
        "optim"
    }
    assert MEGATRON_TIMER_SEGMENTS["params-all-gather"] == "dp_param_gather"
    assert MEGATRON_TIMER_SEGMENTS["all-grads-sync"] == "dp_grad_sync"
    assert GRAD_SYNC_TIMERS[DP_REDUCTION_PHASE] == "all-grads-sync", "the trace aligns DP peers on this one"
    assert MEGATRON_TIMER_SEGMENTS["forward-send-backward-recv"] == "pp_recv"


def test_forward_only_names_land_in_lp_buckets():
    assert FORWARD_ONLY_TIMER_SEGMENTS["forward-compute"] == "lp_fwd"
    assert FORWARD_ONLY_TIMER_SEGMENTS["forward-recv"] == "lp_pp_recv"
    assert "backward-compute" not in FORWARD_ONLY_TIMER_SEGMENTS
    assert "optimizer-inner-step" not in FORWARD_ONLY_TIMER_SEGMENTS


def test_timers_log_and_write_are_explicitly_unsupported():
    timers = StragglerTimers(_FakeSink(), MEGATRON_TIMER_SEGMENTS)
    with pytest.raises(NotImplementedError):
        timers.log(["forward-compute"])
    with pytest.raises(NotImplementedError):
        timers.write(["forward-compute"], None, 0)


@pytest.fixture()
def captured_warnings(monkeypatch):
    warnings = []
    monkeypatch.setattr(timers_module.logger, "warning", lambda msg, *a, **k: warnings.append(msg % a))
    return warnings


def test_known_timer_name_produces_no_warning(captured_warnings):
    timers = StragglerTimers(_FakeSink(), MEGATRON_TIMER_SEGMENTS)
    timers("forward-compute")
    timers("forward-compute")
    assert captured_warnings == []


@pytest.mark.parametrize("phase", ["train", "forward_only"])
def test_ignored_timer_name_produces_no_warning(captured_warnings, phase):
    timers = phase_timers(_FakeSink())[phase]
    for name in IGNORED_TIMER_NAMES:
        assert isinstance(timers(name), _NullTimer)
        timers(name)
    assert captured_warnings == []


def test_unknown_timer_name_warns_exactly_once(captured_warnings):
    timers = StragglerTimers(_FakeSink(), MEGATRON_TIMER_SEGMENTS)
    timers("totally-new-timer")
    timers("totally-new-timer")
    timers("totally-new-timer")
    assert len(captured_warnings) == 1
    assert "totally-new-timer" in captured_warnings[0]
