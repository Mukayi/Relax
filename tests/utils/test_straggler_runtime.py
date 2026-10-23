# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for the per-step reporting the Megatron actor drives, and for when
the profiler gets installed."""

from argparse import Namespace

import pytest

from relax.utils.straggler import runtime
from relax.utils.straggler.detector import WindowReport
from relax.utils.straggler.reporter import REPORTED_METRICS
from relax.utils.straggler.runtime import StragglerProfiler, install_straggler_profiler
from relax.utils.timer import TimelineEvent, Timer
from relax.utils.training import train_metric_utils
from tests.utils.straggler_helpers import FakeEvent, bracket, make_collector, meta


def _args(num_rollout=100):
    return Namespace(num_rollout=num_rollout, wandb_always_use_train_step=False, timeline_dump_dir=None)


def _profiler(interval=2, num_rollout=100, **kwargs):
    return StragglerProfiler(make_collector(interval=interval, **kwargs), _args(num_rollout))


def _step(profiler, rollout_id):
    bracket(profiler.collector)
    return profiler.step_metrics(rollout_id, tokens=1000)


def test_scalars_arrive_when_a_window_closes_and_carry_its_rollout_range():
    profiler = _profiler(interval=2)
    assert _step(profiler, 0) is None
    metrics = _step(profiler, 1)
    assert metrics["straggler/window/first_rollout"] == 0.0 and metrics["straggler/window/last_rollout"] == 1.0
    assert set(metrics) <= REPORTED_METRICS


def test_a_failing_reporter_is_counted_and_never_raises(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("timeline broke")

    monkeypatch.setattr(runtime, "report_straggler_window", boom)
    profiler = _profiler(interval=1)
    assert _step(profiler, 0) is None
    assert profiler.collector.health.errors_total == 1


def test_last_rollout_of_the_run_waits_for_its_own_window():
    profiler = _profiler(interval=1, num_rollout=5, background=True)
    try:
        assert _step(profiler, 4)["straggler/window/last_rollout"] == 4.0
    finally:
        profiler.collector.close()


def test_install_is_a_no_op_when_disabled_or_for_non_training_roles(monkeypatch):
    monkeypatch.setenv("RELAX_STRAGGLER_PROFILER", "0")
    assert install_straggler_profiler("actor", _args()) is None
    monkeypatch.setenv("RELAX_STRAGGLER_PROFILER", "1")
    # These roles never report a training step, so they must not get a profiler
    # (checked before any Megatron parallel state is touched).
    for role in ("critic", "reference", "actor_fwd"):
        assert install_straggler_profiler(role, _args()) is None


class _FakeBackend:
    """Stands in for ``relax.utils.device.device_module`` on a non-CUDA
    accelerator."""

    def __init__(self, capturing=None):
        self.events = []
        if capturing is not None:
            self.is_current_stream_capturing = lambda: capturing

    def Event(self, **kwargs):
        self.events.append(kwargs)
        return object()


def _use_backend(monkeypatch, backend, available=True):
    from relax.utils import device

    monkeypatch.setattr(device, "device_module", backend)
    monkeypatch.setattr(device, "is_available", lambda: available)


def test_timing_events_come_from_the_active_accelerator(monkeypatch):
    backend = _FakeBackend()
    _use_backend(monkeypatch, backend)
    runtime.timing_event_factory()()
    assert backend.events == [{"enable_timing": True}]


@pytest.mark.parametrize(
    ("capturing", "available", "expected"), [(True, True, True), (False, True, False), (True, False, False)]
)
def test_capture_check_uses_the_backend_function_when_there_is_one(monkeypatch, capturing, available, expected):
    _use_backend(monkeypatch, _FakeBackend(capturing=capturing), available=available)
    assert runtime.stream_capture_check()() is expected


def test_capture_check_is_false_when_the_backend_has_none(monkeypatch):
    _use_backend(monkeypatch, _FakeBackend())
    assert runtime.stream_capture_check()() is False


def test_timers_are_none_without_a_profiler():
    assert runtime.straggler_timers("train") is None
    with pytest.raises(ValueError):
        runtime._timers_for(_profiler().collector, "eval")


class _DeviceLengths:
    def __iter__(self):
        raise AssertionError("summing device lengths would synchronize")


def test_micro_batch_tokens_reach_the_profiler_only_as_host_lists(monkeypatch):
    runtime.note_microbatch_tokens([1, 2])  # no profiler: nothing happens
    profiler = _profiler()
    noted = []
    monkeypatch.setattr(runtime, "_PROFILER", profiler)
    monkeypatch.setattr(profiler.collector, "note_tokens", noted.append)
    for lengths in ([100, 250], (7,), None, _DeviceLengths()):
        runtime.note_microbatch_tokens(lengths)
    assert noted == [350, 7]


@pytest.mark.parametrize(
    ("env", "trace_on", "expected"),
    [(None, True, 2), (None, False, 1), ("0", True, 0), ("1", True, 1), ("3", True, 3)],
)
def test_the_start_level_comes_from_the_env_or_from_whether_traces_can_be_written(
    monkeypatch, env, trace_on, expected
):
    if env is None:
        monkeypatch.delenv("RELAX_STRAGGLER_LEVEL", raising=False)
    else:
        monkeypatch.setenv("RELAX_STRAGGLER_LEVEL", env)
    assert runtime.initial_level(trace_on) == expected


def test_the_model_reaches_the_collector_only_with_a_profiler(monkeypatch):
    monkeypatch.setattr(runtime, "_PROFILER", None)
    runtime.register_straggler_model([])
    profiler = _profiler()
    monkeypatch.setattr(runtime, "_PROFILER", profiler)
    runtime.register_straggler_model([])
    assert not profiler.collector.module_timers.counts()
    runtime.register_straggler_model([object()])
    assert profiler.collector.health.errors_total == 1, "a model without modules() is a profiler error"


def test_traces_go_to_the_env_dir_or_under_the_timeline_dir(monkeypatch):
    monkeypatch.delenv("RELAX_STRAGGLER_TRACE_DIR", raising=False)
    assert runtime.trace_dir(Namespace(timeline_dump_dir=None)) == ""
    assert runtime.trace_dir(Namespace(timeline_dump_dir="/runs/x/timeline")) == "/runs/x/timeline/straggler"
    monkeypatch.setenv("RELAX_STRAGGLER_TRACE_DIR", "/traces")
    assert runtime.trace_dir(Namespace(timeline_dump_dir="/runs/x/timeline")) == "/traces"


@pytest.fixture
def timer_records():
    Timer().records.clear()
    yield Timer().records
    Timer().records.clear()


def test_cpu_events_are_the_finished_timers_overlapping_the_step_without_call_stacks(timer_records):
    timer_records.extend(
        [
            TimelineEvent("train_wait", 1.0, 2.0, pid=1, tid=1),
            TimelineEvent("get_data_batch_a", 9.0, 11.0, pid=1, tid=1, call_stack="  File ..."),
            TimelineEvent("straggler/fwd rank0_dp0_tp0_pp0", 10.0, 12.0, pid=1, tid=1),
            TimelineEvent("next_step", 21.0, 22.0, pid=1, tid=1),
        ]
    )
    events = runtime._cpu_events(10.0, 20.0)
    assert [event["name"] for event in events] == ["get_data_batch_a"]
    assert events[0]["args"] == {}


def test_only_the_latest_of_several_ready_windows_reaches_the_metrics(monkeypatch):
    def window(n):
        return WindowReport(
            metrics={"straggler/window/last_rollout": float(n)}, alerts=[], rows=[], first_rollout=n, last_rollout=n
        )

    reported, warnings = [], []
    monkeypatch.setattr(
        runtime, "report_straggler_window", lambda args, report: reported.append(report) or dict(report.metrics)
    )
    monkeypatch.setattr(runtime.logger, "warning", lambda msg, *a: warnings.append(msg % a))
    profiler = _profiler(interval=1)
    monkeypatch.setattr(profiler.collector, "end_step", lambda rollout_id, final=False: [window(0), window(1)])

    metrics = profiler.step_metrics(1, tokens=0)
    assert metrics == {"straggler/window/last_rollout": 1.0}
    assert [r.last_rollout for r in reported] == [0, 1], "every window still reaches the timeline"
    assert len(warnings) == 1 and "2 windows ready at once" in warnings[0]
    assert profiler.collector.health.errors_total == 1


def _ready_windows(monkeypatch, profiler, rollouts, failing):
    """Make ``step_metrics`` see the windows of ``rollouts`` ready at once,
    with the reporter failing on ``failing``; returns the WARNING lines."""

    def report(args, window):
        if window.last_rollout == failing:
            raise RuntimeError("timeline broke")
        return dict(window.metrics)

    def window(n):
        return WindowReport(
            metrics={"straggler/window/last_rollout": float(n)}, alerts=[], rows=[], first_rollout=n, last_rollout=n
        )

    warnings = []
    monkeypatch.setattr(runtime, "report_straggler_window", report)
    monkeypatch.setattr(runtime.logger, "warning", lambda msg, *a: warnings.append(msg % a))
    monkeypatch.setattr(profiler.collector, "end_step", lambda rollout_id, final=False: [window(n) for n in rollouts])
    return warnings


def test_a_failing_latest_window_falls_back_to_the_previous_one_and_counts_one_error(monkeypatch):
    profiler = _profiler(interval=1)
    warnings = _ready_windows(monkeypatch, profiler, rollouts=[1, 2], failing=2)
    for errors in (1, 2):
        assert profiler.step_metrics(2, tokens=0) == {"straggler/window/last_rollout": 1.0}
        assert profiler.collector.health.errors_total == errors, "only the failing window is lost"
    assert warnings == [], "the returned window superseded nothing"


def test_the_superseded_warning_names_the_window_that_is_returned(monkeypatch):
    profiler = _profiler(interval=1)
    warnings = _ready_windows(monkeypatch, profiler, rollouts=[0, 1, 2], failing=2)
    assert profiler.step_metrics(2, tokens=0) == {"straggler/window/last_rollout": 1.0}
    assert len(warnings) == 1 and "only rollouts 1-1 reach the metrics" in warnings[0]
    assert profiler.collector.health.errors_total == 2, "window 0 superseded, window 2 failed"


@pytest.mark.parametrize(("primary", "vpp"), [(True, None), (False, 2)])
def test_install_uses_the_shared_metrics_primary_rank_and_names_brackets_by_schedule_only_under_vpp(
    monkeypatch, primary, vpp
):
    monkeypatch.setenv("RELAX_STRAGGLER_PROFILER", "1")
    monkeypatch.setattr(train_metric_utils, "is_metrics_primary_rank", lambda: primary)
    monkeypatch.setattr(runtime, "local_rank_meta", lambda: meta(1)[0])
    monkeypatch.setattr(runtime, "timing_event_factory", lambda: FakeEvent)
    monkeypatch.setattr(runtime, "stream_capture_check", lambda: runtime._never)
    # The gather group is a collective; its creation is covered separately.
    monkeypatch.setattr(runtime, "init_straggler_gather_group", lambda timeout_s: None)
    monkeypatch.setattr(runtime, "_PROFILER", None)
    args = _args()
    args.virtual_pipeline_model_parallel_size = vpp
    profiler = install_straggler_profiler("actor", args)
    try:
        assert profiler.collector.is_primary is primary
        assert profiler.collector._cpu_events_source is runtime._cpu_events
        assert profiler.collector._interleaved is (vpp is not None)
    finally:
        profiler.collector.close()


# Megatron's schedule table for 4 micro-batches over 2 model chunks, in groups of 2.
_TABLE_4 = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (3, 0), (2, 1), (3, 1)]


@pytest.mark.parametrize(
    ("forward_only", "combined", "expected"), [(False, False, _TABLE_4), (True, True, _TABLE_4), (False, True, [])]
)
def test_the_schedule_table_comes_from_the_model_config(monkeypatch, forward_only, combined, expected):
    profiler = _profiler(interleaved=True)
    monkeypatch.setattr(runtime, "_PROFILER", profiler)
    # The launch arguments may ask for groups of 3; Megatron's schedule reads the wrapped model's config.
    config = Namespace(microbatch_group_size_per_vp_stage=2, overlap_moe_expert_parallel_comm=combined)
    chunk = Namespace(module=Namespace(config=config))
    runtime.note_schedule([chunk, chunk], 4, forward_only)
    _, table, _ = profiler.collector._schedule
    assert list(table) == expected
