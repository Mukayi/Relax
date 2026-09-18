# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for how a window report reaches metrics, timeline and logs."""

from argparse import Namespace

from relax.utils.straggler import reporter
from relax.utils.straggler.detector import DetectorConfig, DetectorState, RankMeta, analyze_window
from relax.utils.straggler.stats import FIELD_INDEX, NUM_FIELDS
from relax.utils.timer import Timer


def _row(**fields):
    values = [0.0] * NUM_FIELDS
    values[FIELD_INDEX["num_steps"]] = 1.0
    for name, value in fields.items():
        values[FIELD_INDEX[name]] = value
    return values


def _report(flag_rank=None):
    table = [_row(fwd=100.0, bwd=200.0, tokens=1000.0) for _ in range(4)]
    if flag_rank is not None:
        table[flag_rank] = _row(fwd=150.0, bwd=300.0, tokens=1000.0)
    meta = [RankMeta(rank=r, dp=r, tp=0, pp=0, host="h0", device=r) for r in range(4)]
    report = analyze_window(table, meta, DetectorConfig(persist_windows=1), DetectorState())
    report.window_start_wall = 1000.0
    report.window_end_wall = 1010.0
    return report


def test_report_logs_metrics_with_rollout_step_key(monkeypatch):
    logged = []
    monkeypatch.setattr(
        reporter.tracking_utils, "log", lambda args, metrics, step_key: logged.append((metrics, step_key))
    )
    args = Namespace(wandb_always_use_train_step=False, timeline_dump_dir=None)
    reporter.report_straggler_window(args, rollout_id=7, report=_report())
    assert len(logged) == 1
    metrics, step_key = logged[0]
    assert step_key == "rollout/step" and metrics["rollout/step"] == 7
    assert metrics["straggler/flagged/count"] == 0
    assert metrics["straggler/fwd/median_ms"] == 100.0


def test_timeline_events_one_row_per_rank_when_enabled(monkeypatch):
    monkeypatch.setattr(reporter.tracking_utils, "log", lambda *a, **k: True)
    timer = Timer()
    timer.records.clear()
    args = Namespace(wandb_always_use_train_step=False, timeline_dump_dir="/tmp/tl")
    reporter.report_straggler_window(args, rollout_id=3, report=_report())
    events = list(timer.records)
    timer.records.clear()
    # 4 ranks x 2 non-zero segments (fwd, bwd)
    assert len(events) == 8
    pids = {event.pid for event in events}
    assert pids == {reporter.TIMELINE_PID_BASE + r for r in range(4)}
    rank0 = sorted((event for event in events if event.pid == reporter.TIMELINE_PID_BASE), key=lambda e: e.start_ts)
    assert rank0[0].name.startswith("straggler/fwd") and rank0[0].start_ts == 1000.0
    assert rank0[0].end_ts == 1000.1 and rank0[1].start_ts == 1000.1
    assert all(event.step == 3 for event in events)


def test_no_timeline_events_when_disabled(monkeypatch):
    monkeypatch.setattr(reporter.tracking_utils, "log", lambda *a, **k: True)
    timer = Timer()
    timer.records.clear()
    args = Namespace(wandb_always_use_train_step=False, timeline_dump_dir=None)
    reporter.report_straggler_window(args, rollout_id=3, report=_report())
    assert timer.records == []


def test_alert_is_logged_as_warning_with_table(monkeypatch):
    monkeypatch.setattr(reporter.tracking_utils, "log", lambda *a, **k: True)
    warnings, infos = [], []
    monkeypatch.setattr(reporter.logger, "warning", lambda msg, *a: warnings.append(msg % a))
    monkeypatch.setattr(reporter.logger, "info", lambda msg, *a: infos.append(msg % a))
    args = Namespace(wandb_always_use_train_step=False, timeline_dump_dir=None)
    reporter.report_straggler_window(args, rollout_id=5, report=_report(flag_rank=2))
    assert len(warnings) == 1 and "rank 2" in warnings[0] and "slow_device" in warnings[0]
    assert len(infos) == 1 and "per-rank window" in infos[0] and "slow_device" in infos[0]
