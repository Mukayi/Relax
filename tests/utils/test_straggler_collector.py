# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for the per-rank collector: lazy readout, windows, gather, GC
hook."""

import gc

import pytest

from relax.utils.straggler.collector import StragglerCollector
from relax.utils.straggler.detector import DetectorConfig, RankMeta
from relax.utils.straggler.stats import FIELD_INDEX


class FakeEvent:
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


class FakeWorld:
    """Simulates ``world`` ranks sharing one gather: every rank's vector is
    appended and the same table handed back."""

    def __init__(self, world):
        self.world = world
        self.tables = []
        self.pending = []

    def gather(self, values):
        self.pending.append(list(values))
        if len(self.pending) == self.world:
            table, self.pending = self.pending, []
            self.tables.append(table)
            return table
        # Other ranks' vectors are filled in by the caller-side simulation below.
        return None

    def gather_objects(self, obj):
        return [RankMeta(rank=r, dp=r, tp=0, pp=0) for r in range(self.world)]


def _collector(world, rank=0, interval=2, is_primary=True, register_gc=False, **kwargs):
    fake_world = FakeWorld(world)
    collector = StragglerCollector(
        rank_meta=RankMeta(rank=rank, dp=rank, tp=0, pp=0),
        is_primary=is_primary,
        report_interval=interval,
        detector_config=DetectorConfig(persist_windows=1),
        event_factory=FakeEvent,
        pool_size=4,
        gather=kwargs.pop("gather", None) or (lambda values: [values] * world),
        gather_objects=fake_world.gather_objects,
        register_gc_callback=register_gc,
        **kwargs,
    )
    return collector


def _one_bracket(collector, name="forward-compute"):
    collector.train_timers(name, log_level=2).start()
    collector.train_timers(name).stop()


def test_drain_folds_only_completed_events_and_recycles_them():
    collector = _collector(world=1)
    _one_bracket(collector)
    _one_bracket(collector, "backward-compute")
    # Hold the second pair back: its end event is "still running".
    _, _, pending_end, _ = collector._pending[1]
    pending_end.done = False
    assert collector.pending_events == 2
    assert collector.drain() == 1
    assert collector.pending_events == 1
    assert collector.window.get("fwd") == 1.0
    assert collector.window.get("num_fwd") == 1.0
    assert collector.window.get("bwd") == 0.0
    pending_end.done = True
    assert collector.drain() == 1
    assert collector.window.get("bwd") == 1.0
    assert collector.window.get("cpu_bwd") >= 0.0
    assert collector.window.get("overhead") >= 0.0
    # The pool got all four events back.
    assert len(collector._pool) == 4


def test_pending_queue_is_bounded_and_counts_drops():
    collector = _collector(world=1, max_pending=2)
    for _ in range(4):
        _one_bracket(collector)
    assert collector.pending_events == 2
    assert collector.window.get("dropped") == 2.0
    # Dropped pairs go back to the pool, so it settles at max_pending pairs + one in-flight pair.
    created = collector._pool.created
    assert created == 2 * 2 + 2
    for _ in range(20):
        _one_bracket(collector)
    assert collector._pool.created == created
    assert collector.window.get("dropped") == 22.0


def test_end_step_reports_every_interval_and_resets_window():
    collector = _collector(world=2, interval=3)
    reports = []
    for step in range(6):
        _one_bracket(collector)
        collector.add_tokens(1000)
        reports.append(collector.end_step(step))
    assert [report is not None for report in reports] == [False, False, True, False, False, True]
    report = reports[2]
    assert report.metrics["straggler/fwd/median_ms"] == pytest.approx(1.0)  # per step
    assert report.metrics["straggler/tokens/median"] == pytest.approx(1000.0)
    assert report.metrics["straggler/flagged/count"] == 0
    assert "straggler/gather_ms" in report.metrics
    assert report.window_end_wall >= report.window_start_wall
    assert collector.window.get("fwd") == 0.0 and collector.window.get("num_steps") == 0.0
    assert collector.all_meta is not None and len(collector.all_meta) == 2


def test_non_primary_rank_participates_but_returns_no_report():
    collector = _collector(world=2, interval=1, is_primary=False)
    _one_bracket(collector)
    assert collector.end_step(0) is None
    # It still took part in the gather (window reset afterwards).
    assert collector.window.get("fwd") == 0.0


def test_gather_table_from_other_ranks_drives_detection():
    # Rank 1 reports 3x compute for the same tokens -> flagged as slow_device.
    def gather(values):
        slow = list(values)
        for name in ("fwd", "bwd", "optim"):
            slow[FIELD_INDEX[name]] *= 3.0
        return [values, slow]

    collector = _collector(world=2, interval=1, gather=gather)
    _one_bracket(collector)
    _one_bracket(collector, "backward-compute")
    collector.add_tokens(500)
    report = collector.end_step(0)
    assert report.metrics["straggler/flagged/rank"] == 1
    assert report.alerts and report.alerts[0].reason == "slow_device"


def test_gc_callback_accumulates_and_close_unregisters():
    collector = _collector(world=1, register_gc=True)
    before = len(gc.callbacks)
    gc.collect()
    assert collector.window.get("gc") > 0.0
    collector.close()
    assert len(gc.callbacks) == before - 1
    collector.close()  # idempotent


def test_forward_only_timers_land_in_lp_buckets():
    collector = _collector(world=1)
    collector.forward_only_timers("forward-compute", log_level=2).start()
    collector.forward_only_timers("forward-compute").stop()
    collector.drain()
    assert collector.window.get("lp_fwd") == 1.0
    assert collector.window.get("fwd") == 0.0


def test_report_interval_must_be_positive():
    with pytest.raises(ValueError):
        _collector(world=1, interval=0)
