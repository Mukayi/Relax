# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for the straggler window analysis (pure numpy)."""

import pytest

from relax.utils.straggler.detector import REASON_CODE, DetectorConfig, DetectorState, RankMeta, analyze_window
from relax.utils.straggler.stats import FIELD_INDEX, NUM_FIELDS


def _row(steps=1, **fields):
    values = [0.0] * NUM_FIELDS
    values[FIELD_INDEX["num_steps"]] = steps
    for name, value in fields.items():
        values[FIELD_INDEX[name]] = value * steps
    return values


def _meta(world, pp_size=1):
    ranks_per_stage = world // pp_size
    return [
        RankMeta(rank=r, dp=r % ranks_per_stage, tp=0, pp=r // ranks_per_stage, host=f"h{r // 8}", device=r % 8)
        for r in range(world)
    ]


def _healthy_row(steps=10, fwd=100.0, bwd=200.0, optim=20.0, tokens=8000.0, **extra):
    return _row(steps=steps, fwd=fwd, bwd=bwd, optim=optim, tokens=tokens, dp_grad_sync=5.0, **extra)


def test_healthy_window_has_no_flags_and_per_step_normalization():
    table = [_healthy_row() for _ in range(8)]
    report = analyze_window(table, _meta(8), DetectorConfig(), DetectorState())
    assert report.metrics["straggler/flagged/count"] == 0
    assert report.metrics["straggler/flagged/rank"] == -1
    assert report.metrics["straggler/self/median_ms"] == pytest.approx(320.0)
    assert report.metrics["straggler/fwd/median_ms"] == pytest.approx(100.0)
    assert report.metrics["straggler/pp_stage_imbalance"] == 1.0
    assert report.alerts == []
    assert len(report.rows) == 8 and report.rows[3]["rank"] == 3


def test_slow_device_flagged_only_after_persist_windows():
    config = DetectorConfig(persist_windows=3)
    state = DetectorState()
    table = [_healthy_row() for _ in range(8)]
    table[5] = _healthy_row(fwd=130.0, bwd=260.0)  # fwd/bwd 30% slower, same tokens; self = 410 vs 320
    for window in range(3):
        report = analyze_window(table, _meta(8), config, state)
        if window < 2:
            assert report.metrics["straggler/flagged/count"] == 0
            assert report.metrics["straggler/self/max_rank"] == 5
            assert report.metrics["straggler/self/spread"] == pytest.approx(410.0 / 320.0 - 1.0, abs=1e-6)
    assert report.metrics["straggler/flagged/count"] == 1
    assert report.metrics["straggler/flagged/rank"] == 5
    assert report.metrics["straggler/flagged/reason"] == REASON_CODE["slow_device"]
    assert [alert.rank for alert in report.alerts] == [5]
    assert report.alerts[0].new is True
    assert "slow_device" in report.alerts[0].message
    # Same state again: still flagged but no longer "new".
    report = analyze_window(table, _meta(8), config, state)
    assert report.alerts[0].new is False


def test_flag_clears_when_rank_recovers():
    config = DetectorConfig(persist_windows=1)
    state = DetectorState()
    slow = [_healthy_row() for _ in range(4)]
    slow[2] = _healthy_row(fwd=150.0, bwd=300.0)
    assert analyze_window(slow, _meta(4), config, state).metrics["straggler/flagged/count"] == 1
    healthy = [_healthy_row() for _ in range(4)]
    report = analyze_window(healthy, _meta(4), config, state)
    assert report.metrics["straggler/flagged/count"] == 0
    assert state.consecutive == {} and state.active == {}


def test_token_heavy_rank_is_data_imbalance_not_slow_device():
    config = DetectorConfig(persist_windows=1)
    table = [_healthy_row() for _ in range(8)]
    # 40% more tokens and proportionally more compute: ms/ktok unchanged.
    table[1] = _healthy_row(fwd=140.0, bwd=280.0, optim=20.0, tokens=11200.0)
    report = analyze_window(table, _meta(8), config, DetectorState())
    assert report.metrics["straggler/flagged/rank"] == 1
    assert report.metrics["straggler/flagged/reason"] == REASON_CODE["data_imbalance"]
    assert report.metrics["straggler/tokens/spread"] == pytest.approx(0.4, abs=1e-6)
    assert report.metrics["straggler/self_per_ktok/spread"] == pytest.approx(0.0, abs=0.02)


def test_gc_heavy_rank_is_cpu_bound():
    config = DetectorConfig(persist_windows=1)
    table = [_healthy_row() for _ in range(4)]
    table[3] = _healthy_row(fwd=130.0, bwd=260.0, gc=40.0)
    report = analyze_window(table, _meta(4), config, DetectorState())
    assert report.metrics["straggler/flagged/reason"] == REASON_CODE["cpu_bound"]
    assert report.metrics["straggler/gc/max_ms"] == pytest.approx(40.0)


def test_pp_groups_are_compared_separately_and_stage_imbalance_reported():
    config = DetectorConfig(persist_windows=1)
    # PP=2, 4 ranks per stage; stage 1 is uniformly 50% heavier (uneven split),
    # which must not flag anyone.
    table = [_healthy_row() for _ in range(4)] + [_healthy_row(fwd=150.0, bwd=300.0, optim=30.0) for _ in range(4)]
    report = analyze_window(table, _meta(8, pp_size=2), config, DetectorState())
    assert report.metrics["straggler/flagged/count"] == 0
    assert report.metrics["straggler/pp_stage_imbalance"] == pytest.approx(1.5)


def test_downstream_stage_waiting_on_slow_upstream_is_not_flagged_as_slow():
    config = DetectorConfig(persist_windows=1)
    table = [_healthy_row() for _ in range(4)] + [_healthy_row(fwd=150.0, bwd=300.0, optim=30.0) for _ in range(4)]
    table[6] = _healthy_row(fwd=150.0, bwd=300.0, optim=30.0, pp_recv=80.0)  # waits much more than its stage peers
    report = analyze_window(table, _meta(8, pp_size=2), config, DetectorState())
    assert report.metrics["straggler/flagged/count"] == 0
    assert report.metrics["straggler/waiting/count"] == 1
    assert [(alert.rank, alert.reason) for alert in report.alerts] == [(6, "upstream_wait")]


def test_two_rank_group_uses_leave_one_out_reference():
    config = DetectorConfig(persist_windows=1, rel_threshold=0.10)
    table = [_healthy_row(), _healthy_row(fwd=115.0, bwd=230.0)]  # self 365 vs 320 = 14% slower than its only peer
    report = analyze_window(table, _meta(2), config, DetectorState())
    assert report.metrics["straggler/flagged/rank"] == 1
    assert report.metrics["straggler/self/spread"] == pytest.approx(365.0 / 320.0 - 1.0, abs=1e-6)


def test_wait_below_absolute_guard_is_not_reported():
    config = DetectorConfig(persist_windows=1, wait_abs_frac=0.05)
    table = [_healthy_row() for _ in range(4)]
    table[2] = _healthy_row(pp_recv=2.0)  # peers 0 -> relative excess is huge, but 2 ms of 320 ms is noise
    report = analyze_window(table, _meta(4), config, DetectorState())
    assert report.metrics["straggler/waiting/count"] == 0
    assert report.alerts == []
    assert report.metrics["straggler/pp_recv/max_rank"] == 2
    assert report.metrics["straggler/pp_recv/spread"] == 100.0  # capped


def test_missing_tokens_disables_token_metrics():
    table = [_row(steps=5, fwd=10.0, bwd=20.0) for _ in range(2)]
    report = analyze_window(table, _meta(2), DetectorConfig(), DetectorState())
    assert "straggler/tokens/median" not in report.metrics
    assert "straggler/self_per_ktok/median_ms" not in report.metrics
    assert report.metrics["straggler/self/median_ms"] == pytest.approx(30.0)


def test_overhead_and_dropped_are_aggregated():
    table = [_row(steps=4, fwd=1.0, overhead=0.5, dropped=2.0) for _ in range(3)]
    report = analyze_window(table, _meta(3), DetectorConfig(), DetectorState())
    assert report.metrics["straggler/self_overhead_ms"] == pytest.approx(0.5)
    assert report.metrics["straggler/dropped_events"] == pytest.approx(24.0)


def test_table_and_meta_size_mismatch_raises():
    with pytest.raises(ValueError):
        analyze_window([_healthy_row()], _meta(2), DetectorConfig(), DetectorState())
