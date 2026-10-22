# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for the sampled-step trace: layout, nesting, scalars, files."""

import json
from array import array

import pytest

from relax.utils.straggler.stats import FIELD_INDEX
from relax.utils.straggler.trace import (
    FILE_PREFIX,
    TIMELINE_PID_BASE,
    RankTrace,
    build_trace,
    write_window_trace,
)
from tests.utils.straggler_helpers import meta as _meta


def _bracket(segment, start, end, phase=None, micro_batch=-1, tokens=-1):
    return (FIELD_INDEX[segment], FIELD_INDEX[phase] if phase else -1, micro_batch, tokens, start, end)


# Stage 0 of a two-stage 1F1B step: two forwards, a drain wait, two backwards,
# an 8 ms host-side gap, then the gradient sync and the optimizer.
_FIRST_STAGE = [
    _bracket("fwd", 0, 10, micro_batch=0, tokens=300),
    _bracket("pp_send", 10, 11),
    _bracket("fwd", 11, 21, micro_batch=1, tokens=500),
    _bracket("pp_send", 21, 22),
    _bracket("pp_recv", 22, 40, phase="pp_wait_drain"),
    _bracket("bwd", 40, 60, micro_batch=0),
    _bracket("pp_recv", 60, 62, phase="pp_wait_drain"),
    _bracket("bwd", 62, 82, micro_batch=1),
    _bracket("dp_grad_sync", 90, 95),
    _bracket("optim", 95, 100),
]
_LAST_STAGE = [
    _bracket("pp_recv", 0, 11, phase="pp_wait_fill"),
    _bracket("fwd", 11, 21, micro_batch=0),
    _bracket("bwd", 21, 41, micro_batch=0),
    _bracket("pp_recv", 41, 42, phase="pp_wait_up"),
    _bracket("fwd", 42, 52, micro_batch=1),
    _bracket("bwd", 52, 72, micro_batch=1),
    _bracket("dp_grad_sync", 90, 95),
    _bracket("optim", 95, 100),
]


def _trace(brackets, wall=1000.0, **kwargs):
    return RankTrace(wall=wall, records=array("f", [value for bracket in brackets for value in bracket]), **kwargs)


def _slices(trace, rank, name=None):
    pid = TIMELINE_PID_BASE + rank
    return [e for e in trace["traceEvents"] if e["ph"] == "X" and e["pid"] == pid and name in (None, e["name"])]


def _meta_name(trace, rank, kind="process_name"):
    pid = TIMELINE_PID_BASE + rank
    (event,) = [e for e in trace["traceEvents"] if e["ph"] == "M" and e["name"] == kind and e["pid"] == pid]
    return event["args"]


def _two_stages(rank1_lag_s=0.0005):
    # Ranks 0, 1: stage 0 (dp 0, 1); ranks 2, 3: stage 1.
    meta = _meta(4, pp_size=2)
    traces = [
        _trace(_FIRST_STAGE),
        _trace(_FIRST_STAGE, wall=1000.0 + rank1_lag_s),
        _trace(_LAST_STAGE),
        _trace(_LAST_STAGE),
    ]
    return traces, meta


def test_rows_follow_dp_then_pp_and_name_the_measured_and_balanced_bubble():
    trace, _ = build_trace(*_two_stages())
    assert [_meta_name(trace, rank, "process_sort_index")["sort_index"] for rank in range(4)] == [0, 2, 1, 3]
    # Stage 0 waits 18 + 2 ms; a balanced two-stage 1F1B idles (2 - 1) x (10 + 20) ms per optimizer step.
    assert _meta_name(trace, 0)["name"] == "pp0 dp0 tp0 · rank 0 · bubble 20.0 ms, balanced 1F1B 30.0 ms"
    assert _meta_name(trace, 2)["name"] == "pp1 dp0 tp0 · rank 2 · bubble 12.0 ms, balanced 1F1B 30.0 ms"
    json.dumps(trace)


def test_waits_untimed_gaps_and_busy_runs_nest_the_brackets():
    trace, _ = build_trace(*_two_stages())
    wall_us = 1000.0 * 1e6
    names = [e["name"] for e in sorted(_slices(trace, 0), key=lambda e: (e["ts"], -e["dur"]))]
    assert names == [
        "busy",
        "F0",
        "send",
        "F1",
        "send",
        "bubble drain",
        "busy",
        "B0",
        "bubble drain",
        "busy",
        "B1",
        "untimed",
        "busy",
        "grad sync",
        "optimizer",
    ]
    (gap,) = _slices(trace, 0, "untimed")
    assert (gap["ts"], gap["dur"]) == (wall_us + 82e3, 8e3)
    first_busy = min(_slices(trace, 0, "busy"), key=lambda e: e["ts"])
    assert (first_busy["ts"], first_busy["dur"]) == (wall_us, 22e3)
    (f1,) = _slices(trace, 0, "F1")
    assert f1["args"] == {"micro_batch": 1, "tokens": 500}
    assert _slices(trace, 0, "bubble drain")[0]["args"] == {"waits_for": "next stage"}
    assert _slices(trace, 2, "bubble fill")[0]["args"] == {"waits_for": "previous stage"}
    assert _slices(trace, 2, "bubble up")[0]["args"] == {"waits_for": "previous stage"}


def test_slices_are_in_whole_microseconds_so_a_run_and_its_last_member_end_together():
    trace, _ = build_trace([_trace(_FIRST_STAGE, wall=1792661892.6237913)], _meta(1))
    assert all(isinstance(e["ts"], int) and isinstance(e["dur"], int) for e in _slices(trace, 0))
    busy = min(_slices(trace, 0, "busy"), key=lambda e: e["ts"])
    last_send = max(_slices(trace, 0, "send"), key=lambda e: e["ts"])
    assert busy["ts"] + busy["dur"] == last_send["ts"] + last_send["dur"]


def test_scalars_report_ranks_alignment_and_the_largest_extra_bubble():
    _, metrics = build_trace(*_two_stages(rank1_lag_s=0.0005))
    assert metrics["straggler/trace/ranks"] == 4.0
    assert metrics["straggler/trace/lost"] == 0.0
    # Ranks 0 and 1 end the same gradient sync, 0.5 ms apart on their wall clocks.
    assert metrics["straggler/trace/align_spread_ms"] == pytest.approx(0.5, abs=1e-3)
    assert (metrics["straggler/trace/bubble_excess_ms_max"], metrics["straggler/trace/bubble_excess_rank"]) == (
        pytest.approx(-10.0),
        0.0,
    )


def test_missing_samples_are_skipped_and_cpu_events_get_their_own_row():
    traces, meta = _two_stages()
    cpu_event = {"name": "get_data_batch_1", "ph": "X", "ts": 1000.0 * 1e6, "dur": 5e3, "pid": 4242, "tid": 7}
    traces[1] = None
    traces[3] = _trace(_LAST_STAGE, lost=3, cpu_events=[cpu_event])
    trace, metrics = build_trace(traces, meta)
    assert (metrics["straggler/trace/ranks"], metrics["straggler/trace/lost"]) == (3.0, 3.0)
    assert not _slices(trace, 1)
    assert cpu_event in trace["traceEvents"]
    assert {"ph": "M", "name": "process_name", "pid": 4242, "args": {"name": "primary CPU (pid 4242)"}} in trace[
        "traceEvents"
    ]


def test_without_pp_there_is_no_bubble_estimate():
    trace, metrics = build_trace([_trace(_FIRST_STAGE[:4])], _meta(1))
    assert _meta_name(trace, 0)["name"] == "pp0 dp0 tp0 · rank 0"
    assert "straggler/trace/bubble_excess_ms_max" not in metrics


def test_files_are_written_per_window_and_only_the_newest_are_kept(tmp_path):
    traces, meta = _two_stages()
    for first_rollout in (10, 20, 30):
        metrics = write_window_trace(str(tmp_path), 2, traces, meta, first_rollout)
    assert metrics["straggler/trace/ranks"] == 4.0
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        f"{FILE_PREFIX}000020.json",
        f"{FILE_PREFIX}000030.json",
    ]
    assert json.loads((tmp_path / f"{FILE_PREFIX}000030.json").read_text())["traceEvents"]

    write_window_trace(str(tmp_path / "empty"), 2, [None] * 4, meta, 40)
    assert not (tmp_path / "empty").exists()
