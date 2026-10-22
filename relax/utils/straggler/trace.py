# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Sampled-step trace: every rank's GPU brackets of one step on a real time
axis, gathered to the primary rank and written as a Chrome trace for Perfetto.

Each rank is one process row, ordered by (dp, pp, tp) so the stages of a
pipeline sit next to each other. Within a row the slices nest by time:

* level 1 (PP): busy runs and the PP waits between them (``bubble <phase>``),
  plus ``untimed`` gaps that no bracket covers (host-side time);
* level 2 (fwd / bwd): inside a busy run, ``F{i}`` / ``B{i}`` per micro-batch,
  sends, the gradient sync and the optimizer;
* level 3 (modules, only at the ``modules`` level): inside ``F{i}`` / ``B{i}``,
  the forward / backward of the attention and the MLP (``moe``) of every
  layer, e.g. ``attn fwd L3``; a forward inside a ``B{i}`` is an activation
  recompute.
"""

from __future__ import annotations

import json
import os
from array import array
from dataclasses import dataclass, field
from typing import Any, Sequence

from relax.utils.logging_utils import get_logger
from relax.utils.straggler.detector import RankMeta
from relax.utils.straggler.stats import FIELD_INDEX, GPU_SEGMENTS, MODULE_INDEX, NUM_FIELDS, PP_PHASE_FIELDS


logger = get_logger(__name__)

# Above any real pid (kernel.pid_max <= 2**22) so the straggler rows never merge
# with a live process's track in Perfetto.
TIMELINE_PID_BASE = 1 << 30
# Segment, PP phase field (-1) or a module's layer number, micro-batch (-1),
# tokens (-1), start and end in ms after the rank's anchor event.
RECORD_WIDTH = 6
FILE_PREFIX = "straggler_trace_rollout"
# Gaps between brackets shorter than this are launch latency, not host stalls.
_UNTIMED_MS = 0.5

_FWD = FIELD_INDEX["fwd"]
_BWD = FIELD_INDEX["bwd"]
_LP_FWD = FIELD_INDEX["lp_fwd"]
_OPTIM = FIELD_INDEX["optim"]
_GRAD_SYNC = FIELD_INDEX["dp_grad_sync"]
_RECV = FIELD_INDEX["pp_recv"]
_LP_RECV = FIELD_INDEX["lp_pp_recv"]
_SEGMENT_NAMES = {FIELD_INDEX[name]: name for name in GPU_SEGMENTS}
_PHASE_NAMES = {FIELD_INDEX[name]: name.removeprefix("pp_wait_") for name in PP_PHASE_FIELDS}
_MODULE_NAMES = {index: name.replace("_", " ") for name, index in MODULE_INDEX.items()}
_MODULE_FORWARDS = frozenset(index for name, index in MODULE_INDEX.items() if name.endswith("_fwd"))
_WAITS_FOR = {"fill": "previous stage", "up": "previous stage", "down": "next stage", "drain": "next stage"}
_LEVEL2_NAMES = {
    "pp_send": "send",
    "lp_pp_send": "lp send",
    "dp_grad_sync": "grad sync",
    "dp_param_gather": "param gather",
    "optim": "optimizer",
}


@dataclass
class RankTrace:
    """One rank's sampled step: the wall-clock time of its anchor event, its
    brackets as flat ``RECORD_WIDTH``-float rows, how many brackets of the step
    are missing, and on the primary rank its own CPU timeline events."""

    wall: float
    records: array = field(default_factory=lambda: array("f"))
    lost: int = 0
    cpu_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _Bracket:
    segment: int
    phase: int
    micro_batch: int
    tokens: int
    start: float
    end: float

    @property
    def is_wait(self) -> bool:
        return self.segment in (_RECV, _LP_RECV)


def _brackets(trace: RankTrace) -> list[_Bracket]:
    values = trace.records
    rows = (values[i : i + RECORD_WIDTH] for i in range(0, len(values), RECORD_WIDTH))
    return sorted(
        (
            _Bracket(int(seg), int(phase), int(mb), int(tokens), start, end)
            for seg, phase, mb, tokens, start, end in rows
        ),
        key=lambda bracket: bracket.start,
    )


def _name(bracket: _Bracket) -> str:
    if bracket.segment == _FWD:
        return f"F{bracket.micro_batch}"
    if bracket.segment == _BWD:
        return f"B{bracket.micro_batch}"
    if bracket.segment == _LP_FWD:
        return f"lp F{bracket.micro_batch}"
    if bracket.segment == _LP_RECV:
        return "lp bubble"
    if bracket.segment == _RECV:
        return f"bubble {_PHASE_NAMES.get(bracket.phase, 'recv')}"
    return _LEVEL2_NAMES[_SEGMENT_NAMES[bracket.segment]]


def _slice(
    name: str, pid: int, wall_us: float, start: float, end: float, args: dict[str, Any] | None = None
) -> dict[str, Any]:
    """A slice from ``start`` to ``end`` ms after the anchor at ``wall_us``.

    In whole microseconds, like the CPU timeline: Perfetto reads a fractional
    ``ts`` as a double in ns, which this far from the epoch is off by up to 256
    ns, enough to push a slice past the end of the one it nests in.
    """
    ts = round(wall_us + start * 1e3)
    event = {
        "name": name,
        "ph": "X",
        "pid": pid,
        "tid": 0,
        "ts": ts,
        "dur": round(wall_us + end * 1e3) - ts,
    }
    if args:
        event["args"] = args
    return event


def _rank_events(brackets: list[_Bracket], pid: int, wall_us: float) -> list[dict[str, Any]]:
    """Level-1 busy runs, waits and untimed gaps, with the level-2 brackets of
    each busy run."""
    events: list[dict[str, Any]] = []
    runs: list[tuple[float, float, list[_Bracket]]] = []
    for bracket in brackets:
        if not bracket.is_wait and runs and runs[-1][2] and bracket.start - runs[-1][1] <= _UNTIMED_MS:
            start, end, members = runs[-1]
            members.append(bracket)
            runs[-1] = (start, max(end, bracket.end), members)
        else:
            runs.append((bracket.start, bracket.end, [] if bracket.is_wait else [bracket]))
            if bracket.is_wait:
                phase = _PHASE_NAMES.get(bracket.phase)
                args = {"waits_for": _WAITS_FOR[phase]} if phase else None
                events.append(_slice(_name(bracket), pid, wall_us, bracket.start, bracket.end, args))
    previous_end = None
    for start, end, members in runs:
        if previous_end is not None and start - previous_end > _UNTIMED_MS:
            events.append(_slice("untimed", pid, wall_us, previous_end, start))
        previous_end = end if previous_end is None else max(previous_end, end)
        if not members:
            continue
        events.append(_slice("busy", pid, wall_us, start, end))
        for bracket in members:
            args = {"micro_batch": bracket.micro_batch} if bracket.micro_batch >= 0 else {}
            if bracket.tokens >= 0:
                args["tokens"] = bracket.tokens
            events.append(_slice(_name(bracket), pid, wall_us, bracket.start, bracket.end, args))
    return events


def _module_events(
    modules: list[_Bracket], brackets: list[_Bracket], pid: int, wall_us: float
) -> list[dict[str, Any]]:
    """Level-3 slices.

    Each module bracket takes the micro-batch of the ``F{i}`` / ``B{i}`` / ``lp
    F{i}`` bracket that contains it; both lists are sorted by start, and
    brackets of one stream do not overlap.
    """
    outers = [b for b in brackets if b.segment in (_FWD, _BWD, _LP_FWD)]
    events: list[dict[str, Any]] = []
    j = 0
    for module in modules:
        while j < len(outers) and outers[j].end < module.start:
            j += 1
        outer = (
            outers[j] if j < len(outers) and outers[j].start <= module.start <= module.end <= outers[j].end else None
        )
        name = f"{_MODULE_NAMES[module.segment]} L{module.phase}"
        args: dict[str, Any] = {"layer": module.phase}
        if outer is not None:
            args["micro_batch"] = outer.micro_batch
            if outer.segment == _BWD and module.segment in _MODULE_FORWARDS:
                name += " (recompute)"
            elif outer.segment == _LP_FWD:
                name = f"lp {name}"
        events.append(_slice(name, pid, wall_us, module.start, module.end, args))
    return events


def _expected_bubble(brackets: list[_Bracket], stages: int) -> float | None:
    """Idle time a balanced pipeline would have in this sample.

    Non-interleaved 1F1B idles (stages - 1) x (mean fwd + mean bwd) per
    optimizer step while it fills and drains; ``None`` without PP or without
    fwd / bwd brackets.
    """
    fwd = [b.end - b.start for b in brackets if b.segment == _FWD]
    bwd = [b.end - b.start for b in brackets if b.segment == _BWD]
    if stages < 2 or not fwd or not bwd:
        return None
    optimizer_steps, in_optimizer = 0, False
    for bracket in brackets:
        if bracket.segment == _OPTIM and not in_optimizer:
            optimizer_steps += 1
        if bracket.segment in (_OPTIM, _FWD, _BWD):
            in_optimizer = bracket.segment == _OPTIM
    return max(optimizer_steps, 1) * (stages - 1) * (sum(fwd) / len(fwd) + sum(bwd) / len(bwd))


def build_trace(
    traces: Sequence[RankTrace | None], meta: Sequence[RankMeta]
) -> tuple[dict[str, Any], dict[str, float]]:
    """The Chrome trace of one sampled step and its ``straggler/trace/*``
    scalars; ``traces[i]`` belongs to ``meta[i]`` (``None`` when that rank has
    no complete sample)."""
    stages = 1 + max((m.pp for m in meta), default=0)
    events: list[dict[str, Any]] = []
    grad_sync_ends: dict[tuple[int, int], list[float]] = {}
    excess: dict[int, float] = {}
    module_brackets = 0
    order = sorted(range(len(meta)), key=lambda i: (meta[i].dp, meta[i].pp, meta[i].tp))
    for sort_index, i in enumerate(order):
        trace, rank_meta = traces[i], meta[i]
        if trace is None:
            continue
        pid = TIMELINE_PID_BASE + rank_meta.rank
        wall_us = trace.wall * 1e6
        brackets = _brackets(trace)
        modules = [b for b in brackets if b.segment >= NUM_FIELDS]
        if modules:
            brackets = [b for b in brackets if b.segment < NUM_FIELDS]
            module_brackets += len(modules)
        bubble = sum(b.end - b.start for b in brackets if b.segment == _RECV)
        expected = _expected_bubble(brackets, stages)
        name = f"pp{rank_meta.pp} dp{rank_meta.dp} tp{rank_meta.tp} · rank {rank_meta.rank}"
        if expected is not None:
            name += f" · bubble {bubble:.1f} ms, balanced 1F1B {expected:.1f} ms"
            excess[rank_meta.rank] = bubble - expected
        events += [
            {"ph": "M", "name": "process_name", "pid": pid, "args": {"name": name}},
            {"ph": "M", "name": "process_sort_index", "pid": pid, "args": {"sort_index": sort_index}},
            {"ph": "M", "name": "thread_name", "pid": pid, "tid": 0, "args": {"name": "GPU compute stream"}},
        ]
        events += _rank_events(brackets, pid, wall_us)
        events += _module_events(modules, brackets, pid, wall_us)
        syncs = [b.end for b in brackets if b.segment == _GRAD_SYNC]
        if syncs:
            grad_sync_ends.setdefault((rank_meta.pp, rank_meta.tp), []).append(trace.wall * 1e3 + syncs[-1])
        events += trace.cpu_events
        events += [
            {"ph": "M", "name": "process_name", "pid": pid_, "args": {"name": f"primary CPU (pid {pid_})"}}
            for pid_ in sorted({event["pid"] for event in trace.cpu_events})
        ]
    traced = [trace for trace in traces if trace is not None]
    metrics = {"straggler/trace/ranks": float(len(traced)), "straggler/trace/lost": float(sum(t.lost for t in traced))}
    if module_brackets:
        metrics["straggler/trace/module_brackets"] = float(module_brackets)
    spreads = [max(ends) - min(ends) for ends in grad_sync_ends.values() if len(ends) > 1]
    if spreads:
        metrics["straggler/trace/align_spread_ms"] = float(max(spreads))
    if excess:
        worst = max(excess, key=excess.__getitem__)
        metrics["straggler/trace/bubble_excess_ms_max"] = float(excess[worst])
        metrics["straggler/trace/bubble_excess_rank"] = float(worst)
    trace = {"traceEvents": events, "displayTimeUnit": "ms", "otherData": {"source": "relax straggler profiler"}}
    return trace, metrics


def write_window_trace(
    trace_dir: str, keep: int, traces: Sequence[RankTrace | None], meta: Sequence[RankMeta], first_rollout: int
) -> dict[str, float]:
    """Write the sampled step of the window starting at ``first_rollout`` and
    keep the newest ``keep`` (at least one) trace files; returns the trace
    scalars."""
    trace, metrics = build_trace(traces, meta)
    if not metrics["straggler/trace/ranks"]:
        return metrics
    os.makedirs(trace_dir, exist_ok=True)
    path = os.path.join(trace_dir, f"{FILE_PREFIX}{first_rollout:06d}.json")
    with open(f"{path}.tmp", "w") as f:
        json.dump(trace, f)
    os.replace(f"{path}.tmp", path)
    names = sorted(name for name in os.listdir(trace_dir) if name.startswith(FILE_PREFIX) and name.endswith(".json"))
    for name in names[: max(len(names) - max(keep, 1), 0)]:
        os.remove(os.path.join(trace_dir, name))
    logger.info("[straggler] trace of rollout %d written to %s", first_rollout, path)
    return metrics
