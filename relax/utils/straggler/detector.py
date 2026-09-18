# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Pure (CPU-only, torch-free) analysis of one gathered straggler window.

Input is the ``[world, NUM_FIELDS]`` table produced by every rank's
``WindowStats`` plus static per-rank metadata. Output is a flat metrics dict,
a list of alerts and a per-rank row table. Keeping this free of torch and of
process-group state makes it unit-testable with synthetic tables.

Peer groups are pipeline stages: ranks on different PP stages legitimately
have different compute times, so a rank is only ever compared with ranks that
run the same layers. Within a group every rank is compared with the
leave-one-out median of its peers, which stays meaningful for groups as small
as two ranks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from relax.utils.straggler.stats import FIELD_INDEX, GPU_SEGMENTS, SELF_SEGMENTS, WAIT_SEGMENTS


REASONS: tuple[str, ...] = ("none", "slow_device", "data_imbalance", "upstream_wait", "cpu_bound")
REASON_CODE: dict[str, int] = {reason: code for code, reason in enumerate(REASONS)}

_EPS = 1e-9


@dataclass(frozen=True)
class RankMeta:
    """Static identity of one training rank."""

    rank: int
    dp: int
    tp: int
    pp: int
    cp: int = 0
    ep: int = 0
    host: str = ""
    device: int = -1

    @property
    def tag(self) -> str:
        return f"rank{self.rank}_dp{self.dp}_tp{self.tp}_pp{self.pp}"


@dataclass(frozen=True)
class DetectorConfig:
    z_threshold: float = 3.0
    rel_threshold: float = 0.10
    persist_windows: int = 3
    gc_ratio_threshold: float = 0.05
    cpu_ratio_threshold: float = 1.2
    # Extra PP waiting must be at least this fraction of the stage's median compute to count.
    wait_abs_frac: float = 0.05


@dataclass
class DetectorState:
    """Carried across windows for the persistence rule."""

    consecutive: dict[int, int] = field(default_factory=dict)
    active: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Alert:
    rank: int
    reason: str
    message: str
    new: bool


@dataclass
class WindowReport:
    metrics: dict[str, float]
    alerts: list[Alert]
    rows: list[dict[str, float | int | str]]
    # Wall-clock span of the window; filled in by the collector for the timeline.
    window_start_wall: float = 0.0
    window_end_wall: float = 0.0


def _leave_one_out_median(values: np.ndarray) -> np.ndarray:
    """``out[i]`` is the median of ``values`` without element ``i``."""
    n = values.shape[0]
    if n < 2:
        return values.copy()
    out = np.empty_like(values)
    for i in range(n):
        out[i] = np.median(np.delete(values, i))
    return out


_REL_CAP = 100.0


def _relative_and_z(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relative excess over, robust z-score against, and median of the leave-
    one-out peers.

    z uses the MAD of the peers scaled to a normal sigma (1.4826). When the
    peers are (near) identical the MAD collapses and z explodes; the relative
    threshold in the caller is what keeps that from flagging noise. When the
    peers are (near) zero the relative excess is capped at ``_REL_CAP`` and z
    is computed against 1% of the value itself so a lone non-zero rank still
    registers; callers add an absolute-significance guard where that matters.
    """
    n = values.shape[0]
    rel = np.zeros(n)
    z = np.zeros(n)
    peer_med = values.copy()
    if n < 2:
        return rel, z, peer_med
    for i in range(n):
        peers = np.delete(values, i)
        med = float(np.median(peers))
        mad = float(np.median(np.abs(peers - med)))
        peer_med[i] = med
        if med > _EPS:
            rel[i] = min(values[i] / med - 1.0, _REL_CAP)
        else:
            rel[i] = _REL_CAP if values[i] > _EPS else 0.0
        scale = 1.4826 * mad
        if scale < _EPS:
            scale = max(_EPS, 0.01 * abs(med), 0.01 * abs(values[i]))
        z[i] = (values[i] - med) / scale
    return rel, z, peer_med


def _column(table: np.ndarray, name: str) -> np.ndarray:
    return table[:, FIELD_INDEX[name]]


def analyze_window(
    table: Sequence[Sequence[float]],
    meta: Sequence[RankMeta],
    config: DetectorConfig,
    state: DetectorState,
) -> WindowReport:
    """Analyze one gathered window.

    Mutates ``state`` for persistence.
    """
    x = np.asarray(table, dtype=np.float64)
    world = x.shape[0]
    if world != len(meta):
        raise ValueError(f"table has {world} rows but {len(meta)} rank metas")

    # Everything below is per step so numbers stay comparable across intervals.
    steps = np.maximum(_column(x, "num_steps"), 1.0)[:, None]
    x = x / steps

    self_ms = sum(_column(x, name) for name in SELF_SEGMENTS)
    wait_ms = sum(_column(x, name) for name in WAIT_SEGMENTS)
    tokens = _column(x, "tokens")
    has_tokens = bool(np.all(tokens > 0))
    per_ktok = self_ms / np.maximum(tokens, 1.0) * 1e3

    rel_self = np.zeros(world)
    z_self = np.zeros(world)
    rel_tok = np.zeros(world)
    rel_ktok = np.zeros(world)
    z_ktok = np.zeros(world)
    rel_recv = np.zeros(world)
    z_recv = np.zeros(world)
    peer_recv = np.zeros(world)
    stage_self_of = np.zeros(world)
    segment_rel = {name: np.zeros(world) for name in GPU_SEGMENTS}
    stage_median_self: dict[int, float] = {}

    groups: dict[int, list[int]] = {}
    for index, rank_meta in enumerate(meta):
        groups.setdefault(rank_meta.pp, []).append(index)

    for pp, members in groups.items():
        idx = np.asarray(members)
        stage_median_self[pp] = float(np.median(self_ms[idx]))
        stage_self_of[idx] = stage_median_self[pp]
        rel_self[idx], z_self[idx], _ = _relative_and_z(self_ms[idx])
        rel_tok[idx], _, _ = _relative_and_z(tokens[idx])
        rel_ktok[idx], z_ktok[idx], _ = _relative_and_z(per_ktok[idx])
        rel_recv[idx], z_recv[idx], peer_recv[idx] = _relative_and_z(_column(x, "pp_recv")[idx])
        for name in GPU_SEGMENTS:
            segment_rel[name][idx], _, _ = _relative_and_z(_column(x, name)[idx])

    # Persistence: a rank must be a candidate for ``persist_windows`` windows in a row.
    candidates = (rel_self >= config.rel_threshold) & (z_self >= config.z_threshold)
    for index, rank_meta in enumerate(meta):
        if candidates[index]:
            state.consecutive[rank_meta.rank] = state.consecutive.get(rank_meta.rank, 0) + 1
        else:
            state.consecutive.pop(rank_meta.rank, None)

    alerts: list[Alert] = []
    rows: list[dict[str, float | int | str]] = []
    new_active: dict[int, str] = {}
    fwd = _column(x, "fwd")
    cpu_fwd = _column(x, "cpu_fwd")
    gc_ms = _column(x, "gc")
    pp_recv = _column(x, "pp_recv")
    waiting_count = 0

    for index, rank_meta in enumerate(meta):
        reason = "none"
        if state.consecutive.get(rank_meta.rank, 0) >= config.persist_windows:
            if has_tokens and rel_tok[index] >= config.rel_threshold and rel_ktok[index] < config.rel_threshold:
                reason = "data_imbalance"
            elif (self_ms[index] > _EPS and gc_ms[index] / self_ms[index] >= config.gc_ratio_threshold) or (
                fwd[index] > _EPS and cpu_fwd[index] >= config.cpu_ratio_threshold * fwd[index]
            ):
                reason = "cpu_bound"
            else:
                reason = "slow_device"
            new_active[rank_meta.rank] = reason
            message = (
                f"rank {rank_meta.rank} ({rank_meta.tag}, host={rank_meta.host}, gpu={rank_meta.device}) "
                f"self {self_ms[index]:.1f} ms/step = {1 + rel_self[index]:.2f}x peers (z={z_self[index]:.1f}) "
                f"for {state.consecutive[rank_meta.rank]} windows; tokens {1 + rel_tok[index]:.2f}x, "
                f"ms/ktok {1 + rel_ktok[index]:.2f}x, gc {gc_ms[index]:.1f} ms -> {reason}"
            )
            alerts.append(Alert(rank_meta.rank, reason, message, new=state.active.get(rank_meta.rank) != reason))
        elif (
            not candidates[index]
            and pp_recv[index] > _EPS
            and rel_recv[index] >= config.rel_threshold
            and z_recv[index] >= config.z_threshold
            # Absolute guard: the extra waiting must matter relative to the stage's compute.
            and pp_recv[index] - peer_recv[index] >= config.wait_abs_frac * stage_self_of[index]
        ):
            reason = "upstream_wait"
            waiting_count += 1
            alerts.append(
                Alert(
                    rank_meta.rank,
                    reason,
                    f"rank {rank_meta.rank} ({rank_meta.tag}) waits on PP peers {pp_recv[index]:.1f} ms/step = "
                    f"{1 + rel_recv[index]:.2f}x its stage; its own compute is normal",
                    new=state.active.get(rank_meta.rank) != reason,
                )
            )
            new_active[rank_meta.rank] = reason
        rows.append(
            {
                "rank": rank_meta.rank,
                "tag": rank_meta.tag,
                "host": rank_meta.host,
                "device": rank_meta.device,
                "self_ms": float(self_ms[index]),
                "wait_ms": float(wait_ms[index]),
                "tokens": float(tokens[index]),
                "ms_per_ktok": float(per_ktok[index]),
                "rel_self": float(rel_self[index]),
                "z_self": float(z_self[index]),
                "gc_ms": float(gc_ms[index]),
                "reason": reason,
                **{name: float(_column(x, name)[index]) for name in GPU_SEGMENTS},
            }
        )
    state.active = new_active

    metrics: dict[str, float] = {}

    def _emit(prefix: str, values: np.ndarray, rel: np.ndarray) -> None:
        if not np.any(values > _EPS):
            return
        worst = int(np.argmax(rel))
        metrics[f"straggler/{prefix}/median_ms"] = float(np.median(values))
        metrics[f"straggler/{prefix}/max_ms"] = float(np.max(values))
        metrics[f"straggler/{prefix}/max_rank"] = float(meta[worst].rank)
        metrics[f"straggler/{prefix}/spread"] = float(max(rel[worst], 0.0))

    for name in GPU_SEGMENTS:
        _emit(name, _column(x, name), segment_rel[name])
    _emit("self", self_ms, rel_self)
    _emit("wait", wait_ms, np.zeros(world))
    if has_tokens:
        _emit("self_per_ktok", per_ktok, rel_ktok)
        metrics["straggler/tokens/median"] = float(np.median(tokens))
        metrics["straggler/tokens/max"] = float(np.max(tokens))
        metrics["straggler/tokens/spread"] = float(max(np.max(rel_tok), 0.0))
    metrics["straggler/gc/median_ms"] = float(np.median(gc_ms))
    metrics["straggler/gc/max_ms"] = float(np.max(gc_ms))

    flagged = sorted((rank, reason) for rank, reason in new_active.items() if reason != "upstream_wait")
    metrics["straggler/flagged/count"] = float(len(flagged))
    metrics["straggler/flagged/rank"] = float(flagged[0][0]) if flagged else -1.0
    metrics["straggler/flagged/reason"] = float(REASON_CODE[flagged[0][1]]) if flagged else 0.0
    metrics["straggler/waiting/count"] = float(waiting_count)
    stage_values = [value for value in stage_median_self.values() if value > _EPS]
    metrics["straggler/pp_stage_imbalance"] = (
        float(max(stage_values) / min(stage_values)) if len(stage_values) > 1 else 1.0
    )
    metrics["straggler/self_overhead_ms"] = float(np.mean(_column(x, "overhead")))
    metrics["straggler/dropped_events"] = float(np.sum(_column(x, "dropped") * steps[:, 0]))

    return WindowReport(metrics=metrics, alerts=alerts, rows=rows)
