# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Fixed layout of the per-rank straggler statistics vector.

Every training rank accumulates one ``WindowStats`` over ``report_interval``
steps and ships it as a flat ``float64[NUM_FIELDS]`` vector, so the cross-rank
gather is a single small collective regardless of model or parallelism.
"""

from __future__ import annotations

from typing import Mapping

from relax.utils.straggler.health import HEALTH_DISABLE


# Compute-stream time (ms, summed over the window) per coarse segment. The
# first block is fed by Megatron's own ``config.timers`` call sites (see
# MEGATRON_TIMER_SEGMENTS); the ``lp_*`` block is the same call sites during
# ``forward_only`` (log-prob passes). Adding a segment (e.g. for MoE dispatch
# hooks) is one entry here plus one mapping below.
GPU_SEGMENTS: tuple[str, ...] = (
    "fwd",
    "bwd",
    "optim",
    "pp_recv",
    "pp_send",
    "dp_grad_sync",
    "dp_param_gather",
    "lp_fwd",
    "lp_pp_recv",
    "lp_pp_send",
)
# The training-phase ``pp_recv`` split by the 1F1B phase of each bracket (ms,
# summed over the window); ``pp_recv`` keeps the total. See PP_PHASE_TIMERS.
PP_PHASE_FIELDS: tuple[str, ...] = ("pp_wait_fill", "pp_wait_up", "pp_wait_down", "pp_wait_drain")
# CPU wall time (ms) of the same fwd / bwd brackets, and Python GC pauses that
# happened while a step was in flight.
CPU_FIELDS: tuple[str, ...] = ("cpu_fwd", "cpu_bwd", "gc")
# Counters: tokens processed by this rank, number of fwd brackets, number of
# steps in the window, event pairs dropped because the pending queue was full,
# and the collector's own CPU cost (ms) spent draining events.
COUNT_FIELDS: tuple[str, ...] = ("tokens", "num_fwd", "num_steps", "dropped", "overhead")
# Window-level extremes over single training micro-batches, not summed per
# step: the slowest and fastest fwd / bwd bracket (ms), which forward of its
# step the slowest one was (from 0) and its token count (-1 when the training
# forward_step did not report one), and the largest and smallest token counts.
# A minimum, or a token extreme, of 0 means no sample.
MICRO_BATCH_FIELDS: tuple[str, ...] = (
    "mb_fwd_max",
    "mb_fwd_min",
    "mb_bwd_max",
    "mb_bwd_min",
    "mb_fwd_argmax",
    "mb_fwd_argmax_tokens",
    "mb_tokens_max",
    "mb_tokens_min",
)
# Window-level status, not summed per step: event pairs still unread when the
# window closed (their time lands in the next window), exceptions the profiler
# caught on this rank during the window, this rank's health code
# (``health.HEALTH_*``; 2 asks every rank to switch the profiler off), and the
# level the primary rank asks every rank to switch to, plus one (0: no request).
STATUS_FIELDS: tuple[str, ...] = ("unread", "errors", "health", "control")

FIELDS: tuple[str, ...] = (
    GPU_SEGMENTS + PP_PHASE_FIELDS + CPU_FIELDS + COUNT_FIELDS + MICRO_BATCH_FIELDS + STATUS_FIELDS
)
FIELD_INDEX: Mapping[str, int] = {name: index for index, name in enumerate(FIELDS)}
NUM_FIELDS: int = len(FIELDS)

# Module brackets of a sampled step at the ``modules`` level: the attention and
# the MLP (``moe`` when it is a MoE layer) of every transformer layer, forward
# and backward. They only go into the trace, never into a window, so their
# indices start after the last field.
MODULE_SEGMENTS: tuple[str, ...] = ("attn_fwd", "attn_bwd", "mlp_fwd", "mlp_bwd", "moe_fwd", "moe_bwd")
MODULE_INDEX: Mapping[str, int] = {name: NUM_FIELDS + index for index, name in enumerate(MODULE_SEGMENTS)}


def disabled_row() -> list[float]:
    """The vector a switched-off rank keeps sending: all zeros except the
    ``health`` field, which asks every rank to switch the profiler off.

    Same length and shape as a normal window, so the gather is unchanged; a
    rank that stops recording time must still join it, or the remaining ranks
    block until the group times out.
    """
    values = [0.0] * NUM_FIELDS
    values[FIELD_INDEX["health"]] = float(HEALTH_DISABLE)
    return values


# Segments whose sum is "time this rank spent computing on its own" versus
# "time this rank spent waiting for a peer". Used by the detector.
SELF_SEGMENTS: tuple[str, ...] = ("fwd", "bwd", "optim")
WAIT_SEGMENTS: tuple[str, ...] = ("pp_recv", "dp_grad_sync", "dp_param_gather")

# The Megatron timers of ``dp_grad_sync``, in the order finalize_model_grads
# runs them. Only ``all-grads-sync`` waits for the DP reduction; the others
# reduce over pipeline-stage or tensor-parallel peers.
GRAD_SYNC_TIMERS: tuple[str, ...] = (
    "all-grads-sync",
    "conditional-embedder-grads-all-reduce",
    "non-tensor-parallel-grads-all-reduce",
    "embedding-grads-all-reduce",
)

# Megatron timer name -> segment, for the training phase. Names come from
# megatron/core/pipeline_parallel/schedules.py, p2p_communication.py,
# distributed/finalize_model_grads.py, optimizer/optimizer.py and
# optimizer/distrib_optimizer.py. ``forward-backward`` wraps a whole step and is
# deliberately absent (it duplicates Relax's ``actor_train`` timer).
MEGATRON_TIMER_SEGMENTS: Mapping[str, str] = {
    "forward-compute": "fwd",
    "backward-compute": "bwd",
    "forward-recv": "pp_recv",
    "backward-recv": "pp_recv",
    "forward-send-backward-recv": "pp_recv",
    "backward-send-forward-recv": "pp_recv",
    "forward-send-forward-recv": "pp_recv",
    "backward-send-backward-recv": "pp_recv",
    "forward-backward-send-forward-backward-recv": "pp_recv",
    "forward-send": "pp_send",
    "backward-send": "pp_send",
    **dict.fromkeys(GRAD_SYNC_TIMERS, "dp_grad_sync"),
    "params-all-gather": "dp_param_gather",
    "optimizer-copy-to-main-grad": "optim",
    "optimizer-unscale-and-check-inf": "optim",
    "optimizer-clip-main-grad": "optim",
    "optimizer-count-zeros": "optim",
    "optimizer-inner-step": "optim",
    "optimizer-copy-main-to-model-params": "optim",
}

# Megatron timer name -> PP_PHASE_FIELDS entry, training phase only. In the
# non-interleaved 1F1B schedules (Megatron's and Relax's streaming one) the
# warmup only calls recv_forward, the steady state only the two combined
# send / recv calls and the cooldown only recv_backward, so the name gives the
# phase and the neighbour waited for: ``up`` waits for the previous stage's
# activations, ``down`` for the next stage's gradients. A balanced pipeline does
# not idle in the steady state, so ``up + down`` beyond the transfer time is
# extra bubble. The interleaved schedule's names count in ``pp_recv`` only.
PP_PHASE_TIMERS: Mapping[str, str] = {
    "forward-recv": "pp_wait_fill",
    "backward-send-forward-recv": "pp_wait_up",
    "forward-send-backward-recv": "pp_wait_down",
    "backward-recv": "pp_wait_drain",
}

# Megatron timer name -> phase slot of its brackets, training phase only: the
# PP_PHASE_FIELDS index of a PP wait, or a gradient sync's position in
# GRAD_SYNC_TIMERS. Other brackets carry -1.
TIMER_PHASES: Mapping[str, int] = {
    **{name: FIELD_INDEX[phase] for name, phase in PP_PHASE_TIMERS.items()},
    **{name: position for position, name in enumerate(GRAD_SYNC_TIMERS)},
}

# Same call sites during ``forward_only`` land in their own buckets so a slow
# log-prob pass does not pollute the training-step numbers.
FORWARD_ONLY_TIMER_SEGMENTS: Mapping[str, str] = {
    name: {"fwd": "lp_fwd", "pp_recv": "lp_pp_recv", "pp_send": "lp_pp_send"}.get(segment, segment)
    for name, segment in MEGATRON_TIMER_SEGMENTS.items()
    if segment in ("fwd", "pp_recv", "pp_send")
}


class WindowStats:
    """Accumulate one rank's window in the ``FIELDS`` layout.

    ``add`` resolves the field name; hot paths that run per event use
    ``add_index`` with an index looked up once at import time.
    """

    __slots__ = ("values",)

    def __init__(self) -> None:
        self.values: list[float] = [0.0] * NUM_FIELDS

    def add(self, field: str, value: float) -> None:
        self.values[FIELD_INDEX[field]] += value

    def add_index(self, index: int, value: float) -> None:
        self.values[index] += value

    def max_index(self, index: int, value: float) -> bool:
        """Keep the larger value; True when ``value`` is the new maximum."""
        if value > self.values[index]:
            self.values[index] = value
            return True
        return False

    def min_index(self, index: int, value: float) -> None:
        """Keep the smaller value, a stored 0 meaning no sample yet."""
        if self.values[index] == 0.0 or value < self.values[index]:
            self.values[index] = value

    def get(self, field: str) -> float:
        return self.values[FIELD_INDEX[field]]

    def set(self, field: str, value: float) -> None:
        self.values[FIELD_INDEX[field]] = value

    def reset(self) -> None:
        self.values = [0.0] * NUM_FIELDS

    def as_list(self) -> list[float]:
        return list(self.values)
