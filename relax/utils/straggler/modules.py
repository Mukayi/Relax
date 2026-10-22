# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Module brackets for the ``modules`` level: forward and backward of the
attention and of the MLP / MoE block of every transformer layer, recorded on
the sampled step only.

The hooks are attached when a sampled step at that level starts and removed
when it ends, so no other step pays for them. Forward uses module pre / post
hooks. Backward uses gradient hooks on the block's output (its gradient
arriving opens the bracket) and on its input (closes it), which also holds
under activation recompute, where the block's forward runs again inside the
backward. Which micro-batch a bracket belongs to, and whether a forward was a
recompute, is read off the trace by containment (see ``trace``).
"""

from __future__ import annotations

from typing import Any, Iterable

from relax.utils.straggler.stats import MODULE_INDEX
from relax.utils.straggler.timers import EventSink


_SKIPPED_BLOCKS = frozenset({"IdentityOp", "IdentityFuncOp"})


class ModuleTimers:
    """The attention and MLP / MoE blocks of the transformer layers found in
    ``model_chunks``, bracketed into ``sink`` (the collector) while attached.

    A layer is any module with ``self_attention``, ``mlp`` and ``layer_number``
    (Megatron's ``TransformerLayer``, including the one inside an MTP layer,
    whose numbers start at 1 again and are recorded negated); identity
    placeholders are skipped.
    """

    def __init__(self, model_chunks: Iterable[Any], sink: EventSink) -> None:
        self._sink = sink
        self._blocks: list[tuple[Any, str, int]] = []
        for chunk in model_chunks:
            for module in chunk.modules():
                layer = getattr(module, "layer_number", None)
                if layer is None:
                    continue
                layer = -int(layer) if getattr(module, "is_mtp_layer", False) else int(layer)
                for name in ("self_attention", "mlp"):
                    block = getattr(module, name, None)
                    if block is None or type(block).__name__ in _SKIPPED_BLOCKS:
                        continue
                    kind = "attn" if name == "self_attention" else ("moe" if "MoE" in type(block).__name__ else "mlp")
                    self._blocks.append((block, kind, layer))
        self._handles: list[Any] = []

    @property
    def blocks(self) -> int:
        return len(self._blocks)

    def attach(self) -> None:
        if self._handles:
            return
        for block, kind, layer in self._blocks:
            hooks = _BlockHooks(self._sink, MODULE_INDEX[f"{kind}_fwd"], MODULE_INDEX[f"{kind}_bwd"], layer)
            self._handles.append(block.register_forward_pre_hook(hooks.pre, with_kwargs=True))
            self._handles.append(block.register_forward_hook(hooks.post, with_kwargs=True))

    def detach(self) -> None:
        handles, self._handles = self._handles, []
        for handle in handles:
            handle.remove()


def _grad_tensor(value: Any) -> Any:
    """``value``, or the first element of a tuple / list, when it is a tensor
    that takes part in the backward; ``None`` otherwise."""
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None
    return value if getattr(value, "requires_grad", False) else None


def _open(sink: EventSink) -> Any:
    """The start event of a bracket; ``None`` when none could be recorded."""
    try:
        return sink.record_event()
    except Exception as exc:
        sink.on_error("modules", exc)
        return None


def _close(sink: EventSink, segment: int, layer: int, start: Any) -> bool:
    """End the bracket opened by ``start`` and push it; ``False`` when it was
    dropped instead."""
    try:
        end = sink.record_event()
        if end is not None:
            sink.push(segment, layer, start, end, 0.0)
            return True
        sink.discard_event(start)
    except Exception as exc:
        sink.discard_event(start)
        sink.on_error("modules", exc)
    return False


class _BlockHooks:
    """Forward bracket of one block, and the gradient hooks that bracket its
    backward.

    Never raises into the model: failures go to ``sink.on_error``.
    """

    __slots__ = ("_sink", "_fwd", "_bwd", "_layer", "_start", "_input")

    def __init__(self, sink: EventSink, fwd: int, bwd: int, layer: int) -> None:
        self._sink = sink
        self._fwd = fwd
        self._bwd = bwd
        self._layer = layer
        self._start: Any = None
        self._input: Any = None

    def pre(self, module: Any, args: tuple, kwargs: dict) -> None:
        if self._start is not None:
            self._sink.discard_event(self._start)
        self._start = _open(self._sink)
        self._input = _grad_tensor(args[0] if args else kwargs.get("hidden_states"))

    def post(self, module: Any, args: tuple, kwargs: dict, output: Any) -> None:
        start, self._start = self._start, None
        source, self._input = self._input, None
        if start is None or not _close(self._sink, self._fwd, self._layer, start):
            return
        result = _grad_tensor(output)
        if result is None or source is None:
            return
        try:
            backward = _BackwardBracket(self._sink, self._bwd, self._layer)
            result.register_hook(backward.open)
            source.register_hook(backward.close)
        except Exception as exc:
            self._sink.on_error("modules", exc)


class _BackwardBracket:
    """One block's backward for one forward call: opened by the gradient of its
    output, closed by the gradient of its input.

    The hooks return ``None``, so the gradients pass through unchanged.
    """

    __slots__ = ("_sink", "_segment", "_layer", "_start")

    def __init__(self, sink: EventSink, segment: int, layer: int) -> None:
        self._sink = sink
        self._segment = segment
        self._layer = layer
        self._start: Any = None

    def open(self, grad: Any) -> None:
        if self._start is None:
            self._start = _open(self._sink)

    def close(self, grad: Any) -> None:
        start, self._start = self._start, None
        if start is not None:
            _close(self._sink, self._segment, self._layer, start)
