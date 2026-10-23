# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Module brackets for the ``modules`` level: forward and backward of the
blocks that ``MODULE_SPECS`` names inside every transformer layer (by default
the attention, the MLP / MoE block and the parts of a MoE layer), recorded on
the sampled step only.

The hooks are attached when a sampled step at that level starts and removed
when it ends, so no other step pays for them. A block that is a module is
timed with module pre / post hooks; a block that is a method (such as the token
exchange of a MoE layer) is timed by a wrapper that exists only while attached.
Backward uses gradient hooks on the block's output (its gradient arriving opens
the bracket) and on its input (closes it), which also holds under activation
recompute, where the block's forward runs again inside the backward. Which
micro-batch a bracket belongs to, and whether a forward was a recompute, is
read off the trace by containment (see ``trace``).

Timing the blocks of another model means adding specs to ``MODULE_SPECS``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from relax.utils.straggler.stats import NUM_FIELDS
from relax.utils.straggler.timers import EventSink


@dataclass(frozen=True)
class ModuleSpec:
    """One kind of block: the object at attribute ``path`` of a transformer
    layer (any module with ``layer_number``), named ``name`` in the trace.

    ``method`` times that method of the object instead of its forward, for a
    part that is not a module; ``when`` requires the object's class name to
    contain it. The first spec that matches a (path, method) pair wins.
    """

    path: str
    name: str
    method: str | None = None
    when: str | None = None


MODULE_SPECS: tuple[ModuleSpec, ...] = (
    ModuleSpec("self_attention", "attn"),
    ModuleSpec("mlp", "moe", when="MoE"),
    ModuleSpec("mlp", "mlp"),
    # Inside a MoE layer: routing, the token exchange around the experts (the EP all-to-all) and the experts.
    ModuleSpec("mlp.router", "router"),
    ModuleSpec("mlp.token_dispatcher", "dispatch", method="token_dispatch"),
    ModuleSpec("mlp.experts", "experts"),
    ModuleSpec("mlp.token_dispatcher", "combine", method="token_combine"),
    ModuleSpec("mlp.shared_experts", "shared experts"),
)
# Block names; a block's forward and backward brackets are record segments NUM_FIELDS + 2 * index (+ 1).
MODULE_NAMES: tuple[str, ...] = tuple(dict.fromkeys(spec.name for spec in MODULE_SPECS))

# Megatron's placeholders for blocks a layer does not have.
PLACEHOLDERS = frozenset({"IdentityOp", "IdentityFuncOp"})


def module_segment(name: str, backward: bool) -> int:
    """The record segment of a block's forward or backward bracket."""
    return NUM_FIELDS + 2 * MODULE_NAMES.index(name) + backward


def module_block(segment: int) -> tuple[str, bool]:
    """Block name of a module record segment and whether it is a backward."""
    offset = segment - NUM_FIELDS
    return MODULE_NAMES[offset // 2], bool(offset % 2)


def _resolve(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


class ModuleTimers:
    """The ``MODULE_SPECS`` blocks of the transformer layers found in
    ``model_chunks``, bracketed into ``sink`` (the collector) while attached.

    The layers inside an MTP layer number from 1 again and are recorded
    negated; identity placeholders are skipped.
    """

    def __init__(self, model_chunks: Iterable[Any], sink: EventSink) -> None:
        self._sink = sink
        # (object, method or None, block name, layer)
        self._blocks: list[tuple[Any, str | None, str, int]] = []
        for chunk in model_chunks:
            for module in chunk.modules():
                layer = getattr(module, "layer_number", None)
                if layer is None:
                    continue
                layer = -int(layer) if getattr(module, "is_mtp_layer", False) else int(layer)
                taken = set()
                for spec in MODULE_SPECS:
                    key = (spec.path, spec.method)
                    block = None if key in taken else _resolve(module, spec.path)
                    if block is None or type(block).__name__ in PLACEHOLDERS:
                        continue
                    if spec.method is not None and not callable(getattr(block, spec.method, None)):
                        continue
                    if spec.when is None or spec.when in type(block).__name__:
                        taken.add(key)
                        self._blocks.append((block, spec.method, spec.name, layer))
        self._handles: list[Any] = []

    def counts(self) -> Counter:
        """Blocks found, by name."""
        return Counter(name for _, _, name, _ in self._blocks)

    def attach(self) -> None:
        if self._handles:
            return
        for block, method, name, layer in self._blocks:
            hooks = _BlockHooks(self._sink, module_segment(name, False), module_segment(name, True), layer)
            if method is None:
                self._handles.append(block.register_forward_pre_hook(hooks.pre, with_kwargs=True))
                self._handles.append(block.register_forward_hook(hooks.post, with_kwargs=True))
            else:
                self._handles.append(_MethodWrapper(block, method, hooks))

    def detach(self) -> None:
        handles, self._handles = self._handles, []
        for handle in handles:
            handle.remove()


class _MethodWrapper:
    """Brackets the calls of ``owner.<method>`` with ``hooks`` by shadowing the
    method with an instance attribute until ``remove``."""

    __slots__ = ("_owner", "_method", "_saved")

    def __init__(self, owner: Any, method: str, hooks: _BlockHooks) -> None:
        self._owner = owner
        self._method = method
        self._saved = owner.__dict__.get(method)
        original = getattr(owner, method)

        def timed(*args: Any, **kwargs: Any) -> Any:
            hooks.pre(owner, args, kwargs)
            output = original(*args, **kwargs)
            hooks.post(owner, args, kwargs, output)
            return output

        setattr(owner, method, timed)

    def remove(self) -> None:
        if self._saved is None:
            delattr(self._owner, self._method)
        else:
            setattr(self._owner, self._method, self._saved)


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
