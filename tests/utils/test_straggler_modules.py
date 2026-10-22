# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for the module brackets of the ``modules`` level: which blocks are
hooked, and the forward / backward brackets they record, with and without
activation recompute."""

import pytest


torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

from relax.utils.straggler.modules import ModuleTimers  # noqa: E402
from relax.utils.straggler.stats import MODULE_INDEX  # noqa: E402


_NAMES = {index: name for name, index in MODULE_INDEX.items()}


class _Sink:
    """The collector side of the hooks: events are increasing integers."""

    def __init__(self):
        self.clock = 0
        self.recording = True
        self.pushed = []
        self.discarded = []
        self.errors = []

    def record_event(self):
        if not self.recording:
            return None
        self.clock += 1
        return self.clock

    def push(self, segment, layer, start, end, cpu_ms):
        self.pushed.append((_NAMES[segment], layer, start, end))

    def discard_event(self, event):
        self.discarded.append(event)

    def on_error(self, where, exc):
        self.errors.append((where, exc))

    @property
    def names(self):
        return [(name, layer) for name, layer, _, _ in self.pushed]


class _Block(nn.Module):
    """Returns ``(output, bias)`` like Megatron's attention and MLP."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, hidden_states, padding_mask=None):
        return self.proj(hidden_states), None


class MoELayer(_Block):
    pass


class IdentityOp(nn.Module):
    def forward(self, x):
        return x


class _Layer(nn.Module):
    def __init__(self, layer_number, mlp=None, recompute=False):
        super().__init__()
        self.layer_number = layer_number
        self.self_attention = _Block()
        self.mlp = mlp if mlp is not None else _Block()
        self.recompute = recompute

    def _body(self, x):
        x = x + self.self_attention(x)[0]
        return x + self.mlp(hidden_states=x)[0]

    def forward(self, x):
        return checkpoint(self._body, x, use_reentrant=True) if self.recompute else self._body(x)


class _MtpLayer(_Layer):
    is_mtp_layer = True


def _run(model, x):
    model(x).sum().backward()


def test_attention_and_mlp_of_every_layer_are_found_and_identity_blocks_skipped():
    model = nn.Sequential(_Layer(1), _Layer(2, mlp=MoELayer()), _Layer(3, mlp=IdentityOp()))
    assert ModuleTimers([model], _Sink()).blocks == 5


def test_forward_and_backward_brackets_follow_the_execution_order_and_leave_gradients_unchanged():
    torch.manual_seed(0)
    model = nn.Sequential(_Layer(1), _Layer(2, mlp=MoELayer()))
    x = torch.randn(2, 4, requires_grad=True)
    _run(model, x)
    expected = [p.grad.clone() for p in model.parameters()] + [x.grad.clone()]
    model.zero_grad()
    x.grad = None

    sink = _Sink()
    timers = ModuleTimers([model], sink)
    timers.attach()
    _run(model, x)
    assert sink.names == [
        ("attn_fwd", 1),
        ("mlp_fwd", 1),
        ("attn_fwd", 2),
        ("moe_fwd", 2),
        ("moe_bwd", 2),
        ("attn_bwd", 2),
        ("mlp_bwd", 1),
        ("attn_bwd", 1),
    ]
    spans = sorted((start, end) for _, _, start, end in sink.pushed)
    assert all(start < end <= next_start for (start, end), (next_start, _) in zip(spans, spans[1:])), spans
    assert all(torch.equal(a, b) for a, b in zip([p.grad for p in model.parameters()] + [x.grad], expected))
    assert (sink.discarded, sink.errors) == ([], [])

    timers.detach()
    _run(model, x)
    assert len(sink.pushed) == 8


def test_an_mtp_layer_is_recorded_apart_from_the_decoder_layer_with_the_same_number():
    model = nn.Sequential(_Layer(1), _MtpLayer(1))
    sink = _Sink()
    ModuleTimers([model], sink).attach()
    _run(model, torch.randn(2, 4, requires_grad=True))
    assert sink.names[:4] == [("attn_fwd", 1), ("mlp_fwd", 1), ("attn_fwd", -1), ("mlp_fwd", -1)]


def test_a_block_whose_input_needs_no_gradient_has_no_backward_bracket():
    sink = _Sink()
    model = _Layer(1)
    ModuleTimers([model], sink).attach()
    _run(model, torch.randn(2, 4))
    assert sink.names == [("attn_fwd", 1), ("mlp_fwd", 1), ("mlp_bwd", 1)]


def test_recompute_brackets_the_forward_again_inside_the_backward():
    sink = _Sink()
    model = nn.Sequential(_Layer(1, recompute=True))
    ModuleTimers([model], sink).attach()
    _run(model, torch.randn(2, 4, requires_grad=True))
    # The first forward runs without autograd, so only the recomputed one is followed by backward brackets.
    assert sink.names == [
        ("attn_fwd", 1),
        ("mlp_fwd", 1),
        ("attn_fwd", 1),
        ("mlp_fwd", 1),
        ("mlp_bwd", 1),
        ("attn_bwd", 1),
    ]


def test_nothing_is_recorded_while_the_collector_does_not_record():
    sink = _Sink()
    sink.recording = False
    model = _Layer(1)
    ModuleTimers([model], sink).attach()
    _run(model, torch.randn(2, 4, requires_grad=True))
    assert (sink.pushed, sink.discarded, sink.errors) == ([], [], [])


def test_attach_is_idempotent():
    sink = _Sink()
    model = _Layer(1)
    timers = ModuleTimers([model], sink)
    timers.attach()
    timers.attach()
    _run(model, torch.randn(2, 4, requires_grad=True))
    assert len(sink.pushed) == 4
