# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for the module brackets of the ``modules`` level: which blocks are
hooked, and the forward / backward brackets they record, with and without
activation recompute."""

import pytest


torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

from relax.utils.straggler.modules import ModuleTimers, module_block  # noqa: E402


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
        name, backward = module_block(segment)
        self.pushed.append((f"{name}_{'bwd' if backward else 'fwd'}", layer, start, end))

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


class _Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 2)

    def forward(self, hidden_states):
        return self.gate(hidden_states).softmax(-1)


class _Dispatcher:
    """A plain object with the token exchange as methods, like Megatron's token
    dispatchers."""

    def token_dispatch(self, hidden_states, probs):
        return hidden_states * 1.0, probs

    def token_combine(self, hidden_states):
        return hidden_states * 1.0


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, hidden_states, probs):
        return self.proj(hidden_states) * probs[..., :1], None


class _RoutedMoELayer(nn.Module):
    """Megatron's MoE layer in miniature: route, dispatch, experts, combine,
    shared experts."""

    def __init__(self):
        super().__init__()
        self.router = _Router()
        self.token_dispatcher = _Dispatcher()
        self.experts = _Experts()
        self.shared_experts = nn.Linear(4, 4)

    def forward(self, hidden_states, padding_mask=None):
        probs = self.router(hidden_states)
        tokens, probs = self.token_dispatcher.token_dispatch(hidden_states.clone(), probs)
        output, _ = self.experts(tokens, probs)
        output = self.token_dispatcher.token_combine(output).view_as(hidden_states)
        return output + self.shared_experts(hidden_states), None


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
    assert ModuleTimers([model], _Sink()).counts() == {"attn": 3, "mlp": 1, "moe": 1}


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


def test_the_parts_of_a_moe_layer_nest_inside_it_forward_and_backward_and_leave_gradients_unchanged():
    torch.manual_seed(0)
    model = nn.Sequential(_Layer(1, mlp=_RoutedMoELayer()))
    x = torch.randn(2, 4, requires_grad=True)
    _run(model, x)
    expected = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()

    parts = ("router", "dispatch", "experts", "combine", "shared experts")
    sink = _Sink()
    timers = ModuleTimers([model], sink)
    assert timers.counts() == {"attn": 1, "moe": 1, **dict.fromkeys(parts, 1)}
    timers.attach()
    _run(model, x)
    spans = {name: (start, end) for name, _, start, end in sink.pushed}
    for direction in ("fwd", "bwd"):
        outer_start, outer_end = spans[f"moe_{direction}"]
        for part in parts:
            start, end = spans[f"{part}_{direction}"]
            assert outer_start < start < end < outer_end, (part, direction, spans)
    assert all(torch.equal(p.grad, e) for p, e in zip(model.parameters(), expected))
    assert (sink.discarded, sink.errors) == ([], [])

    timers.detach()
    assert "token_dispatch" not in model[0].mlp.token_dispatcher.__dict__


def test_a_method_spec_finds_nothing_on_an_object_without_the_method():
    moe = _RoutedMoELayer()
    moe.token_dispatcher = object()
    counts = ModuleTimers([_Layer(1, mlp=moe)], _Sink()).counts()
    assert counts == {"attn": 1, "moe": 1, "router": 1, "experts": 1, "shared experts": 1}


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
