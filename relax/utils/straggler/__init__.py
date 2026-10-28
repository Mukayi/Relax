# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Always-on, low-overhead straggler (slow rank) analysis for the Megatron
backend; off unless ``RELAX_STRAGGLER_PROFILER`` is set.

Four parts, each importing only the ones before it (a CPU test checks):

* ``core``: data collection, the part that runs inside every training step;
  standard library only.
* ``analysis``: detection, reporting and display of the collected windows; no
  torch, Megatron or other Relax imports.
* ``megatron_adapter``: Megatron's timer names, layer blocks, parallel state
  and schedule, pinned by contract tests.
* ``relax_integration``: settings, install, Gloo gathers, metrics, timeline;
  ``control_routes`` is the metrics service's side.

The training code calls only the functions below. They are imported on first
use, so that importing ``core``, ``analysis`` or ``control_routes`` does not
import torch.
"""

from typing import Any


__all__ = [
    "install_straggler_profiler",
    "note_microbatch_tokens",
    "straggler_rollout_end",
    "straggler_timers",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from relax.utils.straggler import relax_integration

        return getattr(relax_integration, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
