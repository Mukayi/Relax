# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""The profiler itself, free of torch, Megatron and the rest of Relax (a CPU
test checks the imports): ``timers`` -> ``collector`` (per rank, one gather per
window) -> ``worker`` thread on the primary rank -> ``detector`` -> ``report``
/ ``trace``.

Device events, gathers, timer names and model structure are passed in; the
Megatron adapter and the Relax integration one level up supply them.
"""
