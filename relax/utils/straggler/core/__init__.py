# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Data collection, the part that runs inside every training step; standard
library only (a CPU test checks the imports).

``timers`` bracket the framework's timer call sites (``modules`` the blocks of
every layer) -> ``collector`` folds the brackets into this rank's ``stats`` row
and gathers one row per window -> the primary rank's ``worker`` thread hands
each closed window to the ``analyze`` passed in; ``health`` decides when the
profiler switches itself off. ``stats`` is everything the analysis receives.

Device events, gathers, timer names, model structure and the analysis are
passed in; the Megatron adapter and the Relax integration one level up supply
them.
"""
