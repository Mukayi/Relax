# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Always-on, low-overhead straggler (slow rank) analysis for the Megatron
backend.

See ``relax/utils/straggler/collector.py`` for the data flow. Enabled with the
``RELAX_STRAGGLER_PROFILER`` environment variable; off by default.
"""

from relax.utils.straggler.collector import (
    StragglerCollector,
    get_straggler_collector,
    install_straggler_collector,
    straggler_timers,
)
from relax.utils.straggler.detector import DetectorConfig, DetectorState, RankMeta, WindowReport, analyze_window
from relax.utils.straggler.reporter import report_straggler_window
from relax.utils.straggler.timers import StragglerTimers


__all__ = [
    "DetectorConfig",
    "DetectorState",
    "RankMeta",
    "StragglerCollector",
    "StragglerTimers",
    "WindowReport",
    "analyze_window",
    "get_straggler_collector",
    "install_straggler_collector",
    "report_straggler_window",
    "straggler_timers",
]
