"""Experiment B only: make the TensorBoard backend of the metrics service slow.

On the replica's PYTHONPATH. After each real ``_TensorboardAdapter.log`` (one
call per /report_step) it sleeps for the number of milliseconds in the file
named by ``MS463_DELAY_FILE``, read on every call so the driver can change it
between levels. The adapter is the same file in both code trees, so the
injection is identical before and after the fix.
"""

import os


_DELAY_FILE = os.environ.get("MS463_DELAY_FILE")
if _DELAY_FILE:
    import time

    import relax.utils.metrics.adapters.tensorboard as _tb

    _orig_log = _tb._TensorboardAdapter.log

    def _log(self, data, step):
        _orig_log(self, data, step)
        try:
            with open(_DELAY_FILE) as f:
                delay_ms = float(f.read().strip() or 0)
        except OSError:
            delay_ms = 0.0
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

    _tb._TensorboardAdapter.log = _log
