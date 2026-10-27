"""Per-request timing probe for the Relax metrics service.

Loaded automatically by CPython when this directory is on PYTHONPATH (see the
``sitecustomize`` protocol). It patches ``requests.adapters.HTTPAdapter.send``
-- the one chokepoint every request passes through, whether it goes out via the
module-level ``requests.post`` or via a ``requests.Session`` -- so the *same*
probe instruments the unfixed client and the fixed one, and neither checkout
needs a single line changed.

Nothing is enabled unless ``RELAX_MS_PROBE`` names an output directory. The
probe must not perturb what it measures: records are appended in memory and
written in batches from the ``send`` call that fills a batch, with the file
handle already open, so no timing-critical path does an ``open``.

Each record is one JSON object per line::

    {"t": <monotonic>, "wall": <epoch>, "rank": <int>, "pid": <int>,
     "ep": "/log_metrics_batch" | "/report_step" | ..., "ms": <float>,
     "ok": <bool>, "status": <int|null>, "step": <int|null>}
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time


_OUT_DIR = os.environ.get("RELAX_MS_PROBE")
if _OUT_DIR:
    import requests.adapters

    try:
        _RANK = int(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or -1)
    except ValueError:
        _RANK = -1

    # apprise / clearml pull in a vendored requests; only the real one matters.
    _PATH = os.path.join(_OUT_DIR, f"ms_probe_rank{_RANK}_pid{os.getpid()}.jsonl")
    _FLUSH_EVERY = 200

    _lock = threading.Lock()
    _records: list[str] = []
    _fh = open(_PATH, "a", buffering=1)  # line-buffered; opened once, off the hot path

    def _flush_locked() -> None:
        if _records:
            _fh.write("".join(_records))
            _records.clear()
            _fh.flush()
            os.fsync(_fh.fileno())

    def _close() -> None:
        with _lock:
            _flush_locked()
            try:
                _fh.close()
            except Exception:
                pass

    atexit.register(_close)

    _orig_send = requests.adapters.HTTPAdapter.send

    def _endpoint(url: str) -> str:
        # Strip scheme/host and any query, keep the path.
        path = url.split("://", 1)[-1] if "://" in url else url
        path = path[path.find("/") :] if "/" in path else "/"
        return path.split("?", 1)[0]

    def _step_of(body) -> object:
        if not body:
            return None
        try:
            if isinstance(body, (bytes, bytearray)):
                body = body.decode("utf-8", "replace")
            return json.loads(body).get("step")
        except Exception:
            return None

    def _send(self, request, **kwargs):
        t0 = time.perf_counter()
        status = None
        ok = True
        err = None
        try:
            response = _orig_send(self, request, **kwargs)
            status = response.status_code
            ok = 200 <= status < 300
            return response
        except BaseException as exc:
            ok = False
            err = type(exc).__name__
            raise
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            record = (
                json.dumps(
                    {
                        "t": t0,
                        "wall": time.time(),
                        "rank": _RANK,
                        "pid": os.getpid(),
                        "ep": _endpoint(request.url),
                        "ms": round(elapsed_ms, 3),
                        "ok": ok,
                        "status": status,
                        "err": err,
                        "step": _step_of(request.body),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
            with _lock:
                _records.append(record.decode())
                if len(_records) >= _FLUSH_EVERY:
                    _flush_locked()

    requests.adapters.HTTPAdapter.send = _send
