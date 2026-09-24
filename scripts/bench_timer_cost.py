"""Cost of the straggler profiler's hot path (StragglerTimers start/stop) on one GPU.

Replays one training step's Megatron timer sequence for Qwen3-0.6B SFT DP8 (PP=1) with the worst
observed 3 micro-batches: fwd/bwd x3, forward-backward (untracked), 4 grad-sync, params-all-gather,
6 optimizer-* -> 17 tracked brackets per step. Three measurements:
  1. CPU time of the timer calls alone (the part `self_overhead_ms` / gather / analyze do not cover).
  2. Wall-time delta per step when every bracket wraps one tiny kernel (launch-bound: the CPU cost
     lands directly on the critical path; worst case).
  3. The same with a 4096^2 bf16 matmul per bracket (compute-bound: CPU cost hidden behind the GPU).
The baseline for 2 and 3 is the identical loop with no timer calls, i.e. config.timers = None.

Usage: CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=<Relax> python bench_timer_cost.py
  BENCH_MODE=cpu  only measurement 1 (no compute kernels; safe on a card shared with another job)
  HOLD_MIB=0      do not reserve memory (default 1024, for an otherwise idle card)
"""

import json
import os
import socket
import statistics as st
import time

import torch

from relax.utils.straggler.collector import StragglerCollector
from relax.utils.straggler.detector import RankMeta

MICRO_BATCHES = 3
STEP = (
    ["forward-compute", "backward-compute"] * MICRO_BATCHES
    + ["forward-backward"]
    + ["all-grads-sync", "conditional-embedder-grads-all-reduce", "embedding-grads-all-reduce",
       "non-tensor-parallel-grads-all-reduce", "params-all-gather"]
    + ["optimizer-copy-to-main-grad", "optimizer-unscale-and-check-inf", "optimizer-clip-main-grad",
       "optimizer-count-zeros", "optimizer-inner-step", "optimizer-copy-main-to-model-params"]
)
TRACKED = sum(1 for n in STEP if n != "forward-backward")

MODE = os.environ.get("BENCH_MODE", "all")
# Hold > 1 GiB so a GPU queue that treats < 1 GiB as free does not dispatch onto this card mid-run.
_hold = torch.empty(int(os.environ.get("HOLD_MIB", "1024")) * 2**20, dtype=torch.uint8, device="cuda")

collector = StragglerCollector(
    rank_meta=RankMeta(rank=0, dp=0, tp=0, pp=0),
    is_primary=True,
    report_interval=10**9,  # never gather: this is a single process
    gather=lambda v: [v],
    gather_objects=lambda m: [m],
)
timers = collector.train_timers


def step_timers_only() -> float:
    t0 = time.perf_counter()
    for name in STEP:
        t = timers(name, log_level=1)
        t.start()
        t.stop()
    return time.perf_counter() - t0


def step_with_work(work, with_timers: bool) -> None:
    for name in STEP:
        if with_timers:
            t = timers(name, log_level=1)
            t.start()
            work()
            t.stop()
        else:
            work()


def wall(work, with_timers: bool, steps: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step_with_work(work, with_timers)
        if with_timers:
            collector.end_step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps


def main() -> None:
    res = {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(), "tracked_brackets_per_step": TRACKED}

    # 1. CPU cost of the timer calls alone
    for _ in range(500):
        step_timers_only()
        collector.end_step()
    torch.cuda.synchronize()
    per_step = []
    for _ in range(20):
        acc = 0.0
        for _ in range(500):
            acc += step_timers_only()
            collector.end_step()
        per_step.append(acc / 500 * 1e3)
    res["cpu_timer_calls_ms_per_step_median"] = st.median(per_step)
    res["cpu_timer_calls_ms_per_step_max"] = max(per_step)
    res["cpu_us_per_bracket"] = st.median(per_step) * 1e3 / TRACKED
    if MODE == "cpu":
        res["pool_created"] = collector._pool.created
        print(json.dumps(res, indent=2))
        return

    # 2. launch-bound and 3. compute-bound wall-time deltas, alternating off/on
    tiny = torch.zeros(1, device="cuda")
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    works = {
        "launch_bound": (lambda: tiny.add_(1.0), 2000),
        "compute_bound": (lambda: torch.mm(a, b), 100),
    }
    for label, (work, steps) in works.items():
        wall(work, False, 50), wall(work, True, 50)  # warm-up
        deltas, offs = [], []
        for _ in range(10):
            off = wall(work, False, steps)
            on = wall(work, True, steps)
            deltas.append((on - off) * 1e3)
            offs.append(off * 1e3)
        res[f"{label}_step_ms_off_median"] = st.median(offs)
        res[f"{label}_delta_ms_per_step_median"] = st.median(deltas)
        res[f"{label}_delta_ms_per_step_max"] = max(deltas)

    res["pool_created"] = collector._pool.created
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
