"""Block-level crossover analysis for exp_accept_perf.sh (the method of summarize_interleave.py).

For pair i, block k (10 steps): the "on" run is phase 0 when k is even, phase 1 when k is odd.
d_k = mean(on) / mean(off) - 1 over the block's perf/step_time; blocks 0-1 are warm-up and dropped.
Run-level offsets between the two runs add +delta on even and -delta on odd blocks, so they cancel
in the mean over an equal number of even and odd blocks. The main estimate drops the steps where the
two runs differ by more than STALL_S: a full garbage collection on one rank, which falls in on and
off blocks alike (gc_crossover_check.py). Also lists the largest value of every straggler loss /
error counter each run reported.

Usage: WORKDIR=... python summarize_perf.py pf-p   (pf-c: the A/A control pair)
"""

import csv
import glob
import math
import os
import re
import statistics as st
import sys
from pathlib import Path

from scipy.stats import t as student_t
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TASK = Path(os.environ["WORKDIR"]) / "relax_work" / "task11"
PERIOD = 10
SKIP_BLOCKS = 2
STALL_S = 0.3
COUNTER = re.compile(r"^straggler/.*(lost|dropped|unread|error)")


def scalars(tag: str) -> dict[str, dict[int, float]]:
    pattern = TASK / "runs" / f"sft-Qwen3-0.6B-tp1pp1-strag*-{tag}" / "tensorboard" / "**" / "events*"
    out: dict[str, dict[int, float]] = {}
    for path in glob.glob(str(pattern), recursive=True):
        events = EventAccumulator(path, size_guidance={"scalars": 0})
        events.Reload()
        for name in events.Tags()["scalars"]:
            if name == "perf/step_time" or COUNTER.match(name):
                out.setdefault(name, {}).update({e.step: e.value for e in events.Scalars(name)})
    return out


def windows_logged(tag: str) -> list[int]:
    log = TASK / "logs" / f"exp_accept_{tag}.log"
    if not log.exists():
        return []
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace"))
    return sorted({int(s) for s in re.findall(r"\[straggler\] step=(\d+) ", text)})


def mean_ci(xs: list[float]) -> tuple[float, float]:
    return st.mean(xs), float(student_t.ppf(0.975, len(xs) - 1)) * st.stdev(xs) / math.sqrt(len(xs))


def main() -> None:
    prefix = sys.argv[1] if len(sys.argv) > 1 else "pf-p"
    lines = [
        f"# In-run on/off crossover `{prefix}*` (8x A100, Qwen3-0.6B SFT DP8, overlap on, metrics service on, "
        "10-step blocks)",
        "",
    ]
    diffs, stall_diffs, offsets, rows = [], [], [], []
    excluded = 0
    pair = 1
    while True:
        tags = (f"{prefix}{pair}-ph0", f"{prefix}{pair}-ph1")
        runs = [scalars(tag) for tag in tags]
        a, b = (run.get("perf/step_time", {}) for run in runs)
        if not any(s >= SKIP_BLOCKS * PERIOD for s in a) or not any(s >= SKIP_BLOCKS * PERIOD for s in b):
            break
        medians = [st.median(v for s, v in times.items() if s >= SKIP_BLOCKS * PERIOD) for times in (a, b)]
        counters = {
            tag: {name: max(values.values()) for name, values in sorted(run.items()) if name != "perf/step_time"}
            for tag, run in zip(tags, runs)
        }
        lines.append(
            f"- pair {pair}: steps {len(a)}/{len(b)}; median step_time ph0 {medians[0]:.3f} s, ph1 {medians[1]:.3f} s;"
            f" straggler windows ph0 {windows_logged(tags[0])[:4]}... ph1 {windows_logged(tags[1])[:4]}...;"
            f" largest loss / error counters {counters}"
        )
        for k in range(SKIP_BLOCKS, (min(max(a), max(b)) + 1) // PERIOD):
            steps = [s for s in range(k * PERIOD, (k + 1) * PERIOD) if s in a and s in b]
            if len(steps) < PERIOD:
                continue
            ma, mb = st.mean(a[s] for s in steps), st.mean(b[s] for s in steps)
            on, off = (ma, mb) if k % 2 == 0 else (mb, ma)
            diffs.append(on / off - 1)
            kept = [s for s in steps if abs(a[s] - b[s]) <= STALL_S]
            excluded += len(steps) - len(kept)
            if len(kept) >= PERIOD - 2:
                ka, kb = st.mean(a[s] for s in kept), st.mean(b[s] for s in kept)
                stall_diffs.append((ka / kb if k % 2 == 0 else kb / ka) - 1)
            offsets.append(ma / mb - 1)
            rows.append([pair, k, "ph0" if k % 2 == 0 else "ph1", ma, mb, on / off - 1])
        pair += 1

    if not diffs:
        lines.append("no complete pairs")
    else:
        ms, hs = mean_ci(stall_diffs)
        m, h = mean_ci(diffs)
        mo, ho = mean_ci(offsets)
        lines += [
            "",
            f"**overhead without stalls (steps where the runs differ by > {STALL_S} s dropped: {excluded}): "
            f"{ms:+.3%} ± {hs:.3%} (95% CI, n={len(stall_diffs)} blocks), i.e. [{ms - hs:+.3%}, {ms + hs:+.3%}]**",
            f"overhead with every step: {m:+.3%} ± {h:.3%} (95% CI, n={len(diffs)} blocks), "
            f"i.e. [{m - h:+.3%}, {m + h:+.3%}]",
            f"run-level offset ph0/ph1 - 1 (cancels in the crossover): {mo:+.3%} ± {ho:.3%}",
            f"block sd {st.stdev(diffs):.3%}",
        ]
    with open(TASK / "logs" / f"exp_accept_perf_blocks_{prefix}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair", "block", "on_run", "mean_step_time_ph0_s", "mean_step_time_ph1_s", "on_over_off_minus_1"])
        writer.writerows(rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
