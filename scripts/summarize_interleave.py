"""Block-level crossover analysis for exp_interleave.sh.

For pair i, block k (10 steps): the "on" run is phase 0 when k is even, phase 1 when k is odd.
d_k = mean(on) / mean(off) - 1 over the block's steps; blocks 0-1 are warm-up and dropped.
Run-level offsets between the two runs add +delta on even and -delta on odd blocks, so they
cancel in the mean over an equal number of even and odd blocks.
"""

import csv
import glob
import math
import re
import statistics as st
import sys
from pathlib import Path

from scipy.stats import t as student_t
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TASK = Path("/ytech_m2v4_hdd/xiatianze/relax_work/task11")
PERIOD = 10
SKIP_BLOCKS = 2
def t975(df: int) -> float:
    return float(student_t.ppf(0.975, df))


def step_times(tag: str) -> dict[int, float]:
    files = glob.glob(str(TASK / "runs" / f"sft-Qwen3-0.6B-tp1pp1-strag*-{tag}" / "tensorboard" / "**" / "events*"), recursive=True)
    out: dict[int, float] = {}
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        if "perf/step_time" in ea.Tags()["scalars"]:
            out.update({e.step: e.value for e in ea.Scalars("perf/step_time")})
    return out


def windows_logged(tag: str) -> list[int]:
    log = TASK / "logs" / f"expil_{tag}.log"
    if not log.exists():
        return []
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace"))
    return sorted({int(s) for s in re.findall(r"\[straggler\] step=(\d+) ", text)})


def mean_ci(xs: list[float]) -> tuple[float, float]:
    m = st.mean(xs)
    return m, t975(len(xs) - 1) * st.stdev(xs) / math.sqrt(len(xs))


def main() -> None:
    # argv[1] = pair prefix: "il-p" (profiler toggled) or "il-c" (A/A control: profiler off in both runs).
    prefix = sys.argv[1] if len(sys.argv) > 1 else "il-p"
    lines = [f"# In-run on/off crossover `{prefix}*` (8x A100, Qwen3-0.6B SFT DP8, 10-step blocks)", ""]
    diffs, offsets, rows = [], [], []
    run_medians: dict[str, float] = {}
    pair = 1
    while True:
        a_tag, b_tag = f"{prefix}{pair}-ph0", f"{prefix}{pair}-ph1"
        a, b = step_times(a_tag), step_times(b_tag)
        if not a or not b:
            break
        wa, wb = windows_logged(a_tag), windows_logged(b_tag)
        ma_all = st.median(v for s, v in a.items() if s >= 20)
        mb_all = st.median(v for s, v in b.items() if s >= 20)
        run_medians[a_tag], run_medians[b_tag] = ma_all, mb_all
        lines.append(
            f"- pair {pair}: steps {len(a)}/{len(b)}; median step_time ph0 {ma_all:.3f} s, ph1 {mb_all:.3f} s; "
            f"straggler windows ph0 {wa[:4]}... ph1 {wb[:4]}..."
        )
        nblocks = (min(max(a), max(b)) + 1) // PERIOD  # steps are 0-based: 300 steps -> 30 blocks
        for k in range(SKIP_BLOCKS, nblocks):
            steps = [s for s in range(k * PERIOD, (k + 1) * PERIOD) if s in a and s in b]
            if len(steps) < PERIOD:
                continue
            ma, mb = st.mean(a[s] for s in steps), st.mean(b[s] for s in steps)
            on, off = (ma, mb) if k % 2 == 0 else (mb, ma)
            diffs.append(on / off - 1)
            offsets.append(ma / mb - 1)
            rows.append([pair, k, "ph0" if k % 2 == 0 else "ph1", ma, mb, on / off - 1])
        pair += 1

    if run_medians:
        # A foreign job sharing the GPUs slows every rank at once (the detector cannot see it);
        # a run whose median step_time is far above the fastest one is suspect.
        fastest = min(run_medians.values())
        suspect = [t for t, m in run_medians.items() if m > 1.3 * fastest]
        lines.append(f"- suspect runs (median step_time > 1.3x fastest {fastest:.3f} s): {suspect or 'none'}")
    if not diffs:
        lines.append("no complete pairs")
    else:
        m, h = mean_ci(diffs)
        mo, ho = mean_ci(offsets)
        lines += [
            "",
            f"**overhead (on/off - 1), block-level: {m:+.3%} ± {h:.3%} (95% CI, n={len(diffs)} blocks), "
            f"i.e. [{m - h:+.3%}, {m + h:+.3%}]**",
            f"run-level offset ph0/ph1 - 1 (cancels in the crossover): {mo:+.3%} ± {ho:.3%}",
            f"block sd {st.stdev(diffs):.3%}",
        ]
    out_csv = TASK / "logs" / ("expil_blocks.csv" if prefix == "il-p" else f"expil_blocks_{prefix.rstrip('-')}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "block", "on_run", "mean_step_time_ph0_s", "mean_step_time_ph1_s", "on_over_off_minus_1"])
        w.writerows(rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
