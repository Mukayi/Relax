"""Overhead from the in-run crossover of exp_accept_f19.sh, on each rollout's full interval.

The interval of rollout s is rank 0's ``[abstep]`` line: from the end of rollout s-1 to the end of rollout s (data
wait, training, step_metrics, logging, flush and the barrier after it); the A/B toggle switches only after it, so
each rollout's whole cost counts in its own block. Method of summarize_interleave.py: for pair i and 10-rollout
block k, the "on" run is phase 0 when k is even, phase 1 when k is odd; d_k = mean(on) / mean(off) - 1; blocks 0-1
are warm-up. Run-level offsets add +delta on even and -delta on odd blocks and cancel in the mean. The blocks of
one pair are not independent (pairs differ by more than their blocks do), so the main estimate takes one value per
pair, the mean of its blocks, and its 95% CI over the pairs; the block-level CI is listed for reference.

The main estimate drops the rollouts whose intervals differ by more than STALL_S between the two runs. A full
garbage collection on one rank stalls the others at the next collective; the check lists how many dropped rollouts
had one (``[gcstat]`` lines; a collection during rollout s is logged at s or, if it falls in the flush, at s + 1)
and how full collections split between on and off blocks. Where the cost goes: on minus off by position in the
block (median over the kept rollouts), for the interval and for rank 0's perf/actor_train_time and
perf/train_wait_time (the wait before rollout s also holds rollout s-1's work after training). The per-step CSV
holds every input.

Usage: python summarize_f19.py <log dir> <csv out> <tag prefix> <pairs>     e.g. logs f19_steps.csv f19-p 7
"""

import ast
import csv
import math
import re
import statistics as st
import sys
from collections import defaultdict

from scipy.stats import t as student_t

PERIOD, SKIP_BLOCKS, STALL_S = 10, 2, 0.3
INTERVAL = re.compile(r"\[abstep\] rank=0 step=(\d+) interval_s=([0-9.]+)")
GCSTAT = re.compile(r"\[gcstat\] rank=(\d+) step=(\d+) gen0=\d+ gen1=\d+ gen2=(\d+)")
PERF = re.compile(r"perf (\d+): (\{.*\})\s*$")
PARTS = ("perf/actor_train_time", "perf/train_wait_time")


def load(path):
    """Rank 0's interval and perf parts per rollout, and the ranks with a full collection per rollout."""
    interval, full, parts = {}, defaultdict(set), {}
    for raw in open(path, errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw)
        if match := INTERVAL.search(line):
            interval[int(match[1])] = float(match[2])
        elif (match := GCSTAT.search(line)) and int(match[3]):
            full[int(match[2])].add(int(match[1]))
        elif match := PERF.search(line):
            values = ast.literal_eval(match[2])
            parts[int(match[1])] = [values[key] for key in PARTS]
    return interval, full, parts


def mean_ci(xs):
    return st.mean(xs), float(student_t.ppf(0.975, len(xs) - 1)) * st.stdev(xs) / math.sqrt(len(xs))


def main():
    logs, out, prefix, pairs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    kept_diffs, diffs, offsets, rows, pair_means = [], [], [], [], []
    dropped = dropped_gc = on_full = off_full = 0
    lines = []
    by_position = defaultdict(list)
    for pair in range(1, pairs + 1):
        (a, gc_a, pa), (b, gc_b, pb) = (load(f"{logs}/exp_accept_{prefix}{pair}-ph{phase}.log") for phase in (0, 1))
        lines.append(f"- pair {pair}: rollouts {len(a)}/{len(b)}; median interval ph0 {st.median(a.values()):.4f} s,"
                     f" ph1 {st.median(b.values()):.4f} s")
        for phase, full in ((0, gc_a), (1, gc_b)):
            for step, ranks in full.items():
                if step >= SKIP_BLOCKS * PERIOD:
                    on = (step // PERIOD + phase) % 2 == 0
                    on_full += len(ranks) if on else 0
                    off_full += 0 if on else len(ranks)
        first_kept = len(kept_diffs)
        for k in range(SKIP_BLOCKS, (min(max(a), max(b)) + 1) // PERIOD):
            steps = [s for s in range(k * PERIOD, (k + 1) * PERIOD) if s in a and s in b]
            if len(steps) < PERIOD - 2:
                continue
            ma, mb = st.mean(a[s] for s in steps), st.mean(b[s] for s in steps)
            diffs.append((ma / mb if k % 2 == 0 else mb / ma) - 1)
            offsets.append(ma / mb - 1)
            kept = [s for s in steps if abs(a[s] - b[s]) <= STALL_S]
            for s in kept:
                on, off = ((a, pa), (b, pb)) if k % 2 == 0 else ((b, pb), (a, pa))
                by_position[s % PERIOD].append([on[0][s] - off[0][s]] + [x - y for x, y in zip(on[1][s], off[1][s])])
            for s in steps:
                stall = s not in kept
                had_gc = any(s in gc or s + 1 in gc for gc in (gc_a, gc_b))
                dropped += stall
                dropped_gc += stall and had_gc
                rows.append([pair, s, k, "ph0" if k % 2 == 0 else "ph1", f"{a[s]:.6f}", f"{b[s]:.6f}",
                             len(gc_a.get(s, ())), len(gc_b.get(s, ())), int(stall)])
            if len(kept) >= PERIOD - 2:
                ka, kb = st.mean(a[s] for s in kept), st.mean(b[s] for s in kept)
                kept_diffs.append((ka / kb if k % 2 == 0 else kb / ka) - 1)
        pair_means.append(st.mean(kept_diffs[first_kept:]))
    m, h = mean_ci(pair_means)
    lines += [
        "",
        f"**overhead without stalls, one value per pair ({dropped} rollouts dropped, {dropped_gc} of them with a full "
        f"collection): {m:+.3%} (95% CI [{m - h:+.3%}, {m + h:+.3%}], n={len(pair_means)} pairs; pairs "
        + " ".join(f"{x:+.2%}" for x in pair_means) + ")**",
    ]
    m, h = mean_ci(kept_diffs)
    lines.append(f"the same with blocks taken as independent: {m:+.3%} ± {h:.3%} (n={len(kept_diffs)} blocks)")
    m, h = mean_ci(diffs)
    mo, ho = mean_ci(offsets)
    lines += [
        f"overhead with every rollout, blocks taken as independent: {m:+.3%} ± {h:.3%} (n={len(diffs)} blocks)",
        f"run-level offset ph0/ph1 - 1 (cancels in the crossover): {mo:+.3%} ± {ho:.3%}; block sd {st.stdev(diffs):.3%}",
        f"full collections (rank-rollouts) in on blocks {on_full}, in off blocks {off_full}",
        "",
        "on minus off by position in the block, median ms (interval, actor_train_time, train_wait_time):",
    ]
    for position, values in sorted(by_position.items()):
        medians = [1000 * st.median(column) for column in zip(*values)]
        lines.append(f"- {position}: " + ", ".join(f"{m:+.1f}" for m in medians) + f" (n={len(values)})")
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair", "rollout", "block", "on_run", "interval_ph0_s", "interval_ph1_s",
                         "ranks_full_gc_ph0", "ranks_full_gc_ph1", "dropped"])
        writer.writerows(rows)
    print(f"# In-run crossover `{prefix}*` on full rollout intervals (8x A100, Qwen3-0.6B SFT DP8, overlap, "
          "metrics service and trace dir on, default level)\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
