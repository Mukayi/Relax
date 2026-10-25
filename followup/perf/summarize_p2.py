"""Overhead from the in-run crossover of exp_accept_p2.sh: the profiler switched on and off every PERIOD rollouts,
each rollout timed by its full interval (rank 0's ``[abstep]`` line: from the end of the previous rollout to the end
of its own; the toggle switches after it).

The first FIRST rollouts are warm-up and excluded.

Phase 0 is on in even blocks, phase 1 in odd ones. For adjacent blocks k, k+1 of a pair (runs A, B), the estimate is
sqrt((A_on / A_off) * (B_on / B_off)) - 1: the two blocks are seconds apart, so slow changes of the host cancel, and
both runs train on the same data, so the data's own block-to-block difference cancels too. Each pair gives the mean of
its estimates; the result is their mean with a 95% CI over the pairs. Rollouts where the two runs differ by more than
STALL_S are dropped (one rank's full garbage collection stalls the others at the next collective; the check lists how
many dropped rollouts had one, from the ``[gcstat]`` lines, and how full collections split between on and off
rollouts). The A/A runs (profiler not installed) give the method's own noise: the same estimate with the same blocks,
over pseudo pairs of a phase-0 and a phase-1 run.

Usage: python summarize_p2.py <log dir> <csv out> <prefix> <pairs> <A/A prefix> <A/A pairs>
       e.g. logs p2_steps.csv p2-p 4 f19-c 2
"""

import csv
import itertools
import math
import re
import statistics as st
import sys
from collections import defaultdict

from scipy.stats import t as student_t

PERIOD, FIRST, STALL_S = 2, 20, 0.3
INTERVAL = re.compile(r"\[abstep\] rank=0 step=(\d+) interval_s=([0-9.]+)")
GCSTAT = re.compile(r"\[gcstat\] rank=(\d+) step=(\d+) gen0=\d+ gen1=\d+ gen2=(\d+)")


def load(path):
    interval, full = {}, defaultdict(set)
    with open(path, errors="replace") as log:
        for raw in log:
            line = re.sub(r"\x1b\[[0-9;]*m", "", raw)
            if match := INTERVAL.search(line):
                interval[int(match[1])] = float(match[2])
            elif (match := GCSTAT.search(line)) and int(match[3]):
                full[int(match[2])].add(int(match[1]))
    return interval, full


def estimates(a, b):
    """Adjacent-block estimates of one pair, and the rollouts kept."""
    kept = [
        s for s in sorted(set(a) & set(b)) if s >= FIRST and abs(a[s] - b[s]) <= STALL_S
    ]
    blocks = defaultdict(list)
    for s in kept:
        blocks[s // PERIOD].append((a[s], b[s]))
    mean = {
        k: (st.mean(x for x, _ in v), st.mean(y for _, y in v))
        for k, v in blocks.items()
    }
    values = []
    for k in sorted(mean):
        if k + 1 in mean:
            (a0, b0), (a1, b1) = mean[k], mean[k + 1]
            ra, rb = (a0 / a1, b1 / b0) if k % 2 == 0 else (a1 / a0, b0 / b1)
            values.append(math.sqrt(ra * rb) - 1)
    return values, set(kept)


def ci(values):
    m = st.mean(values)
    return m, float(student_t.ppf(0.975, len(values) - 1)) * st.stdev(
        values
    ) / math.sqrt(len(values))


def main():
    logs, out, prefix, pairs, aa_prefix, aa_pairs = sys.argv[1:7]
    pair_means, rows = [], []
    dropped = dropped_gc = on_full = off_full = 0
    lines = []
    for pair in range(1, int(pairs) + 1):
        (a, gc_a), (b, gc_b) = (
            load(f"{logs}/exp_accept_{prefix}{pair}-ph{phase}.log") for phase in (0, 1)
        )
        values, kept = estimates(a, b)
        pair_means.append(st.mean(values))
        for phase, full in ((0, gc_a), (1, gc_b)):
            for step, ranks in full.items():
                if step >= FIRST:
                    on = (step // PERIOD + phase) % 2 == 0
                    on_full += len(ranks) if on else 0
                    off_full += 0 if on else len(ranks)
        for s in sorted(set(a) & set(b)):
            if s < FIRST:
                continue
            stall = s not in kept
            dropped += stall
            dropped_gc += stall and any(s in gc or s + 1 in gc for gc in (gc_a, gc_b))
            k = s // PERIOD
            rows.append(
                [
                    pair,
                    s,
                    k,
                    "ph0" if k % 2 == 0 else "ph1",
                    f"{a[s]:.6f}",
                    f"{b[s]:.6f}",
                    len(gc_a.get(s, ())),
                    len(gc_b.get(s, ())),
                    int(stall),
                ]
            )
        lines.append(
            f"- pair {pair}: rollouts {len(a)}/{len(b)}; median interval ph0 {st.median(a.values()):.4f} s,"
            f" ph1 {st.median(b.values()):.4f} s; {len(values)} adjacent-block estimates, mean {pair_means[-1]:+.3%}"
        )
    m, h = ci(pair_means)
    aa = {
        phase: [
            load(f"{logs}/exp_accept_{aa_prefix}{i}-ph{phase}.log")[0]
            for i in range(1, int(aa_pairs) + 1)
        ]
        for phase in (0, 1)
    }
    aa_means = [st.mean(estimates(x, y)[0]) for x, y in itertools.product(aa[0], aa[1])]
    print(
        f"# In-run crossover `{prefix}*`, toggled every {PERIOD} rollouts, full rollout intervals (8x A100, Qwen3-0.6B"
        " SFT DP8, overlap, metrics service and trace dir on, default level)\n"
    )
    print(f"First {FIRST} rollouts are excluded as warm-up.\n")
    print("\n".join(lines))
    print(
        f"\n**overhead: {m:+.3%} (95% CI [{m - h:+.3%}, {m + h:+.3%}], n={len(pair_means)} pairs; pairs "
        + " ".join(f"{x:+.3%}" for x in pair_means)
        + ")**"
    )
    print(
        f"dropped rollouts (runs differ by > {STALL_S} s): {dropped}, {dropped_gc} of them with a full collection;"
        f" full collections (rank-rollouts) in on rollouts {on_full}, in off rollouts {off_full}"
    )
    print(
        f"A/A ({aa_prefix}*, profiler not installed), same blocks, pseudo pairs: "
        + " ".join(f"{x:+.3%}" for x in aa_means)
        + f"; sd {st.stdev(aa_means):.3%}"
    )
    with open(out, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                "pair",
                "rollout",
                "block",
                "on_run",
                "interval_ph0_s",
                "interval_ph1_s",
                "ranks_full_gc_ph0",
                "ranks_full_gc_ph1",
                "dropped",
            ]
        )
        writer.writerows(rows)


if __name__ == "__main__":
    main()
