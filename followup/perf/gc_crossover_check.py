"""For crossover pairs run with GCSTAT=1: are the steps summarize_perf.py drops (the two runs differ by more
than 0.3 s) full garbage collections, and do full collections fall in on and off blocks alike?

Usage: python gc_crossover_check.py <log dir> <first pair> <last pair>
"""

import ast
import re
import sys
from collections import defaultdict

PERIOD, FIRST, STALL_S = 10, 20, 0.3
GCSTAT = re.compile(r"\[gcstat\] rank=(\d+) step=(\d+) gen0=\d+ gen1=\d+ gen2=(\d+) gen2_ms=[0-9.]+")
PERF = re.compile(r"perf (\d+): (\{.*\})\s*$")


def load(path):
    """Per rank and step, the full collections; per step, perf/step_time."""
    gc, times = defaultdict(dict), {}
    for raw in open(path, errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw)
        if match := GCSTAT.search(line):
            gc[int(match[1])][int(match[2])] = int(match[3])
        elif match := PERF.search(line):
            times[int(match[1])] = ast.literal_eval(match[2])["perf/step_time"]
    return gc, times


def full_steps(gc):
    """Steps in which some rank ran a full collection, and the full collections per step."""
    count = {}
    for per in gc.values():
        for step, c in per.items():
            if c:
                count[step] = count.get(step, 0) + c
    return count


def main():
    logs, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    dropped = explained = 0
    on_total = off_total = 0
    for pair in range(first, last + 1):
        runs = [load(f"{logs}/exp_accept_pf-p{pair}-ph{phase}.log") for phase in (0, 1)]
        times = [run[1] for run in runs]
        full = [full_steps(run[0]) for run in runs]
        steps = range(FIRST, min(max(t) for t in times) + 1)
        stalls = [s for s in steps if abs(times[0][s] - times[1][s]) > STALL_S]
        hit = [s for s in stalls if any(s in f or s - 1 in f for f in full)]
        dropped += len(stalls)
        explained += len(hit)
        for phase, counts in enumerate(full):
            on = sum(n for s, n in counts.items() if s >= FIRST and (s // PERIOD + phase) % 2 == 0)
            off = sum(n for s, n in counts.items() if s >= FIRST and (s // PERIOD + phase) % 2 == 1)
            on_total += on
            off_total += off
            print(f"pf-p{pair}-ph{phase}: full collections (all ranks) in on blocks {on}, in off blocks {off}")
        print(f"pf-p{pair}: dropped steps {stalls}; with a full collection in that or the previous step {hit}")
    print(f"total: {explained} of {dropped} dropped steps had a full collection; "
          f"full collections in on blocks {on_total}, in off blocks {off_total}")


if __name__ == "__main__":
    main()
