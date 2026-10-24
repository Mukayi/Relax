"""Per position in the 10-rollout window: the median of each probed cost on one rank (``[cost]`` lines of
exp_accept_cost.sh), for the run with the profiler on and the one without it, and the median full interval; then the
per-window total of the top-level items (segment timers, which include the sampled-step anchor; step_metrics, which
includes drain, end_step and the window close; log_perf_data, flush and barrier, on minus off).

Usage: python cost_breakdown.py <log dir> <rank> [<on tag> <off tag>]   (default cost-on cost-off)
"""

import re
import statistics as st
import sys
from collections import defaultdict

COST = re.compile(r"\[cost\] rank=(\d+) step=(\d+) ?(.*)$")
ITEM = re.compile(r"(\w+)=([0-9.]+)/(\d+)")
INTERVAL = re.compile(r"\[abstep\] rank=(\d+) step=(\d+) interval_s=([0-9.]+)")
FIRST = 20


def load(path, rank):
    costs, intervals = defaultdict(dict), {}
    for raw in open(path, errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw).rstrip()
        if (match := COST.search(line)) and int(match[1]) == rank:
            costs[int(match[2])] = {key: (float(ms), int(calls)) for key, ms, calls in ITEM.findall(match[3])}
        elif (match := INTERVAL.search(line)) and int(match[1]) == rank:
            intervals[int(match[2])] = 1000 * float(match[3])
    return costs, intervals


def main():
    logs, rank = sys.argv[1], int(sys.argv[2])
    on_tag, off_tag = sys.argv[3:5] if len(sys.argv) > 4 else ("cost-on", "cost-off")
    runs = {"cost-on": load(f"{logs}/exp_accept_{on_tag}.log", rank), "cost-off": load(f"{logs}/exp_accept_{off_tag}.log", rank)}
    keys = sorted({key for costs, _ in runs.values() for step in costs.values() for key in step})
    print(f"# rank {rank}: median ms (calls) per rollout by position in the window, from rollout {FIRST}\n")
    print("| position | " + " | ".join(f"{key} on" if key in ("log_perf_data", "flush", "barrier") else key
                                       for key in keys) + " | log_perf_data off | flush off | barrier off |"
          " interval on | interval off |")
    print("| " + " | ".join(["---"] * (len(keys) + 6)) + " |")
    for position in range(10):
        cells = []
        for tag, wanted in (("cost-on", keys), ("cost-off", ("log_perf_data", "flush", "barrier"))):
            costs, _ = runs[tag]
            steps = [s for s in costs if s >= FIRST and s % 10 == position]
            for key in wanted:
                values = [costs[s].get(key, (0.0, 0)) for s in steps]
                cells.append(f"{st.median(v[0] for v in values):.2f} ({st.median(v[1] for v in values):.0f})")
        for tag in ("cost-on", "cost-off"):
            _, intervals = runs[tag]
            cells.append(f"{st.median(v for s, v in intervals.items() if s >= FIRST and s % 10 == position):.1f}")
        print(f"| {position} | " + " | ".join(cells) + " |")
    total = 0.0
    parts = []
    for key, minus_off in (("timer", False), ("step_metrics", False), ("log_perf_data", True), ("flush", True),
                           ("barrier", True)):
        value = 0.0
        for position in range(10):
            for tag, sign in (("cost-on", 1.0), ("cost-off", -1.0)) if minus_off else (("cost-on", 1.0),):
                costs, _ = runs[tag]
                steps = [s for s in costs if s >= FIRST and s % 10 == position]
                value += sign * st.median(costs[s].get(key, (0.0, 0))[0] for s in steps)
        parts.append(f"{key}{' on - off' if minus_off else ''} {value:.1f}")
        total += value
    print(f"\nper window (ms): {', '.join(parts)}; total {total:.1f}, i.e. {total / 10:.2f} ms per rollout")


if __name__ == "__main__":
    main()
