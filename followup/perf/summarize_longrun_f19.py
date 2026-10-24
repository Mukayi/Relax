"""The long run of exp_accept_f19.sh: the profiler's per-window cost and losses over the run, GPU memory and the
python processes' RSS, and the alerts.

A leak shows up as a trend from the first to the last tenth of the windows (or of the monitor samples taken from
rollout 20, after the warm-up, to the last logged rollout); a per-window cost stays flat.

Usage: WORKDIR=... python summarize_longrun_f19.py <tag> <monitor log>
"""

import glob
import os
import re
import statistics as st
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TASK = Path(os.environ["WORKDIR"]) / "relax_work" / "task11"
TAGS = (
    "straggler/gather_ms",
    "straggler/analyze_ms",
    "straggler/latency/analyzed_ms",
    "straggler/self_overhead_ms",
    "straggler/dropped_events",
    "straggler/unread_events",
    "straggler/trace/lost",
    "straggler/health/errors",
    "straggler/flagged/count",
    "straggler/uncertain/count",
)
MONITOR = re.compile(r"^(\S+) ((?:\d+ )+)rss=(\d+) MiB$")
PERF_TIME = re.compile(r"(\d\d:\d\d:\d\d) \| INFO \| \S+ perf \d+: ")
ALERT = re.compile(r"\[straggler\] step=(\d+) rank \d+ \(.*-> (\w+)")


def tenths(values):
    n = max(1, len(values) // 10)
    return st.median(values[:n]), st.median(values[-n:])


def slope_per_100(values):
    x = range(len(values))
    mx, my = st.mean(x), st.mean(values)
    var = sum((i - mx) ** 2 for i in x)
    return 100 * sum((i - mx) * (v - my) for i, v in zip(x, values)) / var if var else 0.0


def main():
    tag, monitor = sys.argv[1], sys.argv[2]
    series = {}
    for path in glob.glob(str(TASK / "runs" / f"sft-Qwen3-0.6B-tp1pp1-strag1-{tag}" / "tensorboard" / "**" / "events*"),
                          recursive=True):
        events = EventAccumulator(path, size_guidance={"scalars": 0})
        events.Reload()
        for name in TAGS:
            if name in events.Tags()["scalars"]:
                series.setdefault(name, {}).update({e.step: e.value for e in events.Scalars(name)})
    print(f"# Long run `{tag}`: per window, first and last tenth (median), max, slope per 100 windows\n")
    for name in TAGS:
        values = [v for _, v in sorted(series.get(name, {}).items())]
        if values:
            first, last = tenths(values)
            print(f"- {name}: {len(values)} windows; {first:.2f} -> {last:.2f}; max {max(values):.2f};"
                  f" slope {slope_per_100(values):+.3f}")
    log = TASK / "logs" / f"exp_accept_{tag}.log"
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace"))
    times = PERF_TIME.findall(text)
    start, end = times[min(20, len(times) - 1)], times[-1]
    samples = []
    for line in open(monitor):
        if (match := MONITOR.match(line.strip())) and start <= match[1] <= end:
            samples.append((st.mean(int(v) for v in match[2].split()), int(match[3])))
    if samples:
        gpu, rss = [s[0] for s in samples], [s[1] for s in samples]
        print(f"- GPU memory (mean of 8, MiB): {len(gpu)} samples; {tenths(gpu)[0]:.0f} -> {tenths(gpu)[1]:.0f};"
              f" max {max(gpu):.0f}")
        print(f"- python RSS (sum, MiB): {tenths(rss)[0]:.0f} -> {tenths(rss)[1]:.0f}; max {max(rss)}")
    alerts = ALERT.findall(text)
    print(f"- windows without a straggler: {len(re.findall(r'no straggler', text))}; alerts: {len(alerts)} {alerts[:5]}")
    completed = re.findall(r"training completed step \d+", text)[-1:]
    print(f"- profiler switched off: {len(re.findall(r'profiler switched off', text))}; {completed}")


if __name__ == "__main__":
    main()
