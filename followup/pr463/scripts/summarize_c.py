#!/usr/bin/env python3
"""Summarize run_c.sh, the end-to-end A/B for #463.

Per run: rollout intervals of the reporting rank (rank 0, [abstep]); its metrics
requests from the probe (latency, failures, time per rollout); client-side
failure lines in the log; straggler alerts and the per-window late arrival and
peer idle time from TensorBoard; and how many (tag, step) points reached
TensorBoard, against the most complete run. Then the same per condition and
code version over the repetitions. Writes c_intervals.csv, c_requests.csv and
c_windows.csv next to the logs.

Usage: summarize_c.py <experiment dir> <runs dir>
"""

import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


FIRST = 20
RUNS = Path(sys.argv[2])
ABSTEP = re.compile(r"pid=(\d+)\).*\[abstep\] rank=(\d+) step=(\d+) interval_s=([\d.]+)")
ALERT = re.compile(r"\[straggler\] step=(\d+) (.*late_arrival.*)")
FAIL = re.compile(r"log_metrics_batch exception|report_step exception|log_metrics_batch failed|report_step failed|"
                  r"Failed to log metrics for step|Failed to report step")
METRIC_EPS = ("/metrics/log_metrics_batch", "/metrics/report_step")


def pct(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def parse_tag(tag):
    _, rep, cond, arm = tag.split("-")
    return rep, cond, arm


def load_run(d, tag):
    log = (d / "logs" / f"c_{tag}.log").read_text(errors="replace")
    intervals = defaultdict(dict)
    rank_of_pid = {}
    for m in ABSTEP.finditer(log):
        rank_of_pid[int(m.group(1))] = int(m.group(2))
        intervals[int(m.group(2))][int(m.group(3))] = float(m.group(4))
    last = re.findall(r"training completed step (\d+)", log)
    fails = Counter()
    for line in log.splitlines():
        if FAIL.search(line):
            kind = ("Read timed out" if "Read timed out" in line else "ProxyError" if "ProxyError" in line
                    else "Failed to log/report (follows the above)" if "Failed to" in line else "other")
            fails[kind] += 1
    alerts = [(int(m.group(1)), m.group(2)) for m in ALERT.finditer(log)]

    rank0_pids = {p for p, r in rank_of_pid.items() if r == 0}
    requests, callers = [], Counter()
    for path in sorted((d / "probe" / tag).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep = rec["ep"]
            callers[(rec["pid"] in rank0_pids, ep)] += 1
            if rec["pid"] in rank0_pids and ep in METRIC_EPS:
                requests.append(rec)
    requests.sort(key=lambda r: r["t"])

    scalars = {}
    tb = RUNS / f"sft-Qwen3-0.6B-tp1pp1-strag1-{tag}" / "tensorboard"
    for events in sorted(tb.glob("events.out.tfevents.*")):
        acc = EventAccumulator(str(events), size_guidance={"scalars": 0})
        acc.Reload()
        for t in acc.Tags()["scalars"]:
            scalars.setdefault(t, {}).update({e.step: e.value for e in acc.Scalars(t)})
    return {"tag": tag, "intervals": intervals, "last": int(last[-1]) if last else -1, "fails": fails,
            "alerts": alerts, "requests": requests, "callers": callers, "scalars": scalars}


def main():
    d = Path(sys.argv[1])
    tags = sorted(p.name[2:-4] for p in (d / "logs").glob("c_c-r*.log"))
    runs = [load_run(d, t) for t in tags]
    # Points per tag in the most complete run: what every run should have written.
    expected = Counter()
    for r in runs:
        for t, steps in r["scalars"].items():
            expected[t] = max(expected[t], len(steps))

    out_int = open(d / "logs" / "c_intervals.csv", "w", newline="")
    out_req = open(d / "logs" / "c_requests.csv", "w", newline="")
    out_win = open(d / "logs" / "c_windows.csv", "w", newline="")
    out_blk = open(d / "logs" / "c_blocked.csv", "w", newline="")
    wi, wr, ww, wb = csv.writer(out_int), csv.writer(out_req), csv.writer(out_win), csv.writer(out_blk)
    wb.writerow(["tag", "rep", "cond", "arm", "rollout", "blocked_ms", "interval_s"])
    wi.writerow(["tag", "rep", "cond", "arm", "rollout", "interval_s"])
    wr.writerow(["tag", "rep", "cond", "arm", "t_s", "endpoint", "ms", "ok", "status", "err"])
    ww.writerow(["tag", "rep", "cond", "arm", "window_step", "late_rank", "late_ms", "peer_idle_ms"])

    print("# #463 end-to-end A/B (8x A100, Qwen3-0.6B SFT DP8, overlap, metrics service on, 120 rollouts per run)\n")
    print(f"Rollouts {FIRST}+ only for intervals, windows and time blocked per rollout; request latencies are the "
          "reporting rank's (rank 0) over the rollouts the probe covers.\n")
    print("| run | last step | rollout interval median / mean (s) | metrics requests (rollouts covered) | "
          "p50 / p90 / p99 / max (ms) | failed | blocked per rollout (ms) | log failures | late_arrival alerts | "
          "late ms / peers idle ms (window mean) | TB points written |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    per_group = defaultdict(list)
    for r in runs:
        rep, cond, arm = parse_tag(r["tag"])
        r0 = r["intervals"].get(0, {})
        iv = [v for s, v in sorted(r0.items()) if s >= FIRST]
        for s, v in sorted(r0.items()):
            wi.writerow([r["tag"], rep, cond, arm, s, v])
        reqs = r["requests"]
        t0 = reqs[0]["t"] if reqs else 0.0
        for q in reqs:
            wr.writerow([r["tag"], rep, cond, arm, round(q["t"] - t0, 4), q["ep"].rsplit("/", 1)[-1], q["ms"],
                         int(q["ok"]), q["status"], q.get("err")])
        ms = [q["ms"] for q in reqs]
        failed = Counter(q.get("err") or f"HTTP {q['status']}" for q in reqs if not q["ok"])
        # The probe loses its last unflushed records when the actor is torn down, so per rollout only over the
        # rollouts its records cover; each request carries the rollout as its step.
        per_roll = defaultdict(float)
        for q in reqs:
            if q["step"] is not None:
                per_roll[q["step"]] += q["ms"]
        for s, v in sorted(per_roll.items()):
            wb.writerow([r["tag"], rep, cond, arm, s, round(v, 3), r0.get(s, "")])
        blocked = [v for s, v in per_roll.items() if s >= FIRST]
        sc = r["scalars"]
        late = sc.get("straggler/late/max_ms", {})
        idle = sc.get("straggler/late/peer_idle_ms", {})
        lrank = sc.get("straggler/late/max_rank", {})
        wins = [(s, lrank.get(s), late[s], idle.get(s)) for s in sorted(late) if s >= FIRST]
        for w in wins:
            ww.writerow([r["tag"], rep, cond, arm, *w])
        points = sum(len(v) for v in sc.values())
        exp = sum(expected.values())
        row = {
            "iv_med": statistics.median(iv) if iv else float("nan"),
            "iv_mean": statistics.fmean(iv) if iv else float("nan"),
            "n_req": len(reqs), "ms": ms, "failed": failed, "covered": len(per_roll),
            "blocked": statistics.fmean(blocked) if blocked else float("nan"),
            "late": statistics.fmean([w[2] for w in wins]) if wins else float("nan"),
            "idle": statistics.fmean([w[3] for w in wins if w[3] is not None]) if wins else float("nan"),
            "late0": sum(1 for w in wins if w[1] == 0), "wins": len(wins), "alerts": len(r["alerts"]),
            "points": points, "exp": exp, "log_fails": sum(r["fails"].values()),
        }
        per_group[(cond, arm)].append(row)
        print(f"| {r['tag']} | {r['last']} | {row['iv_med']:.3f} / {row['iv_mean']:.3f} | {len(reqs)} ({row['covered']}) | "
              f"{pct(ms, .5):.1f} / {pct(ms, .9):.1f} / {pct(ms, .99):.1f} / {max(ms, default=float('nan')):.1f} | "
              f"{dict(failed) or 0} | {row['blocked']:.0f} | {dict(r['fails']) or 0} | {row['alerts']} | "
              f"{row['late']:.1f} / {row['idle']:.1f} (latest is rank 0 in {row['late0']}/{row['wins']}) | "
              f"{points}/{exp} |")
    for f in (out_int, out_req, out_win, out_blk):
        f.close()

    print("\n## Per condition and code version (mean over repetitions; requests pooled)\n")
    print("| condition | code | runs | rollout interval median (s) | blocked per rollout (ms) | request p50 / p99 / max "
          "(ms) | failed requests | late ms / peers idle ms | TB points lost |")
    print("|---|---|---|---|---|---|---|---|---|")
    for cond in ("proxy", "noproxy"):
        for arm in ("before", "after"):
            g = [x for x in per_group.get((cond, arm), []) if x["ms"]]
            if not g:
                continue
            ms = [m for x in g for m in x["ms"]]
            failed = sum(sum(x["failed"].values()) for x in g)
            lost = 1 - sum(x["points"] for x in g) / max(1, sum(x["exp"] for x in g))
            print(f"| {cond} | {arm} | {len(g)} | {statistics.fmean(x['iv_med'] for x in g):.3f} | "
                  f"{statistics.fmean(x['blocked'] for x in g):.0f} | {pct(ms, .5):.1f} / {pct(ms, .99):.1f} / "
                  f"{max(ms):.1f} | {failed} / {len(ms)} | {statistics.fmean(x['late'] for x in g):.1f} / "
                  f"{statistics.fmean(x['idle'] for x in g):.1f} | {lost:.1%} |")
        b = [x for x in per_group.get((cond, "before"), []) if x["ms"]]
        a = [x for x in per_group.get((cond, "after"), []) if x["ms"]]
        if b and a:
            mb, ma = statistics.fmean(x["iv_med"] for x in b), statistics.fmean(x["iv_med"] for x in a)
            print(f"| {cond} | after vs before | | {100 * (ma / mb - 1):+.1f}% | | | | | |")

    print("\n## Who sends requests to the metrics service (all runs)\n")
    callers = Counter()
    for r in runs:
        callers.update(r["callers"])
    for (is_rank0, ep), n in sorted(callers.items(), key=lambda kv: -kv[1]):
        print(f"- {'rank 0' if is_rank0 else 'other processes'} {ep}: {n}")


if __name__ == "__main__":
    main()
