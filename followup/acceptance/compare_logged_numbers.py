"""Step-by-step loss and grad norm of three same-seed runs read from their
training logs (the ``step N: {...}`` lines): A/A (off vs off2) against A/B
(off vs on, off2 vs on). Writes a per-step CSV next to the summary.

Usage: python compare_logged_numbers.py <off.log> <off2.log> <on.log> <out.csv>
"""

import ast
import csv
import re
import statistics
import sys

LINE = re.compile(r"\bstep (?P<step>\d+): (?P<dict>\{.*\})\s*$")
KEYS = ("train/loss", "train/grad_norm")


def load(path):
    steps = {}
    for raw in open(path, errors="replace"):
        match = LINE.search(re.sub(r"\x1b\[[0-9;]*m", "", raw))
        if match:
            values = ast.literal_eval(match["dict"])
            steps[int(match["step"])] = {key: values[key] for key in KEYS}
    return steps


def main():
    off, off2, on = (load(path) for path in sys.argv[1:4])
    steps = sorted(set(off) & set(off2) & set(on))
    with open(sys.argv[4], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + [f"{run}_{key.split('/')[1]}" for run in ("off", "off2", "on") for key in KEYS])
        for step in steps:
            writer.writerow([step] + [run[step][key] for run in (off, off2, on) for key in KEYS])
    print(f"{len(steps)} common steps")
    for name, a, b in (("A/A off2 vs off", off, off2), ("A/B on vs off", off, on), ("A/B on vs off2", off2, on)):
        parts = [name]
        for key in KEYS:
            diffs = [abs(b[s][key] - a[s][key]) / abs(a[s][key]) * 100 for s in steps]
            parts.append(f"{key.split('/')[1]} max {max(diffs):.3f}% median {statistics.median(diffs):.3f}%")
        print(" | ".join(parts))


if __name__ == "__main__":
    main()
