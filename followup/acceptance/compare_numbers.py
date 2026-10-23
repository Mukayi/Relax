"""Compare per-step loss and grad norm between profiler-off and profiler-on runs with the same seed.

A/A: two profiler-off runs; A/B: the first profiler-off run against the profiler-on run. Grad norms come from the
files the framework's --ci-save-grad-norm wrote (one per rollout and optimizer step), losses from each run's
TensorBoard ``train/loss``. "check" counts the steps that fail the framework's own --ci-load-grad-norm test,
math.isclose(rel_tol=0.01, abs_tol=0.01).

Usage: compare_numbers.py <grad_root with off1/ off2/ on/> <run_off1> <run_off2> <run_on>
"""

import glob
import math
import re
import statistics
import sys

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def grad_norms(directory):
    norms = {}
    for path in glob.glob(f"{directory}/grad-r*-s*.pt"):
        rollout, step = map(int, re.search(r"grad-r(\d+)-s(\d+)\.pt$", path).groups())
        norms[rollout, step] = float(torch.load(path))
    return norms


def losses(run_dir):
    path = glob.glob(f"{run_dir}/tensorboard/**/events.out.tfevents.*", recursive=True)[0]
    events = EventAccumulator(path, size_guidance={"scalars": 0})
    events.Reload()
    return {event.step: event.value for event in events.Scalars("train/loss")}


def compare(name, a, b):
    keys = sorted(set(a) & set(b))
    diffs = [abs(a[k] - b[k]) / abs(b[k]) for k in keys]
    failing = [k for k in keys if not math.isclose(a[k], b[k], rel_tol=0.01, abs_tol=0.01)]
    print(
        f"{name}: steps {len(keys)}, identical {sum(d == 0 for d in diffs)}, "
        f"max {max(diffs) * 100:.3f}%, median {statistics.median(diffs) * 100:.3f}%, "
        f"check fails {len(failing)}" + (f" (first {failing[0]})" if failing else "")
    )


if __name__ == "__main__":
    grad_root, run_off1, run_off2, run_on = sys.argv[1:5]
    off1, off2, on = (grad_norms(f"{grad_root}/{leg}") for leg in ("off1", "off2", "on"))
    compare("grad norm A/A", off2, off1)
    compare("grad norm A/B", on, off1)
    loss_off1, loss_off2, loss_on = losses(run_off1), losses(run_off2), losses(run_on)
    compare("loss A/A", loss_off2, loss_off1)
    compare("loss A/B", loss_on, loss_off1)
