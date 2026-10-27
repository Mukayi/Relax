# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check the modules-level slices of straggler traces
(``straggler_trace_rollout*.json``).

For every rank: slices per block, slices outside every ``F{i}`` / ``B{i}``
bracket (no ``micro_batch``), and parts of a MoE layer (router, dispatch,
experts, combine, shared experts) that are not inside the ``moe`` slice of the
same layer, direction and micro-batch. Exit code 1 when any slice is outside.

Usage: python check_module_trace.py <trace.json>...
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict


SLICE = re.compile(r"^(?:lp )?(?P<block>.+) (?P<direction>fwd|bwd) (?:MTP )?L\d+(?: \(recompute\))?$")
MOE_PARTS = frozenset({"router", "dispatch", "experts", "combine", "shared experts"})


def check(path: str) -> bool:
    events = json.load(open(path))["traceEvents"]
    rows = {e["pid"]: e["args"]["name"] for e in events if e["ph"] == "M" and e["name"] == "process_name"}
    clean = True
    print(path)
    for pid, row in sorted(rows.items()):
        slices = [e for e in events if e["ph"] == "X" and e["pid"] == pid and "layer" in e.get("args", {})]
        if not slices:
            continue
        counts, outside, moe, parts = Counter(), 0, defaultdict(list), []
        for event in slices:
            args = event["args"]
            match = SLICE.match(event["name"])
            block = match["block"] if match else event["name"]
            counts[block] += 1
            outside += "micro_batch" not in args
            key = (args["layer"], args.get("mtp", False), match["direction"] if match else "", args.get("micro_batch"))
            span = (event["ts"], event["ts"] + event["dur"])
            if block == "moe":
                moe[key].append(span)
            elif block in MOE_PARTS:
                parts.append((key, span))
        unnested = sum(not any(s <= start and end <= e for s, e in moe[key]) for key, (start, end) in parts)
        clean &= outside == 0 and unnested == 0
        blocks = ", ".join(f"{name} {count}" for name, count in counts.most_common())
        print(f"  {row.split(' · ')[0]}: {blocks}; outside F/B {outside}; MoE parts outside moe {unnested}")
    return clean


if __name__ == "__main__":
    results = [check(path) for path in sys.argv[1:]]
    sys.exit(0 if results and all(results) else 1)
