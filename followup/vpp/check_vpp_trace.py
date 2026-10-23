"""Check the micro-batch / model-chunk identity in straggler traces of an interleaved-PP run.

Independent of how the profiler assigns identities, on real timings:
  * layers: every module slice inside ``F{i} chunk {c}`` / ``B{i} chunk {c}`` belongs to chunk c of its PP rank
    (Megatron's layer offset: chunk * layers / vpp + pp * layers / (pp * vpp));
  * tokens: the forward of one micro-batch reports the same token count on every chunk of a rank;
  * order: a micro-batch's forward on chunk c of stage s ends before it starts on the next stage
    (pp0 c0 -> pp1 c0 -> pp0 c1 -> pp1 c1), and its backward runs the other way;
  * completeness: every rank has each (micro-batch, chunk) exactly once per forward and backward.

Usage: check_vpp_trace.py <layers> <pp> <vpp> <trace.json>...
"""

import json
import re
import sys
from collections import defaultdict

NAME = re.compile(r"^(F|B)(\d+) chunk (\d+)$")
RANK = re.compile(r"pp(\d+) dp(\d+) tp(\d+) · rank (\d+)")
TOLERANCE_US = 300


def chunk_layers(layers, pp_size, vpp, pp, chunk):
    per_chunk = layers // (pp_size * vpp)
    first = chunk * (layers // vpp) + pp * per_chunk + 1
    return range(first, first + per_chunk)


def check(path, layers, pp_size, vpp):
    events = json.load(open(path))["traceEvents"]
    ranks = {}
    for e in events:
        if e["ph"] == "M" and e["name"] == "process_name":
            m = RANK.search(e["args"]["name"])
            if m:
                ranks[e["pid"]] = tuple(int(x) for x in m.groups())
    steps = defaultdict(dict)  # (dp, kind) -> {(pp, mb, chunk): (start, end)}
    out = {"ranks": len(ranks), "fb": 0, "modules": 0, "unnamed_fb": 0, "no_identity_modules": 0,
           "layer_mismatch": 0, "token_mismatch": 0, "incomplete": 0, "duplicates": 0}
    for pid, (pp, dp, tp, rank) in ranks.items():
        slices = [e for e in events if e["ph"] == "X" and e["pid"] == pid]
        seen = defaultdict(list)
        tokens = defaultdict(set)
        for e in slices:
            name, args = e["name"], e.get("args") or {}
            m = NAME.match(name)
            if m:
                kind, mb, chunk = m.group(1), int(m.group(2)), int(m.group(3))
                seen[kind, mb, chunk].append(e)
                out["fb"] += 1
                if kind == "F" and "tokens" in args:
                    tokens[mb].add(args["tokens"])
                if tp == 0:
                    steps[dp, kind][pp, mb, chunk] = (e["ts"], e["ts"] + e["dur"])
            elif re.match(r"^(F|B)\d*$", name):
                out["unnamed_fb"] += 1
            elif "layer" in args:
                out["modules"] += 1
                if "model_chunk" not in args or "micro_batch" not in args:
                    out["no_identity_modules"] += 1
                elif args["layer"] not in chunk_layers(layers, pp_size, vpp, pp, args["model_chunk"]):
                    out["layer_mismatch"] += 1
        out["duplicates"] += sum(len(v) > 1 for v in seen.values())
        num_mb = 1 + max((mb for _, mb, _ in seen), default=-1)
        expected = {(k, mb, c) for k in "FB" for mb in range(num_mb) for c in range(vpp)}
        out["incomplete"] += len(expected ^ set(seen))
        out["token_mismatch"] += sum(len(v) > 1 for v in tokens.values())
        out.setdefault("micro_batches", set()).add(num_mb)
    # Cross-stage order within each DP replica (TP rank 0 only).
    hops_f = [(p, c) for c in range(vpp) for p in range(pp_size)]
    worst, violations, pairs = float("-inf"), 0, 0
    for (dp, kind), table in steps.items():
        hops = hops_f if kind == "F" else hops_f[::-1]
        for mb in {mb for _, mb, _ in table}:
            for (p0, c0), (p1, c1) in zip(hops, hops[1:]):
                a, b = table.get((p0, mb, c0)), table.get((p1, mb, c1))
                if a is None or b is None:
                    continue
                pairs += 1
                late = a[1] - b[0]
                worst = max(worst, late)
                violations += late > TOLERANCE_US
    # Largest (end of the earlier hop - start of the later one); negative: the later hop started after.
    out.update(order_pairs=pairs, order_violations=violations, order_worst_lateness_us=round(worst, 1))
    out["micro_batches"] = sorted(out["micro_batches"])
    return out


if __name__ == "__main__":
    layers, pp_size, vpp = (int(x) for x in sys.argv[1:4])
    for path in sys.argv[4:]:
        print(path.rsplit("/", 1)[-1], json.dumps(check(path, layers, pp_size, vpp)))
