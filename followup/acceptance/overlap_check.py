#!/usr/bin/env python3
"""Compare comm/compute overlap, synchronizing calls and NCCL kernels between
profiler-off and profiler-on Kineto traces.

The straggler profiler claims it leaves overlap alone: it only records CUDA events, never synchronizes the device
and adds no NCCL collective (the window gather is Gloo, on the CPU). This checks all three on real traces:

* overlap ratio = time NCCL kernels run while a compute kernel runs on the same device / total NCCL kernel time,
  both as unions of intervals, per trace file (one file = one rank);
* host-side synchronizing CUDA runtime calls (``cuda*Synchronize``), per trace;
* NCCL kernels per trace.

Usage:
    python overlap_check.py --off OFF_DIR --on ON_DIR [--off2 OFF2_DIR] [--tolerance 0.01] [--out verdict.json]

Each DIR holds the ``*.pt.trace.json[.gz]`` files torch.profiler wrote (searched recursively). The overlap
tolerance must be fixed before looking at ON: pass ``--tolerance``, or pass ``--off2`` (a second profiler-off run
with the same seed) and the tolerance becomes 2 x |mean(off) - mean(off2)|, never below ``--min-tolerance``.
Sync and NCCL counts may not grow beyond the off/off2 difference (zero without ``--off2``).

Exit code: 0 PASS, 1 NOT_PASS, 2 no verdict (no traces, unreadable input, or no tolerance).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from statistics import mean


def load_trace(path: Path) -> dict:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def find_traces(directory: Path) -> list[Path]:
    return sorted(p for p in Path(directory).rglob("*") if p.name.endswith((".pt.trace.json", ".pt.trace.json.gz")))


def union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def length(merged: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in merged)


def intersection_length(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Overlap of two already-merged interval lists (two-pointer sweep)."""
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def trace_stats(trace: dict) -> dict:
    """Overlap, sync and NCCL numbers for one trace (all devices in it
    summed)."""
    comm: dict[object, list[tuple[float, float]]] = {}
    compute: dict[object, list[tuple[float, float]]] = {}
    nccl_kernels = 0
    syncs: dict[str, int] = {}
    event_records = event_queries = 0
    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        category = str(event.get("cat", "")).lower()
        name = str(event.get("name", ""))
        if category == "kernel":
            device = (event.get("args") or {}).get("device")
            span = (float(event["ts"]), float(event["ts"]) + float(event.get("dur", 0.0)))
            if "nccl" in name.lower():
                nccl_kernels += 1
                comm.setdefault(device, []).append(span)
            else:
                compute.setdefault(device, []).append(span)
        elif category in ("cuda_runtime", "cuda_driver"):
            if "Synchronize" in name:
                syncs[name] = syncs.get(name, 0) + 1
            elif name in ("cudaEventRecord", "cudaEventRecordWithFlags"):  # torch 2.13 timing events use WithFlags
                event_records += 1
            elif name == "cudaEventQuery":
                event_queries += 1
    comm_us = overlap_us = 0.0
    for device, spans in comm.items():
        merged = union(spans)
        comm_us += length(merged)
        overlap_us += intersection_length(merged, union(compute.get(device, [])))
    return {
        "comm_us": comm_us,
        "overlap_us": overlap_us,
        "overlap_ratio": overlap_us / comm_us if comm_us > 0 else 0.0,
        "nccl_kernels": nccl_kernels,
        "syncs": syncs,
        "event_records": event_records,
        "event_queries": event_queries,
    }


def group_stats(paths: list[Path]) -> dict:
    """Per-file stats and their means for one arm (off / on / off2)."""
    if not paths:
        raise ValueError("no *.pt.trace.json[.gz] files found")
    files = {str(path): trace_stats(load_trace(path)) for path in paths}
    stats = list(files.values())
    sync_names = sorted({name for s in stats for name in s["syncs"]})
    return {
        "files": files,
        "n": len(stats),
        "overlap_ratio": mean(s["overlap_ratio"] for s in stats),
        "nccl_kernels": mean(s["nccl_kernels"] for s in stats),
        "syncs": {name: mean(s["syncs"].get(name, 0) for s in stats) for name in sync_names},
        "event_records": mean(s["event_records"] for s in stats),
        "event_queries": mean(s["event_queries"] for s in stats),
    }


def compare(
    off: dict, on: dict, off2: dict | None = None, tolerance: float | None = None, min_tolerance: float = 0.002
) -> dict:
    """Verdict for ON against OFF (and the OFF/OFF2 noise when given)."""
    if tolerance is not None:
        tol, source = tolerance, "given"
    elif off2 is not None:
        tol, source = max(2.0 * abs(off["overlap_ratio"] - off2["overlap_ratio"]), min_tolerance), "2 x |off - off2|"
    else:
        tol, source = None, "none"
    drop = off["overlap_ratio"] - on["overlap_ratio"]
    overlap = {
        "off": off["overlap_ratio"],
        "on": on["overlap_ratio"],
        "off2": off2["overlap_ratio"] if off2 else None,
        "drop": drop,
        "tolerance": tol,
        "tolerance_source": source,
        "pass": tol is not None and drop <= tol,
    }

    names = sorted(set(off["syncs"]) | set(on["syncs"]) | set(off2["syncs"] if off2 else ()))
    sync_rows = {}
    for name in names:
        slack = abs(off["syncs"].get(name, 0) - off2["syncs"].get(name, 0)) if off2 else 0.0
        sync_rows[name] = {
            "off": off["syncs"].get(name, 0),
            "on": on["syncs"].get(name, 0),
            "slack": slack,
            "pass": on["syncs"].get(name, 0) <= off["syncs"].get(name, 0) + slack,
        }
    syncs = {"per_call": sync_rows, "pass": all(row["pass"] for row in sync_rows.values())}

    nccl_slack = abs(off["nccl_kernels"] - off2["nccl_kernels"]) if off2 else 0.0
    nccl = {
        "off": off["nccl_kernels"],
        "on": on["nccl_kernels"],
        "slack": nccl_slack,
        "pass": abs(on["nccl_kernels"] - off["nccl_kernels"]) <= nccl_slack,
    }

    if tol is None:
        verdict = "NO_VERDICT"
    else:
        verdict = "PASS" if overlap["pass"] and syncs["pass"] and nccl["pass"] else "NOT_PASS"
    return {
        "verdict": verdict,
        "overlap": overlap,
        "syncs": syncs,
        "nccl_kernels": nccl,
        "event_calls": {arm: {"records": g["event_records"], "queries": g["event_queries"]} for arm, g in
                        (("off", off), ("on", on))},
        "n_traces": {"off": off["n"], "on": on["n"], "off2": off2["n"] if off2 else 0},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--off2", type=Path)
    parser.add_argument("--tolerance", type=float, help="absolute overlap-ratio drop allowed (fix it before ON)")
    parser.add_argument("--min-tolerance", type=float, default=0.002)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        off = group_stats(find_traces(args.off))
        on = group_stats(find_traces(args.on))
        off2 = group_stats(find_traces(args.off2)) if args.off2 else None
    except (OSError, ValueError) as exc:
        print(f"overlap_check: {exc}", file=sys.stderr)
        return 2
    verdict = compare(off, on, off2, args.tolerance, args.min_tolerance)
    text = json.dumps(verdict, indent=2)
    if args.out:
        args.out.write_text(text)
    o = verdict["overlap"]
    print(
        f"{verdict['verdict']}: overlap off {o['off']:.6f} on {o['on']:.6f} drop {o['drop']:+.6f} "
        f"(tolerance {o['tolerance']} from {o['tolerance_source']}); syncs pass={verdict['syncs']['pass']}; "
        f"nccl kernels off {verdict['nccl_kernels']['off']:.1f} on {verdict['nccl_kernels']['on']:.1f}"
    )
    return {"PASS": 0, "NOT_PASS": 1}.get(verdict["verdict"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
