# Straggler Analysis

The Megatron backend ships a low-overhead straggler profiler that can stay enabled in production runs. Every training rank times its forward, backward, optimizer, gradient-sync and parameter all-gather segments with CUDA events on the compute stream; host-side launch gaps between the two events count as well. Every K rollouts, all ranks Gloo-all-gather one fixed-length statistics vector to the primary rank, where a background thread decides whether a rank is holding the group back and why, and writes the verdict to metrics, the timeline and the log.

This is not `torch.profiler`: it traces no kernels, never calls `cuda.synchronize()`, and does not change compute/communication overlap. No exception inside the profiler reaches the training loop: errors are counted, and repeated errors switch the profiler off on every rank together.

## Enabling

The profiler is off by default. Pass the environment variables to the training actors through the Ray runtime environment:

```yaml
# configs/env.yaml
env_vars:
  RELAX_STRAGGLER_PROFILER: "1"
  RELAX_STRAGGLER_REPORT_INTERVAL: "10"
```

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RELAX_STRAGGLER_PROFILER` | bool | false | Master switch. |
| `RELAX_STRAGGLER_REPORT_INTERVAL` | int | 10 | Rollouts per cross-rank gather and analysis; this span is called a window. A rollout may contain several optimizer steps; ms/step values below are per rollout. |
| `RELAX_STRAGGLER_Z_THRESHOLD` | float | 3.0 | Robust z-score (median / MAD within the peer group) a rank must exceed to count as abnormal. |
| `RELAX_STRAGGLER_REL_THRESHOLD` | float | 0.10 | Minimum relative excess over the peer median. |
| `RELAX_STRAGGLER_PERSIST_WINDOWS` | int | 3 | Consecutive abnormal windows before a rank is flagged, for every reason; filters one-off spikes (checkpointing, GC). Must be at least 1. |
| `RELAX_STRAGGLER_RECOVER_WINDOWS` | int | 2 | Consecutive clean windows before an alert clears, so alerts do not flap; uncertain windows are not clean. Must be at least 1. |
| `RELAX_STRAGGLER_GATHER_TIMEOUT_S` | float | 60.0 | Timeout in seconds of the profiler's dedicated Gloo group. A gather normally takes a few milliseconds; this only bounds how long the other ranks wait once a rank process has disappeared. |
| `RELAX_STRAGGLER_TRACE_DIR` | str | empty | Output directory of the sampled-step trace; defaults to `straggler/` under `--timeline-dump-dir`. Without either, no trace is written. |
| `RELAX_STRAGGLER_TRACE_KEEP` | int | 50 | Number of newest trace files to keep. |
| `RELAX_STRAGGLER_LEVEL` | int | −1 | Start level (see *Switching the level while training*); −1 means 2 with a trace directory, 1 otherwise. |
| `RELAX_STRAGGLER_CONTROL_POLL_S` | float | 5.0 | How often, in seconds, the primary rank asks the metrics service for a level. |

Each actor process logs `Straggler profiler enabled: report_interval=… primary=… level=… control=… trace_dir=… meta=RankMeta(rank=…, dp=…, tp=…, pp=…, cp=…, ep=…, host=…, device=…, primary=…)` at start-up; `trace_dir=-` means no trace is written. Critic, reference and `actor_fwd` actors run no training steps and are not profiled.

## What you get

### Log (primary rank)

The primary rank's analysis thread logs as soon as a window's analysis ends; `step` is the step of the window's last rollout. A window without alerts logs one INFO line:

```text
[straggler] step=29 no straggler (uncertain 0); self median 734.6 ms max 745.6 ms (rank 0, +2%),
latest to grad-sync rank 0 by 13.4 ms (peers idle 45.9 ms), pp_stage_imbalance 1.00, overhead 0.10 ms/step
```

When there are alerts, the window in which a rank's reason changes logs one WARNING, and every later window that still has alerts logs an INFO table with one row per rank. Every reason needs `PERSIST_WINDOWS` consecutive windows to be raised and `RECOVER_WINDOWS` consecutive clean windows to clear; clearing logs another WARNING, such as `rank 5 (…) recovered from slow_device after 2 clean windows`:

```text
[straggler] step=29 rank 3 (rank3_dp3_tp0_pp0, host=…, gpu=3) reaches the DP grad-sync 197.4 ms/step
after its DP peers (peers idle 203.3 ms/step = 22% of their compute, z=41.0) for 3 windows;
its own GPU forward is 0.99x peers, gc 1.8 ms, cpu/gpu fwd 1.56x -> late_arrival (host-side stall)
[straggler] step=29 per-rank window (ms/step):
 rank | tag               | host | device | self_ms | wait_ms | late_ms | tokens  | ms_per_ktok | gc_ms | steady_ms | mb_fwd_max | reason       | uncertain
    0 | rank0_dp0_tp0_pp0 | …    |      0 |   933.7 |     6.4 |     6.1 | 14802.1 |        63.1 |   1.8 |       0.0 |      157.8 | none         |
    3 | rank3_dp3_tp0_pp0 | …    |      3 |   737.1 |     5.9 |   197.4 | 14592.7 |        50.5 |   1.8 |       0.0 |      148.0 | late_arrival |
```

Only ranks 0 and 3 are shown; the real table has a row per rank. `steady_ms` is the steady-state PP wait (`pp_wait_up + pp_wait_down`, 0 here because this run has no PP) and `mb_fwd_max` the slowest single forward of the window (ms).

A rank whose data for the window cannot support a verdict is reported as uncertain, with a cause, instead of passing as healthy; a change of cause logs one WARNING:

```text
[straggler] step=19 rank 3 (rank3_dp3_tp0_pp0) is uncertain: 2 event pairs were still in flight when the window
closed; their time lands in the next window
```

| Cause | Meaning |
|-------|---------|
| `dropped_events` / `unread_events` / `profiler_errors` | That rank's timing for the window is incomplete: spans were dropped because the queue was full, were not yet readable when the window closed (they count in the next window), or the profiler caught errors on that rank. An unread `dp_grad_sync` span would make the rank look late, so no verdict is drawn. |
| `no_peers` | Its PP stage has no other rank to compare with. |
| `missing_tokens` | It looks slow, but its stage has no token counts, so data imbalance cannot be told apart from a slow device. |
| `no_wait_reference` | It waits on PP, but every other rank of its stage is flagged, late or uncertain, so there is no wait time to compare with. |

An uncertain window breaks the run of abnormal windows and does not count toward clearing an alert; an existing alert stays as it is.

### Metrics

These scalars use the same step as `perf/*` and go to TensorBoard / WandB / ClearML. They are sent in the same `tracking_utils.log` call as the `perf/*` metrics of the training thread's next `log_perf_data`, never in a request of their own (with `--use-metrics-service` every log call is a synchronous HTTP request on the training thread). Because the analysis runs on a background thread, a window's scalars normally go out with the next rollout's `perf/*`; `straggler/window/{first,last}_rollout` name the window they belong to. The last rollout of a run waits for the last window's analysis before logging.

If the analysis falls behind, several windows can finish within one step (this does not happen in normal runs). Each of them still reaches the timeline and the log, but one log call carries one value per key, so only the scalars of the latest window that reported successfully reach the metrics. Each window it replaces logs a WARNING `… windows ready at once; only rollouts …-… reach the metrics` and counts as one profiler error; a window whose reporting fails counts once, for that failure (see *When the profiler itself fails*).

| Metric | Meaning |
|--------|---------|
| `straggler/{fwd,bwd,optim,pp_recv,pp_send,dp_grad_sync,dp_param_gather,lp_fwd,lp_pp_recv,lp_pp_send}/{median_ms,max_ms,max_rank,spread}` | Per-segment GPU time (ms/step): median and maximum across all ranks; `max_rank` / `spread` are the rank with the largest excess over *the other ranks of its PP stage* and that excess (`value/peer_median − 1`). `lp_*` are the same segments during log-prob / forward-only passes. `dp_param_gather` only has values with `--overlap-param-gather` off (Megatron does not time the overlapped path). |
| `straggler/self/*`, `straggler/self_per_ktok/*` | Own compute time (`fwd + bwd + optim`) and the same per 1000 tokens; the latter is omitted when any rank reports zero tokens. |
| `straggler/wait/{median_ms,max_ms,max_rank,spread}` | Time spent waiting for other ranks (`pp_recv + dp_grad_sync + dp_param_gather`). |
| `straggler/late/max_ms`, `straggler/late/max_rank`, `straggler/late/peer_idle_ms` | The rank that reaches the DP gradient sync last, by how much (compared on `bwd + dp_grad_sync`, see `late_arrival` below), and how long the rest of its DP group idles for it per step. |
| `straggler/tokens/{median,max,spread}` | Token-count imbalance: `median` / `max` across all ranks, `spread` the largest excess within a stage; omitted when any rank reports zero tokens. |
| `straggler/pp_stage_imbalance` | `max/min` of the per-stage median compute time; independent of any slow device. |
| `straggler/{pp_wait_fill,pp_wait_up,pp_wait_down,pp_wait_drain}/{max_ms,max_rank}` | The training-phase `pp_recv` split by non-interleaved 1F1B phase (ms/step): the largest value and its rank. `fill` waits for upstream activations while the pipeline fills, `up` / `down` wait for upstream activations / downstream gradients in the steady state, and `drain` waits for downstream gradients while it drains. A balanced pipeline does not idle in the steady state, so `up + down` beyond the transfer time is extra bubble. When an upstream rank stalls at the start of the step, the whole pipeline is pushed back: this shows up in the downstream stages' `fill`, not in the steady state. Only with PP; the transfers of the interleaved (VPP) schedule count in `pp_recv` only. |
| `straggler/pp_bubble/ratio` | `pp_recv` as a share of the rank's timed training time (own compute + waits + `pp_send`); the largest per-stage median. Only with PP. |
| `straggler/micro_batch/{fwd_max_ms,fwd_max_rank,fwd_argmax,fwd_argmax_tokens,fwd_min_ms,bwd_max_ms,bwd_max_rank,tokens_max,tokens_min}` | Extremes of single training micro-batches over the window (not per step): the slowest forward and its rank, which forward of its rollout it was (from 0) and its token count (−1 when not reported), the fastest forward, the slowest backward and its rank, and the largest / smallest token count of a micro-batch (omitted without token counts). |
| `straggler/gc/{median_ms,max_ms}` | Python GC pauses. |
| `straggler/flagged/count`, `straggler/flagged/rank` (−1 if none), `straggler/flagged/reason` | Alert state. Reason codes: 0 none, 1 slow_device, 2 data_imbalance, 3 upstream_wait, 4 cpu_bound, 5 late_arrival. |
| `straggler/waiting/count` | Ranks waiting on a slow upstream PP stage in this window. They are held up, not slow themselves, and do not count in `flagged`. |
| `straggler/uncertain/count`, `straggler/uncertain/rank` (−1 if none), `straggler/uncertain/cause` | Number of uncertain ranks, the first one and its cause. Cause codes: 0 none, 1 dropped_events, 2 unread_events, 3 profiler_errors, 4 no_peers, 5 missing_tokens, 6 no_wait_reference. |
| `straggler/recovered/count` | Ranks whose alert cleared in this window. |
| `straggler/window/first_rollout`, `straggler/window/last_rollout` | The window these scalars belong to. |
| `straggler/control/level` | The window's level: 0 paused, 1 coarse, 2 detail, 3 modules (see *Switching the level while training*). |
| `straggler/latency/analyzed_ms` | Milliseconds from the window close (before the gather) to the end of its analysis, when the alert lines are logged. |
| `straggler/health/state`, `straggler/health/errors`, `straggler/health/degraded_ranks` | The profiler's own health: the worst state across ranks (0 active, 1 degraded, 2 disabled), errors caught in the window, and ranks currently degraded. |
| `straggler/self_overhead_ms`, `straggler/gather_ms`, `straggler/analyze_ms`, `straggler/dropped_events`, `straggler/unread_events` | The profiler's own cost: CPU time per step spent reading the timings; gather time per window on the primary rank (the first window also includes one metadata gather); analysis time per window on the background thread; spans dropped because the queue was full; spans still unread when the window closed (the last two are normally 0). |

### Timeline

With `--timeline-dump-dir` set (the timeline is written through the metrics-service adapter, so `--use-metrics-service` must be on as well), every window appends one `straggler/<seg> rank{g}_dp{d}_tp{t}_pp{p}` event per rank and non-zero segment (`pid` = 2³⁰ + global rank, above any real pid; `tid` = segment index). In Perfetto that is one row per rank, so a longer `bwd` or a shorter `dp_grad_sync` is visible at a glance.

### Sampled-step trace (Perfetto)

With a trace directory (see `RELAX_STRAGGLER_TRACE_DIR` under *Enabling*), the first rollout of every window after the first, the sampled step, is recorded on a real time axis: the step starts with one device synchronize and an anchor event, and every timed span in it keeps its start and end relative to that anchor. When the window closes, every rank sends its sampled step to the primary rank (a `gather_object` to the primary only), which writes `straggler_trace_rollout{N}.json` from its background thread. The file is in Chrome trace format; open it in [ui.perfetto.dev](https://ui.perfetto.dev). The metrics service is not needed.

Each rank is a row, ordered by (dp, pp, tp) so the stages of a pipeline sit next to each other. The row name carries the rank's measured PP wait and the bubble a balanced 1F1B pipeline would have with the rank's own measured forward and backward (`(PP − 1) × (mean fwd + mean bwd)` × optimizer steps); the difference is extra bubble. On the slowest stage the estimate exceeds the measurement (a negative difference); look at the other stages there. Within a row, slices nest by time:

| Level | Content |
| --- | --- |
| 1 (PP) | `busy`: back-to-back compute and communication; `bubble fill` / `up` / `down` / `drain`: PP waits named by 1F1B phase, with `waits_for` saying whether the previous or the next stage is awaited; `untimed`: more than 0.5 ms covered by no timed span (host-side stalls) |
| 2 (fwd / bwd) | Inside `busy`: `F{i}` / `B{i}` (`i` is the micro-batch number; forwards carry their token count), `send`, `grad sync`, `param gather`, `optimizer`; log-prob passes show `lp F{i}` and so on |
| 3 (modules) | Only at the `modules` level. Inside `F{i}` / `B{i}`: the forward and backward of the attention and of the MLP (`moe` for a MoE layer) of every transformer layer, e.g. `attn fwd L3`, `moe bwd L3`, with `layer` and `micro_batch` in the arguments; a forward inside `B{i}` is an activation recompute and is named `… (recompute)`; log-prob passes show `lp attn fwd L3` and so on |

The primary rank's own CPU-side timers (`Timer` events such as `actor_train` and the per-micro-batch `get_data_batch_*`, without call stacks) get a separate `primary CPU` row; timers still running when the sampled step ends are left out.

| Metric | Meaning |
|--------|---------|
| `straggler/trace/ranks`, `straggler/trace/lost` | Ranks that delivered a trace; timed spans lost (dropped from a full queue, or beyond the per-step cap of 4096 spans plus 16384 module spans) |
| `straggler/trace/module_brackets` | Module spans in the trace, summed over ranks (`modules` level only) |
| `straggler/trace/align_spread_ms` | The largest difference between the wall-clock end of the same gradient sync on the ranks of a DP group, i.e. the cross-rank alignment error |
| `straggler/trace/bubble_excess_ms_max`, `straggler/trace/bubble_excess_rank` | The largest excess of measured PP wait over the balanced 1F1B bubble, and its rank |

Cost and limits:

- One extra device synchronize per window (at the start of the sampled step) and one gather to the primary rank only (24 B per timed span); building and writing the file happen on the primary's background thread. Measured on 8×A100 with Qwen3-0.6B TP1×PP4×DP2 (step ≈ 1.6 s): about 2 ms more per window on the training thread (synchronize ≈ 0.4 ms, gather ≈ 1.3 ms, reading the time offsets ≈ 0.3 ms), about 0.01 % per step with the default 10-rollout window.
- The `modules` level adds hooks only during the sampled step, on every rank alike. Measured on the same setup (7 layers per stage, about 250 module spans per rank and step): that step's forwards are 3–10 % longer (on a small, launch-bound model the hooks' CPU time shows), backwards are unchanged, and the step time stays within the step-to-step noise; reading the module spans adds about 3.5 ms per window. The window statistics include those slightly longer forwards. Nothing is timed under CUDA graph capture, so captured layers have no module slices.
- Ranks are aligned by their wall clocks (`time.time()`), so across machines NTP accuracy applies; `align_spread_ms` shows the error.
- Phase names and the bubble estimate assume non-interleaved 1F1B; with VPP or `combined_1f1b`, read level 2 only. With one-rollout windows the sampled step may not be fully read when the window closes, and that window has no trace.

### Switching the level while training

| Level | Meaning |
| --- | --- |
| 0 `paused` | Nothing is timed; windows still gather (status only) but are not analyzed, alert counts are frozen, and the only scalar is `straggler/control/level` |
| 1 `coarse` | The window statistics and verdicts above |
| 2 `detail` | Plus the sampled-step trace; needs a trace directory and runs as `coarse` without one |
| 3 `modules` | Plus level-3 slices in the trace. The hooks exist only while the sampled step runs, so only that step pays for them; module spans go into the trace only, never into the window statistics or the verdicts. Needs a trace directory like `detail`, and is never the default |

`RELAX_STRAGGLER_LEVEL` sets the start level: by default 2 with a trace directory and 1 otherwise; 3 only when set explicitly. With `--use-metrics-service`, a daemon thread on the primary rank asks the metrics service for a level every `RELAX_STRAGGLER_CONTROL_POLL_S` seconds. While training runs, open `http://<head node>:8000/metrics/straggler` in a browser and press a button, or send the request directly:

```bash
curl -X POST http://<head>:8000/metrics/straggler/control -H 'Content-Type: application/json' -d '{"level": 0}'
```

The requested level goes into the `control` field of the primary rank's statistics vector and reaches every rank with the next window gather; every rank switches from the next window on, so a change takes effect at most one window plus one poll interval later. The page shows the level in effect and when the primary last polled. Once the profiler has switched itself off after errors (see *When the profiler itself fails*) it ignores level requests; without the metrics service the level stays at its start value.

The training processes listen on no port: the page and the endpoints are served by the Ray Serve HTTP server that Relax starts (head node, port 8000), which the primary rank polls. The endpoints have no authentication, so do not expose them outside the cluster; to open the page in a local browser, forward port 8000 yourself.

## Coverage

The timings reuse the 22 timer call sites Megatron already has, which cover these segments:

| Segment | Detail |
| --- | --- |
| `fwd` / `bwd` | One micro-batch's forward / backward (timed as a whole, not per layer) |
| `pp_recv` / `pp_send` | PP point-to-point (9 Megatron timer names folded into 2 segments; which pair of ranks is not distinguished; the 4 receive timers of non-interleaved 1F1B training also count by phase and direction in `pp_wait_*`) |
| `dp_grad_sync` | DP gradient sync (4 timer names folded into 1 segment; the CP and DP portions cannot be separated) |
| `dp_param_gather` | The distributed optimizer's parameter all-gather |
| `optim` | The optimizer phases (6 timer names) |

**Not covered** (Megatron has no timer call site for them):

- **The internals of attention, MoE, head/logits and MTP.** In the window statistics they are part of `fwd` / `bwd`. The `modules` level times each layer's attention and MLP / MoE block as a whole, in the sampled-step trace only; what happens inside a block (MoE dispatch / experts / combine, for example) and the output layer are still not split out.
- **EP all-to-all** (expert-parallel communication in MoE). It is time spent waiting for other ranks, but it counts as compute.
- **The real compute/communication overlap ratio.** A smaller `dp_grad_sync` does **not** mean better overlap: it is the sum of the waiting segments and only reflects overlap indirectly.

**Known granularity limits** (see *Reading the reason*): ranks of the same TP group wait for each other inside forward and backward, so when one of them slows down the whole group is reported and the slow one cannot be singled out; when every rank slows down at once (whole-machine clock throttling, a shared-storage stall) nothing is detected.

## Reading the reason

| reason | Rule | Where to look |
|--------|------|---------------|
| `slow_device` | Own compute time or forward time well above the other ranks of the same PP stage, token count normal | `nvidia-smi -q -d CLOCK,PERFORMANCE,TEMPERATURE`, ECC, neighbours on the node, NVLink topology |
| `data_imbalance` | Own compute or forward time high but normal per token; token count clearly higher | `--balance-data`, dynamic batch splitting, oversize samples |
| `cpu_bound` | Own compute or forward time high *and* Python GC pauses during the step ≥ 5 % of the own compute time | `gc.freeze()`, data threads contending for the GIL, per-token Python loops |
| `upstream_wait` | Own compute time normal, but `pp_recv` well above the same-stage ranks that are neither flagged nor late, by at least 5 % of the stage's median own compute time; a rank that reaches the gradient sync late because it waits on PP is classed here too | The flagged rank on the upstream stage, or `pp_stage_imbalance` (uneven layer split) |
| `late_arrival` | Own GPU time normal, but its `bwd + dp_grad_sync` is much *shorter* than that of the rest of its DP group: a collective finishes for everyone at once, so the last rank to arrive waits least, and every other rank's time includes the wait for it | Host-side gaps between kernels: data fetch, synchronous I/O / HTTP, GIL, launch gaps. `py-spy dump --pid <pid>` on that rank is the quickest confirmation |

`late_arrival` is easy to miss, because a rule that only looks at GPU time cannot see it: the rank's GPU time matches the others', it just reaches every gradient sync late. A common cause is a blocking call on the training thread, such as the primary rank posting metrics over synchronous HTTP.

With `--overlap-grad-reduce` the gradient reduce-scatter is issued bucket by bucket during backward, so the wait for the slowest rank lands in the other DP ranks' `bwd` and only a short tail is left in `dp_grad_sync`. That is why late arrival is compared on `bwd + dp_grad_sync`, which ends when the gradient sync completes with or without the overlap. For the same reason slow ranks are also compared on forward alone, which has no DP collective. The waiting ranks' own compute time grows too, so a `late_arrival` alert quotes the rank's forward against the others', not its own compute time.

## When the profiler itself fails

Exceptions in the timer calls, reading the timings, the window gather, the analysis and the log callback are caught and counted; none reaches the training loop. A rank that has caught 3 errors, or has had dropped spans or errors in 3 consecutive windows, asks for the profiler to be switched off. The request travels in the `health` field of that rank's statistics vector, so every rank sees the same gathered table and switches off in the same window; no rank leaves on its own and leaves the others stuck in the next gather. Once off, the timer calls are skipped, but the rank **keeps joining every window gather**: it sends an all-zero vector of `NUM_FIELDS` values with only `health` set to 2, so the collective keeps its shape and the off state keeps reaching ranks that have not seen it yet.

A failed gather is only visible to the rank it failed on. That rank does **not** leave the collective: the failure is counted, the window is reset as usual, and the next window sends its statistics vector (or the all-zero vector) as usual. After repeated failures the rank switches off locally and starts sending the all-zero vector; the other ranks read `health = 2` in the next gather that completes and switch off in the same window.

The metadata gather of the first window runs once and is never retried: once the other ranks have the metadata they only gather statistics vectors, so a retry on the failed rank alone would misalign the collectives. Instead, that rank sends its statistics vector for the window as usual with `health = 2`, and every rank switches off in that window. The metadata gather is a single collective in which every rank sends one fixed 1 KiB byte string. A rank that fails to serialize its metadata, or has too much of it, still joins and sends an empty entry; every rank then sees the same missing entry and switches off in that window as well.

So that the gathers end even when a rank process is gone and can send nothing, the profiler creates a dedicated Gloo group for them at install time, with a timeout set by `RELAX_STRAGGLER_GATHER_TIMEOUT_S`; this does not change the timeout of the shared Gloo group or of the training path. **Known limitation:** in that case a single gather still blocks until the timeout, and if the whole group is gone every rank can only switch off locally, without the off state spreading between ranks.

## Overhead

- One pair of non-blocking `event.record()` calls per Megatron timer call site (events are pooled). At the end of each step completed events are read with `event.query()`, never `cuda.synchronize()`; the cost is reported as `straggler/self_overhead_ms`.
- One Gloo `all_gather` every `REPORT_INTERVAL` rollouts, reported as `straggler/gather_ms`. Each rank sends 272 B per window (34 float64), independent of the micro-batch count; because it is an `all_gather`, every rank receives the full `N × 272 B` table, and the whole job receives `N × (N−1) × 272 B` per window (excluding Gloo protocol overhead). The first window also has one metadata gather of 1 KiB per rank.
- The analysis runs on a background thread of the primary rank, reported as `straggler/analyze_ms`. It is O(n log n) in the number of ranks per stage: a few milliseconds per window at 8 ranks, about 35 ms at 2048 ranks in a CPU benchmark.
- On 8×A100 with Qwen3-0.6B DP8 (step ≈ 1 s), the summed worst-case upper bound is about 1.9 ms/step (≈ 0.19 %), and end-to-end on/off comparisons show no measurable difference; the profiler adds no synchronizing calls and leaves the overlap of NCCL and compute kernels unchanged. The method and all measurements are in [RFC #362](https://github.com/redai-studio/Relax/issues/362).
- The cost of the sampled-step trace and the `modules` level is under *Sampled-step trace*.

## Alert latency

Detection runs only when a window closes, and an alert needs `PERSIST_WINDOWS` consecutive abnormal windows. With the defaults (`REPORT_INTERVAL=10`, `PERSIST_WINDOWS=3`) an alert appears about 21–30 rollouts after a rank starts slowing down; if the slowdown starts late in a window, the window total may stay below the thresholds, which adds one more window. Lowering either setting alerts sooner, at the cost of reacting to one-off spikes.

The alert line is logged when the analysis finishes (`straggler/latency/analyzed_ms`, typically a few milliseconds); the scalars go out with the next `perf/*` report and are shown by whichever output backend is in use (metrics service, W&B, TensorBoard or ClearML).

## Where it lives

- `relax/utils/straggler/timers.py`: the non-blocking replacement for Megatron's `config.timers`. Megatron's own `Timer.start/stop` call `cuda.synchronize()`, which is why Relax sets `config.timers` to `None` when the profiler is off.
- `relax/utils/straggler/stats.py`: the statistics vector layout and the map from Megatron timer names to segments.
- `relax/utils/straggler/collector.py`: one per rank: the event pool, lazy reading, GC callbacks and the per-window gather; torch-free, with events and gathers passed in.
- `relax/utils/straggler/worker.py`: the background thread on the primary rank that runs the analyses in order.
- `relax/utils/straggler/detector.py`: pure-numpy window analysis and verdicts (including uncertain verdicts and alert clearing), testable without a GPU.
- `relax/utils/straggler/health.py`: the profiler's own health (active → degraded → disabled).
- `relax/utils/straggler/reporter.py`: metrics / timeline output (training thread) and log output (background thread).
- `relax/utils/straggler/trace.py`: lays every rank's sampled-step timings out as nested slices and writes the Perfetto JSON (on the primary's background thread).
- `relax/utils/straggler/control.py`: switching the level while training runs: the state on the metrics-service side, the primary rank's polling thread and the control page.
- `relax/utils/straggler/modules.py`: the forward hooks and gradient hooks the `modules` level adds to every layer's attention and MLP / MoE block during the sampled step.
- `relax/utils/straggler/runtime.py`: the only module that touches the accelerator (through `relax.utils.device`, so CUDA and NPU alike), the process groups or Megatron's parallel state; `StragglerProfiler` does the per-step reporting around `log_perf_data`.
- Integration points: `relax/backends/megatron/model.py` (`config.timers = straggler_timers(...)` at three sites), `relax/backends/megatron/actor.py` (`install_straggler_profiler`, `register_straggler_model`, per-step `_log_perf_data`) and `relax/utils/metrics/service.py` (the control page and the `/straggler/*` endpoints).
