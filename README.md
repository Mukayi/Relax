# Task 11 straggler profiler — evidence

Code: `Mukayi/Relax` branch `feat/task11-straggler-profiler` (PR redai-studio/Relax#363, RFC redai-studio/Relax#362).
Recipe: Qwen3-0.6B SFT DP8, OpenMathReasoning-mini, GBS 32; one rollout = one optimizer step in this recipe.
Hosts: 8×A100 (A100-1, A100-2) and 8×A800 (A800-2, A800-1). Overhead runs use the metrics service **off** unless noted.
The measurements were taken on `f203bff`; `fig_mechanism` and `fig_detector` show the current code (analysis on a background thread, uncertain verdicts, alert clearing, profiler health).

## Headline numbers

- **Overhead, in-run crossover** (profiler toggled every 10 rollouts; opposite-phase pair): metrics-service-off pairs −0.04 % / +2.04 % / −0.16 % / −0.15 % (each 28 blocks). Pair-level mean +0.42 % ± 1.72 % (n = 4). The +2 % pair on A100-1 has full-GC spikes only in on-blocks. A/A (profiler off both runs) on three hosts: +0.41 % / −0.70 % / −1.41 %. Metrics service on: +0.78 % ± 1.78 % (1 pair). Details in `crossover_summary.md`.
- **GC count** (pre-registered): turning the profiler on does **not** increase generation-2 GC count (11/12 runs gen2 = 1; the only gen2 = 2 is profiler-off). See `gc_counts.md`.
- **Overhead, separate runs**: 3 off/on pairs × 60 steps differ by −1.69 % / −0.54 % / +0.46 %; the off runs alone differ by 1.2 %. Run-level 95 % CI −3.3 % to +2.1 % (n = 3).
- **Overhead, summed upper bound** (8 ranks, critical-path assumption): typical 1.2 ms/step (≈ 0.10 %), worst 1.9 ms/step (≈ 0.17 %). See `bench_timer_cost_a100.txt`.
- **Loss**: max relative |Δ| off↔on 0.090 %, off↔off 0.094 % (same seed).
- **False alarms**: 0 WARNINGs in 12 reportable windows (3 clean profiler-on runs × steps 29–59).
- **Injection** (bf16 matmul hog on GPU 5, 2.5×): step 29 — WARNING rank 5 → `slow_device`.
- **Real CPU-side slow rank** (`fig_real_case`): A800-1, earlier prototype, report interval 5, metrics service on.

## Files

| File | Content |
| --- | --- |
| `crossover_summary.md`, `crossover_blocks.csv`, `crossover_all_blocks.csv` | All crossover / A/A pairs; figure uses the 4 metrics-off pairs |
| `gc_counts.md` | Generation-2 GC counts on vs off (4 hosts × 3 runs) |
| `bench_timer_cost_a100.txt` | Single-GPU hot-path microbenchmark output |
| `steps.csv`, `summary.md` | Separate-run A/B: per-step `step_time` / `loss` for 3 off/on pairs |
| `figures/fig_mechanism` | End-to-end mechanism (hot path orange, cold path blue, analysis thread on the primary rank) |
| `figures/fig_detector` | Detector: peer groups, robust z, uncertain pre-check, raise after 3 / clear after 2 windows, priority ladder |
| `figures/fig_late_arrival` | Why the latest rank has the shortest `dp_grad_sync` |
| `figures/fig_injection` | Per-rank self / wait at step 29 with a matmul hog on GPU 5 |
| `figures/fig_real_case` | Real CPU-side slow rank |
| `figures/fig_overhead` | Separate-run A/B and in-run crossover (4 pairs) |
| `figures/fig_loss` | Same-seed loss off vs on |
| `figures/make_figures.py` | Regenerates every figure (PNG + SVG) |
| `scripts/` | Microbenchmark, crossover driver, holders, experiment-only toggle patch (set `WORKDIR` before running) |

## Follow-up on the final code (2026-10-23)

PR head `a829f13` plus #462 (the torch 2.13 fix) and #463 (the metrics-service fix, as in the acceptance version); Qwen3-0.6B SFT DP8, overlap on.

- **Numbers** (`followup/acceptance/numbers_*`): 40 rollouts with the same seed, every run saving its grad norms. With the profiler at its most intrusive setting (modules level, trace, a window every rollout) against a profiler-off run: grad norm max 2.39 % / median 0.27 %, loss max 0.053 % / median 0.013 %. Two profiler-off runs: 2.31 % / 0.34 % and 0.070 % / 0.018 %.
- **Numbers with reporting on** (`followup/acceptance/numbers_reporting_on_*`): the three overlap runs below, profiler at its default level, 40 rollouts: loss max 0.069 % and grad norm max 2.13 % against a profiler-off run; 0.059 % and 3.20 % between the two profiler-off runs.
- **Overhead** (`followup/perf/`), with full functionality (default level, i.e. level 2, a trace directory, the metrics service and overlap on): probes (`cost/`, on `0c9afc1`) put the profiler's direct cost on rank 0 at 15.2 ms per window (1.52 ms per rollout, about 0.15 %), about 60 % of it in the segment timers; the other ranks wait for rank 0 at the next collective. End to end (on `07934fe`, which differs from it only in the log format), an in-run crossover cannot resolve 0.5 % on this shared host. The profiler is switched on and off every 10 rollouts, the two runs of a pair in opposite order; each rollout's time is its full interval, from the end of the previous rollout to the end of its own, and the toggle switches only after it (`patches/0011`); rollouts where the two runs differ by more than 0.3 s are dropped, nearly all of them one rank's full garbage collection, whose count the profiler does not change. Pairs differ by about 0.8 %, more than their blocks do, so each pair gives one value: 4 pairs, -0.13 % (95 % CI -1.46 % to +1.20 %); A/A, 2 pairs, -0.02 %. About 30 pairs would bring the CI to +-0.25 %. `overhead_steps.csv` holds every rollout's inputs.
- **Long run** (`followup/perf/longrun*`): 1200 rollouts on `a829f13` with the profiler on throughout, same settings: 0 alerts, 0 dropped or unread events, 0 lost trace slices and 0 errors; the per-window gather, analysis and self overhead stay flat over 120 windows; GPU memory does not grow; the python processes' RSS stays at 10.04-10.12 GB while training (the first ~60 rollouts peak at about 110 GB).
- **Overlap** (`followup/acceptance/overlap_*`, metrics service on, torch.profiler on two rollouts): at level 1 and on the ordinary steps of level 2, the overlap ratio, the synchronizing calls and the NCCL kernels stay within the spread of two profiler-off runs; the sampled step of level 2 (once per window) adds its one `cudaDeviceSynchronize`.
- **Detection with reporting on** (`followup/acceptance/detection_*.log`): no injection, 0 alerts in 6 windows; GPU 5 SM hog, rank 5 `slow_device` at step 29; rank 3 sleeping 200 ms per step, rank 3 `late_arrival` at step 29.
- **VPP identity** (`followup/vpp/`): PP2 with 2 model chunks per rank and DP4, modules level; `check_vpp_trace.py` checks layer ranges, token counts and cross-stage order. With `--microbatch-group-size-per-virtual-pipeline-stage 3`, which Bridge does not copy into the model config, the code before `e8d3dae` put half of the module slices in the wrong chunk; from `e8d3dae` on, all of them are right.

| File | Content |
| --- | --- |
| `followup/acceptance/` | Per-step grad norm / loss and their summary (`compare_numbers.py`); the same with reporting on, from the training logs (`compare_logged_numbers.py`); overlap verdicts (`overlap_check.py`); detection log lines; the run scripts (set `WORKDIR`, `RELAX_CODE`, `RELAX_ENV`) |
| `followup/perf/` | Crossover and A/A summaries with every rollout's inputs (`summarize_f19.py`); the direct-cost probes (`cost/`: per-position table and total, `cost_breakdown.py`); the long-run summary and monitor samples (`summarize_longrun_f19.py`); the run scripts; the launch scripts (`launch/`); the patch series on top of the PR commit (`patches/`: #462, #463 and the experiment-only sleep injection, A/B toggle, GC counts, full-rollout interval and cost probes) |
| `followup/vpp/` | VPP check outputs, merged traces before and after the group-size fix, the checker and the run script |
