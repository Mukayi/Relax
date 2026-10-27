# Task 11 straggler profiler — evidence

Code: `Mukayi/Relax` branch `feat/task11-straggler-profiler` (PR redai-studio/Relax#363, RFC redai-studio/Relax#362).
Recipe: Qwen3-0.6B SFT DP8, OpenMathReasoning-mini, GBS 32; one rollout = one optimizer step in this recipe.
Hosts: 8×A100 (A100-1, A100-2) and 8×A800 (A800-2, A800-1). The final overhead runs use the metrics service and overlap.
The original measurements were taken on `f203bff`; `followup/` holds the final acceptance evidence. `fig_mechanism` and `fig_detector` show the current code (analysis on a background thread, uncertain verdicts, alert clearing, profiler health).

## Headline numbers

- **Overhead** (`followup/perf/`, tested PR commit `891d0cf`): rank 0's directly probed cost is 1.52 ms per rollout (about 0.15 %). Four opposite-phase in-run crossover pairs give +0.13 % (95 % CI −0.05 % to +0.30 %), whose upper bound is below 0.5 %.
- **Modules level** (`followup/perf/modules/`, after the module commits: `MODULE_SPECS`, forward wrappers, `torch.Event`): the same crossover at level 3 gives +0.34 % (95 % CI +0.20 % to +0.48 %); the MoE and activation-recompute smokes on the PR branch put every module slice inside its forward / backward.
- **Numerics with reporting on** (`followup/acceptance/numbers_reporting_on_*`): loss and grad-norm differences are no larger than an A/A control with the profiler off in both runs.
- **Overlap** (`followup/acceptance/overlap_*`): ordinary steps stay within the A/A spread; the sampled step adds one `cudaDeviceSynchronize`, as designed.
- **Long run** (`followup/perf/longrun*`): 1200 rollouts, 0 alerts, 0 dropped or unread events, 0 lost trace slices, 0 errors and no memory growth.
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

## Follow-up on the final code (2026-10-23 to 2026-10-25)

Each item names its PR commit; acceptance runs also apply #462 (the torch 2.13 fix), #463 (the metrics-service fix) and the listed experiment-only patches. The workload is Qwen3-0.6B SFT DP8 with overlap on.

- **Numbers** (`followup/acceptance/numbers_*`): 40 rollouts with the same seed, every run saving its grad norms. With the profiler at its most intrusive setting (modules level, trace, a window every rollout) against a profiler-off run: grad norm max 2.39 % / median 0.27 %, loss max 0.053 % / median 0.013 %. Two profiler-off runs: 2.31 % / 0.34 % and 0.070 % / 0.018 %.
- **Numbers with reporting on** (`followup/acceptance/numbers_reporting_on_*`): the three overlap runs below, profiler at its default level, 40 rollouts: loss max 0.069 % and grad norm max 2.13 % against a profiler-off run; 0.059 % and 3.20 % between the two profiler-off runs.
- **Overhead** (`followup/perf/`), with full functionality (default level, i.e. level 2, a trace directory, the metrics service and overlap on): probes (`cost/`, on `0c9afc1`) put the profiler's direct cost on rank 0 at 15.2 ms per window (1.52 ms per rollout, about 0.15 %), about 60 % of it in the segment timers; the other ranks wait for rank 0 at the next collective. End to end, the first 20 warm-up rollouts are excluded, then the profiler is switched on and off every 2 rollouts on PR commit `891d0cf`, with the two runs of each pair in opposite order. Every rollout's time is its full interval, from the end of the previous rollout to the end of its own, and the toggle switches only after it (`patches/0011`). Off rollouts only skip the training thread's calls (segment timers, `step_metrics`, schedule and token notes; `patches/0009`); the profiler itself stays installed: its level-poll thread keeps running, and the analysis and trace writing that a window close in an on rollout hands to the background thread can run on into the following off block, so the crossover does not necessarily count that background cost. Adjacent blocks are compared together to cancel data effects and slow host drift. Rollouts where the two runs differ by more than 0.3 s are dropped: 96 of 101 such rollouts have a full garbage collection; full collections split 131 / 140 between on / off rollouts. Each pair gives one value: +0.145 %, −0.011 %, +0.110 % and +0.260 %; the mean is +0.126 % (95 % CI −0.052 % to +0.304 %). Four A/A pseudo-pairs, from runs on PR commit `a829f13` with the profiler not installed (outside docs and tests, `a829f13` differs from `891d0cf` only in the profiler's `reporter.py`), have a standard deviation of 0.043 %. `overhead_steps.csv` holds every rollout's inputs.
- **Long run** (`followup/perf/longrun*`): 1200 rollouts on `a829f13` with the profiler on throughout, same settings: 0 alerts, 0 dropped or unread events, 0 lost trace slices and 0 errors; the per-window gather, analysis and self overhead stay flat over 120 windows; GPU memory does not grow; the python processes' RSS stays at 10.04-10.12 GB while training (the first ~60 rollouts peak at about 110 GB).
- **Overlap** (`followup/acceptance/overlap_*`, metrics service on, torch.profiler on two rollouts): at level 1 and on the ordinary steps of level 2, the overlap ratio, the synchronizing calls and the NCCL kernels stay within the spread of two profiler-off runs; the sampled step of level 2 (once per window) adds its one `cudaDeviceSynchronize`.
- **Detection with reporting on** (`followup/acceptance/detection_*.log`): no injection, 0 alerts in 6 windows; GPU 5 SM hog, rank 5 `slow_device` at step 29; rank 3 sleeping 200 ms per step, rank 3 `late_arrival` at step 29.
- **Modules level** (`followup/perf/modules/`): the crossover of `../exp_accept_p2.sh` at level 3, on `a829f13` + `modules_code.patch` (the module commits, code and tests) + `../patches/0001-0012`: four pairs +0.31 %, +0.31 %, +0.47 %, +0.27 %, overall +0.34 % (95 % CI +0.20 % to +0.48 %); the A/A pseudo pairs on the same blocks −0.06 % to +0.03 %; all 8 runs reach rollout 599 with 0 lost trace slices and 0 errors. Smoke on the PR branch with the module commits (`8444de8` + the same patches), level 3, 20 rollouts: MoE-tiny DP8×EP8 has 32–48 slices per block and rank, Qwen3-0.6B PP4 with full activation recompute 168–189 (7 layers × 8–9 micro-batches × original forward, recompute forward and backward); none outside F / B and no MoE part outside `moe` (`check_module_trace.py`).
- **VPP identity** (`followup/vpp/`): PP2 with 2 model chunks per rank and DP4, modules level; `check_vpp_trace.py` checks layer ranges, token counts and cross-stage order. With `--microbatch-group-size-per-virtual-pipeline-stage 3`, which Bridge does not copy into the model config, the code before `e8d3dae` put half of the module slices in the wrong chunk; from `e8d3dae` on, all of them are right.

| File | Content |
| --- | --- |
| `followup/acceptance/` | Per-step grad norm / loss and their summary (`compare_numbers.py`); the same with reporting on, from the training logs (`compare_logged_numbers.py`); overlap verdicts (`overlap_check.py`); detection log lines; the run scripts (set `WORKDIR`, `RELAX_CODE`, `RELAX_ENV`) |
| `followup/perf/` | Final crossover and A/A summaries with every rollout's inputs (`summarize_p2.py`); the direct-cost probes (`cost/`: per-position table and total, `cost_breakdown.py`); the long-run summary and monitor samples (`summarize_longrun_f19.py`); the run scripts; the launch scripts (`launch/`); the patch series on top of `891d0cf` (`patches/`: #462, #463 and the experiment-only sleep injection, A/B toggle, GC counts, full-rollout interval and cost probes) |
| `followup/perf/modules/` | Modules-level crossover summary and every rollout's inputs (`../summarize_p2.py`), the measured code as a patch on `a829f13`, run health, the smoke trace checks and their checker, and the run script (set `WORKDIR`, `RELAX_CODE` and `PART`) |
| `followup/vpp/` | VPP check outputs, merged traces before and after the group-size fix, the checker and the run script |
| `followup/pr463/` | Reproduction and measurements for upstream PR #463 (metrics client and service fix), in Chinese: client calls with and without the container's proxy variables, requests that arrive while `/report_step` writes to a slow backend, and an 8-GPU end-to-end A/B. README with the tables and figures, per-request data and the scripts |
