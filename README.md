# Task 11 straggler profiler — evidence

Code: `Mukayi/Relax` branch `feat/task11-straggler-profiler` (PR redai-studio/Relax#363, RFC redai-studio/Relax#362).
Recipe: Qwen3-0.6B SFT DP8 on 8×A100-SXM4-80GB (ge40-11), OpenMathReasoning-mini, GBS 32; one rollout = one optimizer step in this recipe. All overhead measurements were taken with the metrics service **off**.

## Headline numbers

- **Overhead, in-run crossover** (profiler toggled every 10 rollouts inside a run, second run in the opposite phase, so each 10-rollout block is measured once on and once off on identical data): −0.04 % ± 0.36 % (95 % CI −0.40 % to +0.31 %, 28 blocks, lag-1 autocorrelation −0.08). One pair only, no A/A control; run on `dfa7d23` + the experiment-only toggle (`scripts/in_run_ab_toggle.patch`), before the reporting change `5e32e06` (which only removes a request).
- **Overhead, separate runs**: 3 off/on pairs × 60 steps differ by −1.69 % / −0.54 % / +0.46 %; the off runs alone differ by 1.2 %. Run-level 95 % CI −3.3 % to +2.1 % (n = 3), too wide to show < 0.5 %.
- **Overhead, summed upper bound** (8 ranks, assuming everything is on the critical path): typical 1.2 ms/step (≈ 0.10 % of a 1.15 s step), worst 1.9 ms/step (≈ 0.17 %). Timer calls + end-of-step readout: +0.80 ms/step (max 0.97) in a kernel-launch-bound loop on an idle A100 (`bench_timer_cost_ge40.txt`; CPU time of the timer calls alone 0.51 ms/step, ≈ 30 µs per call); gather + analysis amortised: 0.4 ms (max 0.93).
- **Loss**: max relative |Δ| off↔on 0.090 %, off↔off 0.094 % (same seed).
- **False alarms**: 0 WARNINGs in 12 reportable windows (3 clean profiler-on runs × steps 29–59).
- **Injection** (bf16 matmul hog on GPU 5, one strength only, 2.5×): step 29 — the earliest window the 3-window rule allows — WARNING rank 5 → `slow_device`.
- **Real CPU-side slow rank** (`fig_real_case`): ge26-59, 8×A800, earlier prototype commit, report interval 5, metrics service on.

## Files

| File | Content |
| --- | --- |
| `crossover_blocks.csv`, `crossover_summary.md` | In-run crossover: per-block means of both runs and on/off − 1; summary with CI |
| `bench_timer_cost_ge40.txt` | Single-GPU hot-path microbenchmark output |
| `steps.csv`, `summary.md` | Separate-run A/B: per-step `step_time` / `loss` for 3 off/on pairs (steps 10–59); summary |
| `figures/fig_mechanism` | End-to-end mechanism (hot path orange, cold path blue) |
| `figures/fig_detector` | Detector: peer groups, robust z, 3-window persistence, priority ladder |
| `figures/fig_late_arrival` | Why the latest rank has the shortest `dp_grad_sync` (schematic) |
| `figures/fig_injection` | Per-rank self / wait at step 29 with a matmul hog on GPU 5 |
| `figures/fig_real_case` | Real CPU-side slow rank (upstream default config) |
| `figures/fig_overhead` | Separate-run A/B and in-run crossover |
| `figures/fig_loss` | Same-seed loss off vs on, per-step relative difference vs off/off |
| `figures/make_figures.py` | Regenerates every figure (PNG + SVG) |
| `scripts/bench_timer_cost.py` | Microbenchmark |
| `scripts/exp_interleave.sh`, `scripts/summarize_interleave.py`, `scripts/gpu_hold.py` | Crossover driver (with GPU holders against a shared GPU queue) and analysis |
| `scripts/in_run_ab_toggle.patch` | Experiment-only on/off toggle (not part of the PR) |
