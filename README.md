# Task 11 straggler profiler — evidence

Code: `Mukayi/Relax` branch `feat/task11-straggler-profiler` (PR redai-studio/Relax#363, RFC redai-studio/Relax#362).
Recipe: Qwen3-0.6B SFT DP8, OpenMathReasoning-mini, GBS 32; one rollout = one optimizer step in this recipe.
Hosts: 8×A100 (ge40-11, ge50-12) and 8×A800 (ge26-60, ge26-59). Overhead runs use the metrics service **off** unless noted.

## Headline numbers

- **Overhead, in-run crossover** (profiler toggled every 10 rollouts; opposite-phase pair): metrics-service-off pairs −0.04 % / +2.04 % / −0.16 % / −0.15 % (each 28 blocks). Pair-level mean +0.42 % ± 1.72 % (n = 4). The +2 % pair on ge40-11 has full-GC spikes only in on-blocks. A/A (profiler off both runs) on three hosts: +0.41 % / −0.70 % / −1.41 %. Metrics service on: +0.78 % ± 1.78 % (1 pair). Details in `crossover_summary.md`.
- **GC count** (pre-registered): turning the profiler on does **not** increase generation-2 GC count (11/12 runs gen2 = 1; the only gen2 = 2 is profiler-off). See `gc_counts.md`.
- **Overhead, separate runs**: 3 off/on pairs × 60 steps differ by −1.69 % / −0.54 % / +0.46 %; the off runs alone differ by 1.2 %. Run-level 95 % CI −3.3 % to +2.1 % (n = 3).
- **Overhead, summed upper bound** (8 ranks, critical-path assumption): typical 1.2 ms/step (≈ 0.10 %), worst 1.9 ms/step (≈ 0.17 %). See `bench_timer_cost_ge40.txt`.
- **Loss**: max relative |Δ| off↔on 0.090 %, off↔off 0.094 % (same seed).
- **False alarms**: 0 WARNINGs in 12 reportable windows (3 clean profiler-on runs × steps 29–59).
- **Injection** (bf16 matmul hog on GPU 5, 2.5×): step 29 — WARNING rank 5 → `slow_device`.
- **Real CPU-side slow rank** (`fig_real_case`): ge26-59, earlier prototype, report interval 5, metrics service on.

## Files

| File | Content |
| --- | --- |
| `crossover_summary.md`, `crossover_blocks.csv`, `crossover_all_blocks.csv` | All crossover / A/A pairs; figure uses the 4 metrics-off pairs |
| `gc_counts.md` | Generation-2 GC counts on vs off (4 hosts × 3 runs) |
| `bench_timer_cost_ge40.txt` | Single-GPU hot-path microbenchmark output |
| `steps.csv`, `summary.md` | Separate-run A/B: per-step `step_time` / `loss` for 3 off/on pairs |
| `figures/fig_mechanism` | End-to-end mechanism (hot path orange, cold path blue) |
| `figures/fig_detector` | Detector: peer groups, robust z, 3-window persistence, priority ladder |
| `figures/fig_late_arrival` | Why the latest rank has the shortest `dp_grad_sync` |
| `figures/fig_injection` | Per-rank self / wait at step 29 with a matmul hog on GPU 5 |
| `figures/fig_real_case` | Real CPU-side slow rank |
| `figures/fig_overhead` | Separate-run A/B and in-run crossover (4 pairs) |
| `figures/fig_loss` | Same-seed loss off vs on |
| `figures/make_figures.py` | Regenerates every figure (PNG + SVG) |
| `scripts/` | Microbenchmark, crossover driver, holders, experiment-only toggle patch |
