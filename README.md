# Task 11 straggler profiler — evidence (2026-09-23)

Machine: ge40-11, 8×A100-SXM4-80GB. Code: `Mukayi/Relax@dfa7d23` (`feat/task11-straggler-profiler`).
Recipe: Qwen3-0.6B SFT DP8, OpenMathReasoning-mini, 60 steps, `RELAX_STRAGGLER_REPORT_INTERVAL=10`.

| File | Content |
| --- | --- |
| `summary.md` | Overhead, loss diffs, false-alarm count, injection hit |
| `steps.csv` | Per-step `step_time` / `loss` for 3 off/on pairs (steps 10–59) |
| `figures/fig_mechanism` | End-to-end mechanism (hot path / cold path / primary rank) |
| `figures/fig_detector` | Detector: peer groups, robust z, 3-window persistence, priority ladder |
| `figures/fig_late_arrival` | Why the latest rank has the shortest `dp_grad_sync` (schematic) |
| `figures/fig_injection` | Per-rank self / wait at step 29 with a matmul hog on GPU 5 |
| `figures/fig_real_case` | Real host-side straggler (upstream default config, report interval 5) |
| `figures/fig_overhead` | Per-run step_time and run-level difference vs run-to-run noise |
| `figures/fig_loss` | Same-seed loss off vs on, per-step relative difference vs off/off |
| `figures/make_figures.py` | Regenerates every figure (PNG + SVG) |

**Headline numbers**

- Overhead: tool-reported ≈ 0.6 ms/step (≈ 0.05% of step_time, excl. `event.record()`); end-to-end on/off difference per pair −1.69% / −0.54% / +0.46% is the same order as run-to-run noise (off runs alone differ by 1.2%); no positive overhead detected. Run-level 95% CI −3.3% ~ +2.1% (n=3) — not enough pairs to prove < 0.5% end-to-end. The per-step pooled −0.51% ± 0.60% (n=150) ignores run-level variance.
- Loss: max relative |Δ| off↔on 0.090%, off↔off 0.094%
- False alarms: 0 WARNINGs in 12 reportable windows (3 clean on-runs × steps 29–59)
- Injection (matmul hog on GPU 5, one strength only): step 29 WARNING rank 5 → `slow_device` (2.52× peers)
- Real host-side case (`fig_real_case`): ge26-59, 8×A800, earlier prototype commit, report interval 5
