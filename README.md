# Task 11 straggler profiler — evidence (2026-09-23)

Machine: ge40-11, 8×A100-SXM4-80GB. Code: `Mukayi/Relax@dfa7d23` (`feat/task11-straggler-profiler`).
Recipe: Qwen3-0.6B SFT DP8, OpenMathReasoning-mini, 60 steps, `RELAX_STRAGGLER_REPORT_INTERVAL=10`.

| File | Content |
| --- | --- |
| `summary.md` | Overhead CI, loss diffs, false-alarm count, injection hit |
| `steps.csv` | Per-step `step_time` / `loss` for 3 off/on pairs (steps 10–59) |
| `overhead.png` | Paired step_time + overhead histogram |
| `loss.png` | off vs on loss scatter |
| `injection.png` | Per-rank `self_ms` at step 29 with hog on GPU 5 |

**Headline numbers**

- Overhead: tool-reported ≈ 0.6 ms/step (≈ 0.05% of step_time, excl. `event.record()`); end-to-end on/off difference per pair −1.69% / −0.54% / +0.46% is within run-to-run noise (off runs alone differ by 1.2%). Run-level 95% CI −3.3% ~ +2.1% (n=3) — not enough pairs to prove < 0.5% end-to-end. The per-step pooled −0.51% ± 0.60% (n=150) ignores run-level variance.
- Loss: max |Δ| off↔on ≈ 3–4×10⁻⁴, same order as off↔off
- False alarms on clean on-runs: **0**
- Injection (matmul hog on GPU 5): step 29 WARNING rank 5 → `slow_device` (2.52× peers)
