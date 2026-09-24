# Crossover overhead — all pairs (09-23 morning + 09-24 afternoon)

Pre-registered metric: for each 10-rollout block (after 2 warm-up blocks),
`mean(on) / mean(off) − 1`, with phase-0 and phase-1 runs of opposite toggle phase.
Code: PR HEAD + experiment-only on/off toggle (worktree `exp/task11-interleave-v2`).
Metrics service **off** unless noted. One rollout = one optimizer step in this recipe.

## Per-pair results (block mean)

| Pair | Host | Kind | Overhead | 95% CI | Blocks | Block sd | Median step_time ph0/ph1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| il-p1 | ge40-11 A100 | crossover | −0.04% ± 0.36% | −0.40% ~ +0.31% | 28 | 0.92% | 1.141 / 1.142 s |
| v2a-p1 | ge40-11 A100 | crossover | +2.04% ± 1.70% | +0.34% ~ +3.74% | 28 | 4.38% | 1.139 / 1.137 s |
| v2b-p1 | ge50-12 A100 | crossover | −0.16% ± 0.79% | −0.95% ~ +0.63% | 28 | 2.04% | 0.911 / 0.916 s |
| v2c-p1 | ge26-60 A800 | crossover | −0.15% ± 1.89% | −2.04% ~ +1.74% | 28 | 4.87% | 0.962 / 0.958 s |
| v2m-p1 | ge26-59 A800 | crossover, metrics service **on** | +0.78% ± 1.78% | −0.99% ~ +2.56% | 28 | 4.58% | 0.964 / 0.960 s |
| v2a-c1 | ge40-11 A100 | A/A (profiler off both runs) | +0.41% ± 0.59% | −0.18% ~ +0.99% | 28 | 1.51% | 1.143 / 1.145 s |
| v2b-c1 | ge50-12 A100 | A/A | −0.70% ± 1.67% | −2.37% ~ +0.97% | 28 | 4.31% | 0.917 / 0.916 s |
| v2c-c1 | ge26-60 A800 | A/A | −1.41% ± 1.46% | −2.87% ~ +0.05% | 28 | 3.76% | 0.962 / 0.960 s |

Pair-level mean of the 4 metrics-service-off crossover pairs: **+0.42% ± 1.72%** (95% CI −1.30% ~ +2.14%, n = 4).
A/A pair means (+0.41 / −0.70 / −1.41) span a similar range: the crossover design's own noise floor is not much smaller than the on/off differences on these hosts.

## Why one crossover pair is an outlier

On v2a-p1 (ge40-11), one-sided `actor_train_time` spikes all landed in on-blocks; the same windows show full (generation-2) GC of ~68 ms/step on every rank. A/A on the same host has far fewer one-sided spikes. On ge50-12 / ge26-60, A/A runs have as many one-sided spikes as the crossover runs, so the afternoon noise is not unique to the toggle.

## Post-hoc (not pre-registered): drop spike steps

Drop any step whose `step_time` is > 1.25× that run's median in either run of the pair (~5% of steps), then recompute the same block means:

| Pair | After drop |
| --- | --- |
| il-p1 | −0.18% ± 0.38% |
| v2a-p1 | −0.06% ± 0.41% |
| v2b-p1 | +0.07% ± 0.26% |
| v2c-p1 | −0.09% ± 0.25% |
| v2m-p1 | +0.23% ± 0.34% |
| v2a-c1 / v2b-c1 / v2c-c1 | +0.15% / +0.18% / −0.07% |

## Raw blocks

See `crossover_all_blocks.csv` (columns: pair, block, on_over_off_minus_1, mean_ph0, mean_ph1).
