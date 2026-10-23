# In-run on/off crossover `pf-c*` (8x A100, Qwen3-0.6B SFT DP8, overlap on, metrics service on, 10-step blocks)

- pair 1: steps 300/300; median step_time ph0 0.989 s, ph1 0.985 s; straggler windows ph0 []... ph1 []...; largest loss / error counters {'pf-c1-ph0': {}, 'pf-c1-ph1': {}}

**overhead without stalls (steps where the runs differ by > 0.3 s dropped: 5): +0.575% ± 0.865% (95% CI, n=28 blocks), i.e. [-0.290%, +1.440%]**
overhead with every step: +0.733% ± 1.180% (95% CI, n=28 blocks), i.e. [-0.447%, +1.913%]
run-level offset ph0/ph1 - 1 (cancels in the crossover): -0.502% ± 1.152%
block sd 3.043%
