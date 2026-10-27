# In-run crossover `m3-p*`, toggled every 2 rollouts, full rollout intervals (8x A100, Qwen3-0.6B SFT DP8, overlap, metrics service and trace dir on, default level)

First 20 rollouts are excluded as warm-up.

- pair 1: rollouts 599/599; median interval ph0 0.9974 s, ph1 0.9963 s; 285 adjacent-block estimates, mean +0.312%
- pair 2: rollouts 599/599; median interval ph0 1.0012 s, ph1 0.9961 s; 279 adjacent-block estimates, mean +0.308%
- pair 3: rollouts 599/599; median interval ph0 0.9958 s, ph1 0.9952 s; 283 adjacent-block estimates, mean +0.469%
- pair 4: rollouts 599/599; median interval ph0 0.9973 s, ph1 0.9958 s; 281 adjacent-block estimates, mean +0.269%

**overhead: +0.340% (95% CI [+0.199%, +0.480%], n=4 pairs; pairs +0.312% +0.308% +0.469% +0.269%)**
dropped rollouts (runs differ by > 0.3 s): 155, 149 of them with a full collection; full collections (rank-rollouts) in on rollouts 176, in off rollouts 151
A/A (f19-c*, profiler not installed), same blocks, pseudo pairs: -0.008% -0.056% +0.027% -0.064%; sd 0.043%
