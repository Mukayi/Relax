# In-run crossover `p2-p*`, toggled every 2 rollouts, full rollout intervals (8x A100, Qwen3-0.6B SFT DP8, overlap, metrics service and trace dir on, default level)

First 20 rollouts are excluded as warm-up.

- pair 1: rollouts 599/599; median interval ph0 0.9903 s, ph1 0.9985 s; 285 adjacent-block estimates, mean +0.145%
- pair 2: rollouts 599/599; median interval ph0 0.9941 s, ph1 0.9958 s; 283 adjacent-block estimates, mean -0.011%
- pair 3: rollouts 599/599; median interval ph0 0.9967 s, ph1 0.9952 s; 276 adjacent-block estimates, mean +0.110%
- pair 4: rollouts 599/599; median interval ph0 0.9958 s, ph1 0.9961 s; 289 adjacent-block estimates, mean +0.260%

**overhead: +0.126% (95% CI [-0.052%, +0.304%], n=4 pairs; pairs +0.145% -0.011% +0.110% +0.260%)**
dropped rollouts (runs differ by > 0.3 s): 101, 96 of them with a full collection; full collections (rank-rollouts) in on rollouts 131, in off rollouts 140
A/A (f19-c*, profiler not installed), same blocks, pseudo pairs: -0.008% -0.056% +0.027% -0.064%; sd 0.043%
