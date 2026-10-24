# In-run crossover `f19-p*` on full rollout intervals (8x A100, Qwen3-0.6B SFT DP8, overlap, metrics service and trace dir on, default level)

- pair 1: rollouts 599/599; median interval ph0 0.9966 s, ph1 0.9938 s
- pair 2: rollouts 599/599; median interval ph0 0.9898 s, ph1 0.9932 s
- pair 3: rollouts 599/599; median interval ph0 0.9942 s, ph1 0.9942 s
- pair 4: rollouts 599/599; median interval ph0 0.9983 s, ph1 0.9949 s
- pair 5: rollouts 599/599; median interval ph0 0.9916 s, ph1 0.9954 s
- pair 6: rollouts 599/599; median interval ph0 0.9946 s, ph1 0.9936 s
- pair 7: rollouts 599/599; median interval ph0 0.9904 s, ph1 0.9948 s

**overhead without stalls, one value per pair (198 rollouts dropped, 187 of them with a full collection): +0.520% (95% CI [+0.037%, +1.002%], n=7 pairs; pairs -0.05% +0.60% +0.90% -0.09% +0.12% +0.99% +1.15%)**
the same with blocks taken as independent: +0.519% ± 0.210% (n=383 blocks)
overhead with every rollout, blocks taken as independent: +0.752% ± 0.929% (n=406 blocks)
run-level offset ph0/ph1 - 1 (cancels in the crossover): -0.412% ± 0.647%; block sd 9.520%
full collections (rank-rollouts) in on blocks 227, in off blocks 234

on minus off by position in the block, median ms (interval, actor_train_time, train_wait_time):
- 0: +9.5, +1.9, -2.3 (n=390)
- 1: +2.1, +1.6, +8.2 (n=385)
- 2: +2.2, +1.6, +0.7 (n=380)
- 3: +3.0, +2.3, +0.5 (n=382)
- 4: +3.2, +2.7, +0.3 (n=379)
- 5: +2.5, +2.6, +0.8 (n=386)
- 6: +0.6, -0.3, +0.6 (n=384)
- 7: +2.7, +2.2, +0.5 (n=393)
- 8: +0.8, +0.7, +0.2 (n=399)
- 9: +2.9, +0.2, -0.1 (n=384)
