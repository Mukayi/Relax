# In-run crossover `f19-c*` on full rollout intervals (8x A100, Qwen3-0.6B SFT DP8, overlap, metrics service and trace dir on, default level)

- pair 1: rollouts 599/599; median interval ph0 0.9947 s, ph1 0.9913 s
- pair 2: rollouts 599/599; median interval ph0 0.9928 s, ph1 0.9910 s

**overhead without stalls, one value per pair (56 rollouts dropped, 56 of them with a full collection): -0.020% (95% CI [-2.584%, +2.545%], n=2 pairs; pairs +0.18% -0.22%)**
the same with blocks taken as independent: -0.021% ± 0.437% (n=109 blocks)
overhead with every rollout, blocks taken as independent: +0.074% ± 0.695% (n=116 blocks)
run-level offset ph0/ph1 - 1 (cancels in the crossover): +0.011% ± 0.695%; block sd 3.781%
full collections (rank-rollouts) in on blocks 66, in off blocks 71

on minus off by position in the block, median ms (interval, actor_train_time, train_wait_time):
- 0: +1.5, +1.9, +0.4 (n=112)
- 1: -1.6, -3.9, +0.3 (n=110)
- 2: +1.0, -0.4, +0.7 (n=107)
- 3: -1.4, -0.5, -0.4 (n=110)
- 4: +4.7, +3.3, +0.2 (n=112)
- 5: +2.1, +1.4, +0.2 (n=110)
- 6: +2.8, +2.3, +0.4 (n=108)
- 7: +3.4, +2.0, +0.0 (n=112)
- 8: -2.9, -3.2, +0.1 (n=112)
- 9: -0.4, +2.1, -0.8 (n=111)
