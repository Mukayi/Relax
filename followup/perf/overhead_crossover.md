# In-run crossover `tr-p*` on full rollout intervals (8x A100, Qwen3-0.6B SFT DP8, overlap, metrics service and trace dir on, default level)

- pair 1: rollouts 599/599; median interval ph0 0.9986 s, ph1 0.9969 s
- pair 2: rollouts 599/599; median interval ph0 0.9947 s, ph1 0.9952 s
- pair 3: rollouts 599/599; median interval ph0 0.9915 s, ph1 0.9934 s
- pair 4: rollouts 599/599; median interval ph0 0.9942 s, ph1 0.9956 s

**overhead without stalls, one value per pair (127 rollouts dropped, 124 of them with a full collection): -0.129% (95% CI [-1.457%, +1.199%], n=4 pairs; pairs -0.53% -0.51% -0.60% +1.12%)**
the same with blocks taken as independent: -0.115% ± 0.293% (n=221 blocks)
overhead with every rollout, blocks taken as independent: +0.120% ± 0.571% (n=232 blocks)
run-level offset ph0/ph1 - 1 (cancels in the crossover): +0.151% ± 0.571%; block sd 4.413%
full collections (rank-rollouts) in on blocks 138, in off blocks 128

on minus off by position in the block, median ms (interval, actor_train_time, train_wait_time):
- 0: +5.3, +3.5, -2.2 (n=221)
- 1: +4.6, +3.9, +3.6 (n=222)
- 2: -0.3, +0.8, +0.3 (n=217)
- 3: +0.8, -0.0, -0.2 (n=218)
- 4: -0.8, -1.0, +0.1 (n=220)
- 5: -1.4, -1.0, -0.8 (n=220)
- 6: +0.3, +0.3, -0.3 (n=219)
- 7: -2.1, -1.9, -0.6 (n=216)
- 8: -1.3, -0.4, -0.1 (n=223)
- 9: -0.1, -1.9, -0.5 (n=217)
