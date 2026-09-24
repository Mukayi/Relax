# In-run on/off crossover `il-p*` (8x A100, Qwen3-0.6B SFT DP8, 10-step blocks)

- pair 1: steps 300/300; median step_time ph0 1.141 s, ph1 1.142 s; straggler windows ph0 [9, 29, 49, 69]... ph1 [19, 39, 59, 79]...
- suspect runs (median step_time > 1.3x fastest 1.141 s): none

**overhead (on/off - 1), block-level: -0.041% ± 0.356% (95% CI, n=28 blocks), i.e. [-0.397%, +0.315%]**
run-level offset ph0/ph1 - 1 (cancels in the crossover): +0.004% ± 0.357%
block sd 0.917%
