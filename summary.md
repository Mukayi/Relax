# exp_0923 summary (8x A100-SXM4-80GB, ge40-11, Relax dfa7d23)

## Overhead (steps 10-59, paired off_i vs on_i)

| pair | off mean | on mean | diff |
|---|---|---|---|
| 1 | 1.1554 s | 1.1359 s | -1.69% (n=50) |
| 2 | 1.1508 s | 1.1446 s | -0.54% (n=50) |
| 3 | 1.1415 s | 1.1467 s | +0.46% (n=50) |
| **pooled** | | | **-0.51% ± 0.60%** (95% CI, n=150) |

## Same-seed loss (all logged steps)

| comparison | steps | max abs diff | identical steps |
|---|---|---|---|
| off1 vs off2 | 60 | 2.863e-04 | 1 |
| off1 vs off3 | 60 | 4.302e-04 | 1 |
| off2 vs off3 | 60 | 3.834e-04 | 1 |
| off1 vs on1 | 60 | 3.376e-04 | 1 |
| off2 vs on2 | 60 | 4.337e-04 | 1 |
| off3 vs on3 | 60 | 3.075e-04 | 1 |
| on1 vs on2 | 60 | 3.972e-04 | 1 |

## Straggler WARNINGs (false alarms in on-runs / hits in injection)

- a100-on1: 0 straggler WARNING line(s)
- a100-on2: 0 straggler WARNING line(s)
- a100-on3: 0 straggler WARNING line(s)
- a100-inj-gpu5: 1 straggler WARNING line(s)
    - `step=29 rank 5 (rank5_dp5_tp0_pp0, host=aiplatform-wlf2-ge40-11.idchb2az2.hb2.kwaidc.com, gpu=5) self 1895.4 ms/step = 2.52x peers (z=109.3) for 3 windows; tokens 0.99x, ms/ktok 2.55x, gc 1.7 ms -> slow_device`

## Injection mapping (hog on physical GPU 5)

- rank -> GPU: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
- rank on GPU 5: [5]; ranks in straggler WARNINGs: [5]
