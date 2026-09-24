# GC count experiment (pre-registered analysis, 09-24)

Question: does installing the profiler increase the number of full (generation-2) Python GC collections?

Method: `GCSTAT=1` (experiment-only patch on `exp/task11-interleave-v2`) registers a `gc.callbacks` counter on every rank, independent of the profiler. Each host ran three 300-step jobs with the same seed: ABA = on / off / on, or BAB = off / on / off. Metrics service off. Primary metric: gen2 count and gen2 wall time at step 299 (Ray may collapse identical rank lines to `rank=0 … [repeated 8x]`).

| Host | Tag | Profiler | gen2 | gen2_s | collections [g0, g1, g2] | median step_time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| ge40-11 A100 | gca-gon1 | on | 1 | 0.662 | [6470, 590, 17] | 1.144 |
| ge40-11 A100 | gca-goff1 | off | 1 | 0.714 | [6466, 589, 17] | 1.134 |
| ge40-11 A100 | gca-gon2 | on | 1 | 1.098 | [6470, 590, 17] | 1.134 |
| ge50-12 A100 | gcb-goff1 | off | **2** | 1.076 | [6459, 588, 18] | 0.913 |
| ge50-12 A100 | gcb-gon1 | on | 1 | 0.536 | [6467, 589, 17] | 0.916 |
| ge50-12 A100 | gcb-goff2 | off | 1 | 0.619 | [6463, 588, 17] | 0.915 |
| ge26-60 A800 | gcc-gon1 | on | 1 | 0.561 | [6468, 589, 17] | 0.959 |
| ge26-60 A800 | gcc-goff1 | off | 1 | 0.570 | [6463, 588, 17] | 0.960 |
| ge26-60 A800 | gcc-gon2 | on | 1 | 0.780 | [6467, 589, 17] | 0.958 |
| ge26-59 A800 | gcd-goff1 | off | 1 | 0.636 | [6460, 588, 17] | 0.945 |
| ge26-59 A800 | gcd-gon1 | on | 1 | 0.559 | [6468, 589, 17] | 0.951 |
| ge26-59 A800 | gcd-goff2 | off | 1 | 0.626 | [6460, 588, 17] | 0.945 |

**Result:** turning the profiler on does **not** increase gen2 count. Eleven of twelve runs have gen2 = 1; the only gen2 = 2 run is a profiler-off run. Total collection counts differ by a few units. gen2 wall time varies more across repeats of the same condition than between on and off. Median step_time on/off differences stay inside the same-host run-to-run noise.

Implication for the crossover outlier: attributing a GC spike to an on-block is consistent with a **timing shift** of an existing full GC into an on window, not with the profiler causing more full GCs.
