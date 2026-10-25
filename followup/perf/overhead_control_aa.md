# A/A control for the two-rollout estimator

The profiler is not installed in any run. After the first 20 warm-up rollouts, the final estimator relabels the full-rollout intervals in two-rollout blocks. The true effect is therefore zero.

- natural pair 1: −0.008%
- natural pair 2: −0.064%
- all phase-0 × phase-1 pseudo-pairs: −0.008%, −0.056%, +0.027%, −0.064%
- pseudo-pair standard deviation: **0.043%**

The natural pairs drop 56 rollouts where the two runs differ by more than 0.3 s; all 56 have a full garbage collection. `overhead_steps_control_aa.csv` contains their per-rollout inputs.
