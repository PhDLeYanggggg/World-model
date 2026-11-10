# OpenDD: predicting missing labels better does not improve ranking

I trained a small head to predict whether each future step will be labeled,
from eight observed steps and current query size. For every held/calibration
location pair, only the other five locations were used for fitting. The
head never sees future positions. Both the head and a simpler control that
uses only donor step averages penalize the same fixed support score by the
expected movement on unlabeled steps. All arms choose 2,196 recording
views, with both movement budgets within 1% of the original merit mask.

| Result on fixed proposal actors | Donor step profile | Learned causal probabilities |
|---|---:|---:|
| Missing-label Brier score | 0.11727 | 0.11211 |
| Normalized debit mean absolute error | 0.14484 | 0.13957 |
| Risk and both utility passes | 742 | 745 |
| Original risk passes | 849 | 856 |
| Undefined selected-easy risk | 1,112 | 1,111 |

The learned minus profile Brier difference is -0.00515 [-0.00644, -0.00375]
and debit MAE difference is -0.00527 [-0.00910, -0.00195]. Nevertheless,
the paired conservative ADE gain is **-0.000120 pp [-0.000229, -0.000034]**,
and conservative relative-motion gain is -0.00102 native units
[-0.00162, -0.00058]. The dual-pass rate difference is -0.01 pp
[-0.56, +0.59] after weighting locations equally. The 95% intervals resample
seven locations 3,000 times, after averaging repeated directions and seeds.

Against the unchanged support-weighted merit, which qualifies 734 views,
the learned penalty changes conservative trajectory gain by -0.000275 pp
[-0.000563, -0.000073]. The profile penalty also loses. This rules out a
simple plug-in penalty as a reliable fix in this cohort. It does not rule
out learning an incremental conservative contrast against a specific
replacement mask. Better marginal probabilities need not improve the
ranking of near-tied choices. The observed-to-conservative difference in
a direct comparison charges missing steps for changed forecasts on both
sides, not just the added proposal's expected missingness.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_step_visibility_rank.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_step_visibility_rank.py score
```

`fresh_run`: 63 native CPU fits, 12,600 updates, three seeds, 126 held
directions. Four threads, no workers, checkpoints with optimizer and
sampler state every 100 updates, and a heartbeat. [Frozen predictions and
choices](freeze.json) published at `0d2971e8c` before held scoring.
`cached_verified`: the fixed forecasts, parent support masks and cohort.
`not_run`: CREATE, Athens, any new external confirmation or deployment.
The original 2% checks are unchanged. [Protocol](protocol.md) |
[Full readout](readout.json).
