# Does training for two changes improve recipient choice?

I gave every agent one candidate, nominated by errors on short
forecasts within the eight observed steps. All arms use exactly a
quarter of the same candidate displacement from OLS4. Thus the
comparison changes the recipients, not their proposed trajectories.
Pointwise regression estimates per-agent gain. The top-two arm
optimizes the exact expected gain of two weighted draws without
replacement and takes the two highest scores at inference. The
context arm adds seven current-query features relative to other
agents. The sampling model is established
[ranking methodology](https://doi.org/10.2307/2346567), not a novelty
claim.

## Results

81 fresh fits, three seeds and nine source-held folds; each uses 600
optimizer updates. There are 69,746 windows in 1,176 queries and
exactly 2,352 changes per active arm per seed. The pairs have the
same per-query counts; they can select candidates with different
absolute displacements. Intervals use 3,000 paired source draws
after seed averaging. Scores are points on the observed CV error
denominator, not percent reductions in ADE.

| Direct contrast | Full-grid lower gain [95% interval] |
| --- | ---: |
| Top-two vs OLS4 | -0.0672 [-0.1379, -0.0034] |
| Top-two vs minimum change | -0.1102 [-0.1941, -0.0356] |
| Top-two vs pointwise | -0.1351 [-0.2498, -0.0481] |
| Top-two + context vs OLS4 | -0.0200 [-0.0926, +0.0285] |
| Top-two + context vs top-two | -0.1332 [-0.2054, -0.0671] |
| Top-two + context vs minimum change | -0.0555 [-0.1101, -0.0114] |
| Pointwise vs OLS4 | -0.0077 [-0.0640, +0.0344] |

Direct lower bounds are not subtractable: the disagreement between
two forecasts differs from their disagreement with OLS4. Negative
lower scores fail to establish a positive conservative gain; they
do not identify unobserved forecast errors. Available ADE/FDE are
91.533/234.147 image pixels for top-two, 91.496/233.959 with
context, and 91.597/234.452 for OLS4. Top-two's hard contribution
is -0.0262 points. Its worst supported known-easy positive harm
is 0.27%-0.39% against OLS4, but 260.13%-261.89% against CV.
One source lacks an easy denominator. Neither the matched counts
nor the OLS4 comparison passes the original recording-risk rule.

## Generalization And Available Gain

Re-inference from all 81 checkpoints exactly reproduces held scores.
On the same windows, the top-two arm averages +0.0291 points when
that source is in training and -0.0672 when held out. Context moves
from +0.0745 to -0.0200; pointwise moves from +0.0233 to -0.0077.
Train-seen values are resubstitution diagnostics, not validation.
They are consistent with limited transfer of the selection rule;
they do not distinguish source bias from overfitting alone.

Post-hoc hindsight controls use the same counts and query roster:

| Hindsight candidates | Quarter displacement | Full displacement |
| --- | ---: | ---: |
| Fixed nomination | +0.5509 [+0.3674, +0.7500] | +1.7924 [+1.1638, +2.4681] |
| Full menu | +1.0527 [+0.6103, +1.5443] | +3.2732 [+2.0566, +4.5773] |

The restriction and the smaller amplitude each reduce possible
gain, but neither explains why the learned ranker captures none.
These label-aware controls do not select a new amplitude or establish
that larger changes are safe. The next experiment should improve
the transferable evidence for intervention (for example causal
track support and observation quality) and verify this on held
sources, before increasing the ranker's size or restoring joint terms.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_recipient_top2.py train
.venv-pytorch/bin/python scripts/run_m3w_recipient_top2.py score
.venv-pytorch/bin/python scripts/run_m3w_recipient_top2.py replay
.venv-pytorch/bin/python scripts/report_m3w_recipient_top2_generalization.py
```

Local native arm64 CPU, four compute threads and zero loader workers.
Training took 68.4 seconds; checkpoints are saved every 100 steps.
Use `train --resume` after interruption. Eight targeted tests passed.
Frozen scoring and the separate generalization diagnostic each ran
twice with identical results. No new CREATE jobs.

[Protocol](protocol.md) | [Results](analysis.json) |
[Checkpoint and hindsight diagnostic](generalization_diagnostic.json).
Fitting-source evidence only: no changed threshold, deployment,
DroneCrowd calibration or confirmation access, or Stage5C/SMC.
Distances are image pixels and horizons are annotation steps.
