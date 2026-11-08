# Conditional gain at a fixed label-support budget

I froze the previous held-source label-support estimates, OLS4
reference and quarter-amplitude candidate. I trained only the
conditional gain: either per-step regression, or the same regression
with a within-query ranking term using only steps labeled for both
agents. Conditional gain heads see neither the frozen support
probabilities nor the held source in training. The same support
scores are added after inference.

This compares 54 fresh Torch fits: three seeds, nine held fitting
sources, two arms, 500 steps per fit. Matching holds the change
count fixed in every within-query quartile of predicted label
availability. All arms change two agents in each query: 2,352
changes per seed across 1,176 queries. Predicted-support matching
does not equate actual masks. The 69,746-window roster is unchanged.

| Support-matched contrast | Observed gain | Full-grid lower gain [95% CI] |
| --- | ---: | ---: |
| Ranked vs support only | +0.1225 | +0.0240 [-0.0712, +0.1201] |
| Pointwise vs support only | +0.1288 | +0.0309 [-0.0496, +0.1168] |
| Ranked vs pointwise | -0.0063 | -0.0250 [-0.0517, -0.0039] |
| Ranked vs OLS4 | +0.1348 | +0.0403 [-0.0686, +0.1384] |
| Ranked vs minimum change | +0.1178 | +0.0192 [-0.0768, +0.1159] |
| Ranked vs original shared head | +0.0155 | -0.0334 [-0.0570, -0.0100] |

Units are points on the observed CV error denominator. The 95%
interval uses 3,000 paired source resamples after seed averaging.
Direct lower bounds debit the two compared forecasts' disagreement;
they cannot be subtracted or flipped to give the opposite bound.
The count-only analysis without support quartiles does not change
the conclusion: ranked vs OLS4 is +0.0805 [-0.0469, +0.2239]
and ranked vs pointwise is -0.0352 [-0.0655, -0.0101].

## What Changed

Disagreement-weighted conditional absolute error is 0.6471 for
pointwise, 0.6551 for ranked, and 0.7172 for donor means. More
accurate conditional regression is possible, but the change from
support only is not statistically clear in trajectory utility.
Matched hindsight is +0.3531 points vs OLS4, not a learned score.
The ranking arm changes about 1,000 queries per seed relative to
support only, so a literal fallback to the same recipients is not
the explanation for the null gain.

Ranked available ADE/FDE is 91.473/234.220 pixels, versus
91.597/234.452 for OLS4. Its hard contribution is -0.0148
points. Among the selected rows, 9.06% of steps are actually
unlabeled in the ranked arm and 7.64% in support only (query-weighted
source averages). This is not a comparison at equal realized support.
Worst supported known-easy harm is 0.38%-0.57% against OLS4 and
261.01%-261.78% against CV; one source has an undefined denominator.
The original 2% rule is not passed.

The next priority is new causal evidence for changes in motion,
such as past-video movement or independently captured observations
with better label completeness. The present common-step loss does
not justify expanding the same selector or opening the sealed test.
General regression deferral is established in
[Mao et al.](https://proceedings.mlr.press/v235/mao24d.html); our
support-stratified, partial-label policy readout does not inherit
its consistency guarantees.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_conditional_gain.py train
.venv-pytorch/bin/python scripts/run_m3w_conditional_gain.py score
.venv-pytorch/bin/python scripts/run_m3w_conditional_gain.py replay
```

Native arm64 CPU, four compute threads, zero loader workers,
checkpoints every 100 updates, training 26.0 s. Use `train --resume`
after confirming the original process ended. Nine targeted tests
passed and the score was replayed byte-for-byte.
[Protocol](protocol.md) | [Results](analysis.json) |
[Paper supplement](../evidence_manuscript_v26/README.md).

Fitting-source evidence only, not independent generalization or
deployment. The DroneCrowd confirmation split and original risk
rule stay unchanged. Pixel coordinates, annotation-step horizons;
no metric, seconds, true 3D, foundation, Stage5C or SMC claim.
