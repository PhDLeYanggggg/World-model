# Strong-Base Auxiliary Study: Complete Readout, No Promotion

## Result
Preserving the strong original estimator does not make this cap-event
auxiliary a reliable expected-cost improvement. All144 fresh no-auxiliary
controls reproduce the original model state and held predictions bit-exactly.
The primary cost, tail/coverage guard and task-information gates still fail.
No model or deployment policy is promoted.

The completed experiment contains432 native-Torch heads,864000 updates,
three seeds and144 source-held readouts. Prediction freeze df49e670 preceded
readout;1152 direct component-MSE checks pass. No new trajectory forecaster,
policy, threshold, independent selection, reserved calibration or confirmation
was evaluated. Source forecasts and nested teachers are cached_verified;
the new training, original-control reconstruction and readout are fresh_run.

## Complete Primary Summary
Counts below are positive/negative/overlapping95% locality intervals across
six overlapping assignments, not independent discoveries.

| Comparison | Full inputs | Motion only | Full point range (%) |
|---|---|---|---|
| True auxiliary / original | 1 / 1 / 4 | 0 / 1 / 5 | -14.8538 to +0.6485 |
| True auxiliary / reconstructed control | 1 / 1 / 4 | 0 / 1 / 5 | -14.8538 to +0.6485 |
| True auxiliary / shuffled | 0 / 2 / 4 | 0 / 2 / 4 | -5.2828 to +5.0921 |
| Reconstructed control / original | 0 / 0 / 6 | 0 / 0 / 6 | exactly0 |

The two original/control comparisons coincide by successful reconstruction;
they are not two independent pieces of evidence. The point range is neither
a pooled gain nor a confidence interval. The primary is positive-disagreement
easy-harm MSE, not trajectory ADE/FDE or easy-case degradation.

The single positive full-input assignment is P0/C1:+0.2051%,CI[+0.0391,+0.3676].
P0/C2 is adverse:-14.8538%,CI[-29.1706,-0.5371]. All assignments remain in
[results.md](results.md) and the [figure](strong_cap_auxiliary_contrasts.svg).
Full all-harm MSE has1 positive/1 negative/4 overlap;positive-disagreement
top10 capture has0/0/6;mean-cost coverage-error reduction has0/1/5.
Thus the failed guard is substantive, not only an inconclusive primary.

## What Was Learned
The auxiliary still learns event information: full-input cap-probability
AUROC has a descriptive median0.8462 across72 dependent views, and69/72
have better log loss than the fitting-only prior. This is a separate event
diagnostic, not the cost-score ranking measure or a deployment gate.
It does not establish that predicted error magnitudes are better.

Full-input fitting primary MSE is worse in52/72 views versus cost only.
There are29 fit-negative/held-negative and14 fit-positive/held-negative
views. These counts support investigating both optimization tradeoffs and
transport, not explaining everything as held-scene overfitting. They do not
identify a gradient-level causal mechanism. The prior base-design confound
has been controlled here; the individual reasons for the earlier failure
have not been decomposed.

## Limits and Decision
Average three seeds within each of four localities, then3000 paired locality
resamples. Assignments overlap and source outcomes have prior development
exposure. No multiplicity adjustment or independent confirmation is claimed.
Full/motion changes forecasts and populations, not only input features.
Three motion-only event-ranking views remain not_estimable.

Keep the original estimator and deployment unchanged. Independent roles
remain unopened. Stage5C/SMC remain off. This is not submission readiness,
true3D, foundation, metric, seconds-level, human-gold or physical-safety
evidence. Obs8/pred12 are native annotation steps in detector-image pixels.

Full checkpoint/readout replay is recorded separately in verification.json
when complete; training completion alone does not assert replay completion.
