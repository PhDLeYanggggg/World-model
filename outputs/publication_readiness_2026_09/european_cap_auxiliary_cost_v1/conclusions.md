# Cap-Event Auxiliary Cost Study: No Promotion

## Completed Evidence
Fresh native-Torch fitting completed432 heads and864000 updates: three matched
arms, three seeds, two forecast families and all source assignments. The
100-update pilot resumed to2000, not a separate counted model. All predictions
were frozen in f4bc6c2d before144 source-held readouts and1152 direct component-
MSE checks. Three seeds are averaged within each locality before3000 paired
bootstrap draws of four localities. Six assignments overlap; all are retained.

## Registered Primary Result
Full-input easy-harm MSE contrasts, positive favors the auxiliary model:

| Comparator | Positive / negative / overlap intervals | Six point estimates: range (%) |
|---|---|---|
| Matched cost-only | 2 / 0 / 4 | -5.86 to +14.35 |
| Frozen original | 0 / 3 / 3 | -37.74 to +0.68 |
| Matched shuffled auxiliary | 1 / 1 / 4 | -13.58 to +14.27 |

These ranges are not confidence intervals or a pooled overall improvement.
Against the original, the three adverse assignment results are:

| Producer / controller | Point (%) | 95% locality CI (%) |
|---|---:|---|
| 0 / 2 | -19.72 | [-33.56, -5.89] |
| 2 / 0 | -4.77 | [-10.20, -0.89] |
| 2 / 1 | -37.74 | [-104.40, -3.54] |

The primary gate, tail/coverage/all-harm guards and true-versus-shuffled gate
all fail. Against the original, two top-tail and two all-harm MSE intervals
are adverse. There is no supported auxiliary cost contribution under the
registered rule. No model or favorable assignment is selected after readout.

## What Improved, and What Did Not
Full-input learned cost ranking improves AUROC in all six contrasts against
both new matched controls. Top10 easy-harm capture improves in4/6 versus
cost-only and6/6 versus shuffled. This is real source-development ranking
evidence, not better magnitude calibration or a trajectory/policy gain.
Against the original strong model, tail gains have0 positive/2 negative/4
overlapping intervals. Beating a weak new control is insufficient.

The auxiliary event head's median supported AUROC is0.83635 across72 dependent
full-input views. That descriptive median does not make72 independent tests.
Motion-only easy-harm MSE has0 positive/1 negative/5 overlap versus cost-only,
and1 positive/1 negative/4 overlap versus original. Three motion event views
have no rank-estimable positive support and remain explicitly unavailable.

## Decision
Do not deploy this auxiliary model. Existing deployment stays unchanged.
No new trajectory forecasts or scene-joint policy were evaluated. Independent
selection, reserved calibration and confirmation remain unopened. Existing
exposed historical scores are not restored as independent tests by this run.
The result does not establish a world-model contribution or submission readiness.

The next step is a matched original-estimator reconstruction/control and
fitting-only magnitude/transport diagnosis, not another threshold sweep.
Keep the original architecture, cost objective and training support explicit
before attributing a future repair to auxiliary supervision.

## Verification and Units
Checkpoint/result replay is documented separately in verification.json when
it exists and passes. It is not independent retraining or confirmation.
The scoped verifier retains reports, figure, diagnostic and code hashes.
Obs8/pred12 native annotation steps; detector-image pixels. No metric,
seconds-level, human-gold, physical-safety, true3D or foundation claim.
Stage5C and SMC remain off. The long-term research goal remains active.
