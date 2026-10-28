# Why the Matched Decision-Risk Trial Does Not Advance

## Evidence Source

This is a post-readout descriptive check of already saved TRAIN monitors, not
another model evaluation or a hyperparameter search. The fixed128-query monitor
uses known TRAIN labels and the registered sampling seed. It is not all TRAIN,
independent calibration, a new dataset or an independent risk guarantee.
Every receipt and checkpoint is checked against the complete216-fit freeze.

## What Is Established

At the fixed2,000-update endpoint, task-risk still has positive soft easy-risk
constraint excess in **61/72** monitors and
positive all-risk excess in **38/72**. Its median
soft easy excess is0.000111244 in normalized cost units.
Its median duals are[6.963614463806152, 0.6696987152099609]; maximum duals are[12.591381072998047, 3.067976474761963].
No fit reaches the registered100 dual cap. Thus cap saturation is not an observed
explanation; the empirical constraints were not solved even on most fixed TRAIN
monitors. More epochs or different normalization have not yet been tested.

The separate frozen development readout supports utility improvement over
quadratic, including same-count matching, but not over the task-utility control.
Known easy-risk violations remain58/72 and conservative upper-bound violations
71/72. All72 candidate views underestimate selected easy harm, with median
predicted/actual ratio about0.010576. This is not repaired calibration.

The mean soft action is0.108922 while mean hard intervention is0.283421 across
views. A0.5 diagnostic gives0.283336 disagreement. These soft values are not
calibrated probabilities; this comparison suggests a surrogate/deployment
mismatch, not a proved unique cause. It does not authorize tuning that cutoff.

## Interpretation and Next Falsifiable Step

- Decision-oriented training has a genuine development utility signal under the
  fixed forecast producers. This is not new trajectory dynamics or independent
  cross-domain evidence.
- Adding the current dual penalty gives up utility relative to the unconstrained
  task objective without enforcing the required risk budget. It is not a safety
  solution simply because some violation counts decrease.
- Missing-outcome bounds remain an additional obstruction, but known-label
  violations show that missingness alone cannot explain failure.
- Next isolate TRAIN constraint optimization: verify each constraint's scale,
  gradient and soft/hard agreement on the same fixed TRAIN queries, then register
  one matched repair before training. Candidate repairs include constraint-scale
  normalization or a better hard-rule surrogate. Their effectiveness is not yet
  established; do not choose them using independent labels or a new test sweep.

Keep the current deployment unchanged. Do not promote a utility-only arm after
seeing these results, weaken the2%definition, remove unsupported views, or open
independent confirmation to rescue a failed development policy. Stage5C and SMC
remain off; no metric, seconds, true3D or foundation claim is added.
