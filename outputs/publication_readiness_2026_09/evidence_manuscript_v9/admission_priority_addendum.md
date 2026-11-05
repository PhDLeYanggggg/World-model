# Fixed Error-Replay Objective: 5 November 2026

This addendum records a completed negative model-side control. The paper PDF
is not rebuilt by this note; old failures and unavailable primary comparisons
are retained. Original risk tolerance and support requirements are unchanged.

We trained six neural risk heads, comparing ordinary continued MSE with a
fixed weight-10 emphasis on the warm model's observed false-safe fitting
errors. Each pair starts from identical kinematic weights and shares the
architecture, source roles, normalization, source-query sampler and 2,000-step
budget. The warm error mask is training supervision only. All unknown outcomes
retain their previous bound-aware anchor. Inference uses neither the error
mask nor future labels. The treatment is standard fixed hard-example replay,
not a novel dynamics method, calibrated expectation or safety guarantee.

The target mechanism operates: fixed-error MSE is 57.96-89.21% lower than
the matched continued-training control. Across repeated fitting contexts,
650 of 1,002 warm false-safe easy signs are corrected, but 333 new errors
appear and 352 old errors persist. These are resubstitution output-sign
counts, not independent observations or whole-policy safety outcomes.

On seven exposed held SOURCE units, known ADE gain over the floor is 3.0323%
for priority training, 3.6006% for continued MSE and 3.9418% for the warm
kinematic head. The matched priority contrast is -0.5683 percentage points,
with missing-outcome completion bounds [-0.8147, -0.3218], not a sampling CI.
Hard gain falls from 4.6257% to 3.8505%. The treatment intervenes on 13.5856%
of rows in the source-macro summary, versus 15.0144% for the control.

Both new arms have 0/17 complete risk/support passes, 14 known all-risk
violations, 13 easy violations and two undefined selected-easy views. The
treatment's worst defined signed easy-net degradation is 22.6698%. Lower
intervention and improved fitting error do not meet the original 2% contract.
Only one of three fixed contexts meets the prospective expansion condition;
the variant is not promoted and no favorable context is selected as a winner.

The experiment identifies a failure of this fixed replay repair, not the
impossibility of risk learning. The error mask is based on resubstitution;
each fitting context has only two or three independent sources. Representation
transport, limited source diversity and incomplete optimization remain distinct
explanations. In one context the treatment even has a larger combined weighted
fitting objective than the MSE control, so optimization is not claimed solved.

All six actual neural fits, the independent full refit, both resume checks,
all three inference/decision/metric blocks and parent controls reproduce
exactly. One initial zero-target synthetic fixture failure was corrected and
recorded before any real training. This is one-producer-seed development, not
new independent confirmation. The original 24-group primary stays NA.

Evidence: [complete report](../moving_admission_risk_v1/README.md),
[mechanism and limitations](../moving_admission_risk_v1/result_interpretation.md),
[protocol](../moving_admission_risk_v1/protocol.md),
[full fitting losses](../moving_admission_risk_v1/full_fitting_losses.json),
[replay](../moving_admission_risk_v1/verification.json).

Native obs8/pred12 image pixels. No verified metric/seconds, human gold,
true 3D, foundation, deployment or submission-ready claim. Stage5C and SMC
remain disabled. No evaluation-rule change or identity-fallback certificate.
