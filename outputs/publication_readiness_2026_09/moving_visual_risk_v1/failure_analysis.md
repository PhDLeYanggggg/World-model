# Did Past Images Repair Intervention Risk?

The registered motion component screen is False.
The original complete motion-policy risk check is False.
The primary paired lower contrast is +0.4615pp [-1.0949, +2.2936].

## Separate Questions, Not Interchangeable Successes

1. Image availability/alignment and exact recovery establish engineering.
2. Lower training loss measures fit, not held-source policy generalization.
3. Motion-minus-quality tests motion information; bounds-only success cannot replace it.
4. Positive mean gain cannot replace every required original risk/support check.
5. Exposed development and risk-head seeds are not independent new test evidence.

## Matched Training Objective

| Contrast | Lower final minibatch objective pairs | Mean final objective difference |
|---|---:|---:|
| image_bounds minus no_image | 6/6 | -0.0767 |
| motion_geometry minus quality_geometry | 6/6 | -0.0065 |
| motion_geometry minus no_image | 6/6 | -0.1016 |

These are matched final training minibatches, not validation MSE or evidence
of calibrated uncertainty. All24traces remain in the loss CSV.

## Actual Held-Source Prediction Errors

Motion lowers observation Brier from0.049945to0.038765,
but conditional cost generalization is worse in common history-scale units:
harm MSE 0.001455to0.003445,
easy-harm MSE 0.000328to0.000608,
reference MSE 0.120155to0.133059.
All five conditional errors worsen versus quality_geometry despite improved
final fitting loss in all six pairs. Better observability prediction is
not a reliable forecast-harm estimate. This is a mechanism diagnostic,
not proof that images contain no useful information or a calibration guarantee.
[Complete calibration audit](calibration_analysis.md).

## Actual Tail Case

Sourcemoving_source_component_073,recordingmoving_scene_34__2,frame11,risk seed17.
Selection is worst signed-easy recording, then largest positive easy-harm
window. It is explicitly posthoc and not representative sampling.
The quality control retains the floor with ADE5.2922pixels.
Motion selects action6with ADE96.4910,
adding91.1988pixels,while predicting harm
0.3323and easy harm0.2623.
The window has12/12labels. This failure is not merely
an unknown-future completion penalty. A pointwise predicted harm need not
satisfy the2%budget alone because the joint decision pools query budgets.
[Frozen scalar case](failure_case.json).

## Risks And Representation Limits

Past-image transforms can fail on weak texture, parallax, independently moving
background content and masks. Unsupported transitions remain explicit, not
zero-motion certificates. Full-source acquisition retains those failures.
The small cost head does not learn pixels or retrain the trajectory actor.
Any retained actor errors and source shifts can remain after better context.
Pointwise fit and per-record joint deployment are different tasks.

Error/reference ratios can exceed100%;they are not probabilities or
physical collision risk. The worst signed easy loss is not a mean.

There is no after-the-fact threshold search, dropped difficult recording,
identity certificate or reopening of reserved outcomes. A failed result
requires model/data repair under a new prospective experiment, not a
relaxed denominator or replacement primary metric. The whole project is
not submission-ready; no positive result here establishes novelty by itself.
