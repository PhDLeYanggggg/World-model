# Failure Taxonomy After the Protected-Floor Diagnosis

| Hypothesis | Current evidence | Interpretation / repair |
|---|---|---|
| Neural candidate has no useful incremental forecast value | Floor/neural oracle 12.89--23.98%; rebased all-ADE positive in 36/36 conditional intervals | Rejected as a complete explanation; oracle is not achievable learned performance |
| Weak default action conceals useful neural decisions | Same switch masks, stronger default; original negative, rebase positive in all 36 | Supported development explanation; make reference action consistent in targets and deployment |
| Wrong or harmful switches are solved by rebase | Zero-CV harm remains in 24 repeated views | Rejected; changed default cannot change an already selected neural trajectory |
| Low utility blocks useful candidates | Positive missed-benefit mass behind utility gate in all 36 | Target/features need examination; post-hoc benefit does not authorize relaxing gate |
| Risk veto is only excessive conservatism | Large missed benefit but harm support remains incomplete | Not established; estimate incremental gain/harm against actual floor, retain budget |
| Positive result is caused only by partial labels | Complete and partial strata both positive | Not supported as sole explanation; incomplete/automatic labels still limit physical claims |
| Many rows establish broad independent safety | Twelve development localities, overlapping windows, four zero-CV rows from one locality | False; source-level calibration and independent confirmation are still required |
| Positive scene-average gain protects every locality | Worst hard-locality degradation 3.0515%; seventeen rebased views have a negative hard-locality point | False; retain worst-locality and tail constraints rather than average-only promotion |

## What Changed, What Did Not

This round adds error accounting, an offline default-action contrast, label
support sensitivity and independently checked arithmetic. It does not train a
new predictor, change any saved neural switch, tune a threshold, open a reserved
scene role, or recertify historical deployment results.

The positive diagnostic is smaller than the unavailable oracle. Depending on
the head, only 0.595--22.039% of incremental oracle benefit is captured. All-event
and easy-event heads solve different targets; their outcomes must not be pooled
to choose an unregistered winner. Min(CV,floor,neural) also contains the benefit
of removing harmful damping and is not all neural contribution.

## Immediate Repair

The next training comparison should change the reference of gain/harm learning
from CV to the actual protected floor. Its label producer must be source-cross-
fitted; the floor itself has learned heads and cannot bypass producer exclusion.
Preserve zero-reference cases, fully report tail/worst-locality costs, and keep
an explicit support-abstention path. Calibrating a combined policy is a separate
step, not something inherited from the original CV-risk threshold.

No threshold is changed by this report. No released veto, new oracle policy,
deployment, Stage5C or SMC is permitted by this diagnostic.
