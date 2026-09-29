# Selected Easy Harm Is Underestimated Before Transfer

## Evidence Status

fresh_run local diagnostic and exact per-packet numerical replay; all 288 packet
bytes match the committed remote manifest. Frozen inputs and checkpoints are
cached_verified. CREATE numerical replication is not_run: the job was last
observed pending and subsequent access timed out. No new model was trained.
This is developmental evidence on the same 12 opened localities, not independent
selection, calibration or confirmation. All source-role exclusions remain intact.

## What the Decomposition Shows

At the same per-query intervention count, the nonlinear head predicts all-risk
of 0.5940% on its training-source selections and 0.5187% on internal transfer.
Realized selected positive-harm ratios are 2.8062% and 5.6317%, respectively.
For easy risk, predicted values are 0.3218% and 0.3111%, versus realized values
of 7.9003% and 14.0113%. These are equal-locality means of selected positive-harm
ratios, not whole-population ADE degradation. The 2% selected-risk budget is
unchanged; net benefit cannot cancel positive harm for this estimand.

The exact decomposition uses the same actual selected reference in every term:

`actual risk = 2% + predicted excess + harm-underestimation + reference-overestimation`.

For nonlinear transfer easy risk, in percentage points:

`14.0113 = 2 - 1.1642 + 13.8401 - 0.6646` (rounded).

The easy reference term is negative, not positive. Thus reference overestimation
does not explain the easy violation in this signed accounting: harm
underestimation dominates. The corresponding fitting harm error is already
+7.7106 pp. This is not solely an unseen-locality phenomenon.

For all-case nonlinear transfer, harm error contributes +4.4607 pp and reference
error +2.8275 pp, offset by predicted excess -3.6565 pp. Both cost components
matter here. Repairing only the reference head cannot account for the easy result.

## Why a Good Global Fit Can Still Be Unsafe

On the complete eligible fitting pool, nonlinear predicted all-risk is 12.8058%
versus actual 13.4859%. That comparatively close aggregate is not retained on the
selected subset: predicted 0.5940% versus actual 2.8062%. These scopes differ in
composition and weighting; this is descriptive selection-associated error, not
proof of a unique causal selection-bias mechanism or population miscalibration.

The parent experiment already showed lower training loss and a small matched-ADE
gain without reliable risk prediction. This decomposition rules out the simple
story that only held-locality transfer broke an otherwise safe fitting selector.
It does not prove whether model capacity, optimization, omitted context, label
noise or source-specific conditional shift is the dominant causal mechanism.

## Not Just the Most Extreme Harm

The severe-tail cutoff was fixed from each original training source: the 95th
percentile of positive harm. In matched nonlinear transfer views, the conditional
equal-locality tail shares are 16.06% of all harm and 22.81% of easy harm. Only
10 of 12 localities have fully defined tail-share summaries; the other two remain
undefined, not zero. The corresponding positive-underestimation shares are
16.49% and 22.84%. These defined summaries do not support treating this as only
a handful of training-defined extreme-tail events. A new tail-only weighting
sweep is not justified by this result.

## Failure Taxonomy

1. Selected risk is already optimistic on fitting sources, especially easy harm.
2. Transfer amplifies the mismatch: nonlinear easy risk rises from 7.90% to 14.01%.
3. All-risk errors include both harm and reference errors; easy errors are
   dominated by underestimated harm, with an offsetting reference term.
4. The training-defined extreme tail explains a minority in the defined
   matched-transfer summaries; the remaining error is not established harmless.
5. Unknown-label matched interventions remain 7,094 affine and 8,727 nonlinear
   repeated occurrences. Their true risk and easy membership are unobserved.
6. Repeated views and overlapping trajectories are dependent, and the locality
   pool is development-exposed. No independent safety or deployment claim follows.

## Next Controlled Repair

Test checkpoint selection on a held recording/block from the original training
locality, never on the transfer locality. Train the same causal nonlinear head
for its full fixed budget; compare the final checkpoint with a checkpoint selected
using only source-internal validation. Keep training inputs, selected-reference
denominator, 2% risk budget and action logic fixed between those two policies.
Use a purged temporal split only if a locality lacks multiple recordings, with
the full observed/future label footprint separated. Preprocessing must use only
the optimization portion. This is a test of validation-selected cost learning,
not a promise that early stopping fixes safety. The original final-step control
must be retained even if it loses.

The contrast addresses an untested selection mechanism after the prior capacity,
loss, occurrence and tail controls. It must report held-source signed cost MSE,
selected harm, easy ADE degradation, unknown coverage and matched-count utility.
If source validation improves global loss but not selected risk, do not promote
the model or loosen thresholds; the conditional decision-cost problem remains.

No deployment change. Obs8/pred12 at stride12 raw frames, image-local detector-
silver only; no metric/seconds, human-gold, true3D or foundation claim. Stage5C
and SMC remain disabled.
