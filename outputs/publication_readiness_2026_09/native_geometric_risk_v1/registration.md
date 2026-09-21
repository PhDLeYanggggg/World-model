# Geometric Protected Risk: Fixed 2x2 Source Experiment

## Material Passport

2026-09-21. Prospective registration after the disclosed training-only conditional
support audit. Four previously explored source scenes; three seeds17/29/43.
Original val/test/main/external/bookstore stay closed. No independent calibration,
confirmation, threshold search, model selection or deployment in this experiment.

## Hypothesis

For complete requested futures, let Z=I(CV_ADE=0) and let D be the known native
ADE between the frozen neural and CV rollouts. Protected harm H0=D*Z exactly.
Learning P(Z|past) and multiplying by known D may be better conditioned than
learning rare protected-harm magnitudes directly. Exact baseline predictability
can use labels from candidate-agreement rows as well. The triangle identity is
elementary; novelty and safe intervention do not follow from the identity alone.

## Fixed Factorial

All arms use the same 361-input/64-GELU/single-logit model, 23,233 parameters,
matched initial states and batches. Base arms pad four extra features with zero;
kinematic arms use maximum past velocity deviation from the last velocity,
mean past acceleration magnitude, mean and maximum past CV backcast residual.
All use the previous357causal features. Extra values use only8observations and
past timestamps; no future labels/masks, scene ID or goals are inference inputs.

Arms: base_event, base_geometric, kinematic_event, kinematic_geometric.
Event arms minimize BCE for Z. Geometric arms add ((sigmoid(logit)-Z)*D/s)^2,
where s is the complete training-only mean CV ADE, identical across arms.
This is native protected-cost MSE, not a ratio to a possibly zero CV error.
Mean/std fit complete training rows with equal training-scene weighting.
Partial labels remain NaN. On candidate-agreement rows, Z is still learned;
reported intervention event/cost becomes zero through D=0, not a false Z label.
No class weighting, rare-row oversampling or post-hoc threshold adjustment.

Four excluded sites x3seeds x4arms =48fits. Each3,000updates/batch256:
144,000updates,36,864,000draws. AdamW lr.001/wd.0001,clip5,checkpoint500,
heartbeat100. Coupa/seed17/base_event100-update pilot counts toward budget.
Native arm64 CPU4/inter-op1/workers0. All48endpoints before new held readout.
Resume full RNG/optimizer/normalizers. A smaller previous2-output head is a
cached historical control, not a parameter-identical arm of this factorial.

## Frozen Readout

Use unchanged MSE gain predictions and report all fixed rules:
positive net gain plus q<=.01; the previous MSE strict condition plus q<=.01;
and common-count allocations from the strict eligibility pools. q<=.01 is a
diagnostic score cut, not a1%population guarantee or tolerated empirical harm.
Controls: net-only, MSE strict, past-stop plus MSE strict, old protected-risk
net/strict. No broad general-harm guard constrains the new matched budget.

Common K per scene/seed is min(previous asymmetric-strict request, capacity of
all4new strict pools, MSE strict, stop-strict, old protected-strict). Each uses
the same frozen net-gain ordering with query-ID tie breaks. Never force unsafe
rows into a pool. Report capacity loss, identity coincidences and missing labels.
The matched rank comparison is offline, not an online deployment algorithm.

Factorial contrasts: geometric-minus-event at base/kinematic inputs;
kinematic-minus-base at event/geometric losses. Evaluate strict and matched
rules without choosing a winner. Report full results, native ADE/FDE, complete
future, exact-zero harm, frozen positive-easy/hard diagnostics, tails/worst scene,
intervention rate, event Brier/AUROC/AUPRC/ECE and native protected-cost MSE.
Paired physical-scene bootstrap3,000times after seed-error averaging. Multiple
contrasts are exploratory; no confirmatory significance or best-seed selection.

Exact zero-reference protection is unchanged. Any harmed complete zero-CV query
fails it; missing labels are unknown. Positive-easy trainingq25 remains diagnostic,
not a newly redefined formal gate. Eight observed/twelve future annotation steps;
no seconds, meter, true3D, foundation or human-gold claim. Stage5C/SMC off.

## Verification

Bind parent analysis, immutable source code/config, nested views, frozen gain
and prior-risk scores. Check distance identities on training labels, causal
feature extraction, unknown handling, exact resumed optimization and matched
draws. Rebuild every normalizer and replay all48probability endpoints. Preserve
all failed arms and both losses; no changed scientific rule after readout.
