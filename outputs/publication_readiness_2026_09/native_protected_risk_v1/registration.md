# Separate Protected-Risk Head: Fixed Source Experiment

## Material Passport

2026-09-21. Prospective fixed training/readout design, following disclosed
post-hoc source diagnostics. All four sites remain design-exposed. No independent
calibration, confirmation, threshold search, final-role opening or deployment.

## Hypothesis and Controls

Net-gain ranking retains more ADE benefit than a ratio at the same count, but
zero-reference harms persist. A separate zero-reference risk target may preserve
useful allocation better than merely increasing all-harm regression penalties.
This is a targeted learning hypothesis, not an architecture novelty claim.

Reuse the same frozen native forecasters, nested cost views and MSE gain head.
Fit two equally sized neural risk heads per outer source site/seed:

- all_harm: predict P(H>0 | past) and E[H | past].
- zero_reference_harm: predict P(CV_ADE=0 and H>0 | past) and
  E[H * I(CV_ADE=0) | past].

H is positive ADE increase of the candidate over CV. Both targets require all
twelve future annotation labels. Partial/absent futures remain unknown, never
negative event labels. Future CV error and harm are training/evaluation targets
only. The same support, normalizers, scene-uniform batches, initialization and
budget are used for both arms. Complete zero-reference harm has 1,999-2,677
training rows per view; those repeated overlapping counts are not independent
event support or sufficient risk-calibration evidence.

## Inputs, Loss and Budget

Inputs: the existing355causal features plus two exact past flags: zero last
observed displacement and stationary entire eight-step history. Both arms use
all357features. Candidate-equals-CV imposes exactly zero predicted event/cost,
because that outcome identity is known without future labels. No future mask,
endpoint, goal, oracle decision or outer-site ID is an input.

Width64 GELU head, sigmoid event output and Softplus cost output. Minimize binary
cross-entropy plus squared cost error after dividing cost by the shared complete
training mean CV ADE. No class oversampling or positive-class weighting. Normalizers
use supported training rows only. Initial hidden weights match; final weights0,
event probability0.5 and normalized cost1 for both arms. Outputs are not assumed
calibrated across scenes, and expected cost is not a confidence bound.

Three seeds17/29/43, four excluded scenes,24heads,3,000updates/head=72,000updates.
Batch256,AdamW lr0.001/weight_decay0.0001,gradient clip5. Checkpoint500,
heartbeat100. A100-update coupa/seed17/all_harm pilot counts toward the final
budget. Native arm64CPU4/inter-op1/workers0, local; use existing runtime/forecasts.
All24fixed endpoints must complete before any new held-source quality readout.

## Fixed Diagnostic Readout

Measure event AUROC/AUPRC/Brier/ECE and expected-cost error on complete futures,
with train-constant controls. Report all-harm scores against their own target;
against zero-reference events they are a broader-task ranking control, not a
calibrated probability estimate of that different event. Keep unknown selected
outcomes, native ADE/FDE, tails/worst scene, hard q75/positive-easy q25 diagnostics
using the prior frozen training cuts, and strict zero-CV absolute harm.

Keep MSE predicted net gain fixed. Four input-only eligibility pools:
net gain>0; plus zero-last-observed-step veto; plus all-harm event score<=0.01;
plus zero-reference-harm event score<=0.01. Candidate-equals-CV is never an
intervention. The0.01cut is a fixed diagnostic score threshold, **not** a new
allowed failure rate, independent calibration or relaxation of strict protection.
Report each row-local rule without a count cap.

For matched allocation use K requested by the prior underharm4 strict rule per
scene/seed. Common K is the minimum of that request and all four pool capacities.
Select the largest predicted net gains from each pool at exactly common K. Report
requested/common K and capacity loss; never force rejected risk rows into a pool
to fill a budget. If K collapses, report loss of power, not safety success. These
matched batch controls are offline diagnostics, not online causal deployment.

Primary paired mechanism contrast: protected versus general risk guard at common
K, reporting ADE gain **and** zero-reference harm. Compare net-only and simple
past-stop controls. Report fixed parent MSE/underharm4 strict policies as context.
Preserve all seeds/results; no favorable operating point selection after readout.
Bootstrap four physical scenes3,000times after averaging seed errors. Overlapping
queries are not independent evidence. No method promotes unless actual safety
and independent evidence requirements are met; this experiment cannot supply the
latter merely by fitting exclusion.

## Scope and Verification

Hash-bind parent forecasts, heads, nested views, code/config and identities.
Test unknown-target handling, exact-zero semantics, causal input flags, equal
initialization, matched sampling, score bounds, safe count capacity and exact
checkpoint resume. Rebuild fit-only preprocessing and replay all risk scores.
Original val/test, main/external and bookstore stay closed. Raw labels are not
human gold. SDD annotation pixels/steps only; no metric/seconds/true3D/foundation
claim. No Stage5C execution or SMC. Keep private caches/checkpoints out of Git.
