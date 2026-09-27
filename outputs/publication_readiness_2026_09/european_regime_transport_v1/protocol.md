# Fitting Regime and Easy-Definition Transport

## Material Passport

Registered source-development mechanism experiment. Parent: completed honest
OOF magnitude study, commit8271d0d7. No new forecast or policy, no independent
selection/calibration/confirmation access. Obs8/pred12 native steps, detector
pixels. No metric/seconds,human-gold,true3D,foundation or safety claim.

## Question and Scope

Can the failed magnitude transport be explained by changing the fitting-row
regime, by changing its easy definition, or by an interaction? Earlier
common-event residual relabelling failed and did not retrain the underlying
cost targets. Do not repeat that relabelling or infer causes from native
two-site versus three-site heads alone: both factors changed there.

For each unchanged outer held locality D and fitting localities A/B/C, use
all three choices of omitted fitting locality C. Cross:

| Cell | Gradient rows and preprocessing | Cut-estimation sites | Status |
|---|---|---|---|
| two_cut2 | A/B | A/B | cached native control |
| two_cut3 | A/B | A/B/C | fresh crossed head |
| three_cut2 | A/B/C | A/B | fresh crossed head |
| three_cut3 | A/B/C | A/B/C | cached native control |

All four cells predict D only. In two_cut3, C is excluded from gradient rows
but NOT from label-definition provenance. It is therefore NOT an inner-OOF
head for C and must never supply C predictions for fitting another readout.
No label,cut,normalizer,reference or parameter uses D. The forecast producer
roster remains disjoint from all controller sites. Targets never enter causal
inference features. All new predictions are committed before held scoring.

The regime factor includes row composition,preprocessing and site-balanced
sampling,not only numerical sample size. The cut factor includes its necessary
supervised targets,easy-dependent initialization and RMS loss normalization.
Within a regime, inputs,known rows,sampler draws,fixed batch,non-easy loss
scales and cost normalization must match exactly across cuts. Do not claim
isolation of pure sample count or real-world causality.

## Fixed Training and Controls

144 outer views,3 omitted-locality replicas each,2 crossed heads:864 fresh
heads,1,728,000 updates. Width64,2000 updates,batch256,AdamW lr0.0003,
weight decay0.0001,clip5;seeds17/29/43. CPU4,interop1,workers0;checkpoint and
heartbeat200. No tuning,early stopping,checkpoint selection or auxiliary task.
Reuse432 native two-site and144 native three-site controls with hashes.

Before full training, a real2000-update native bridge must exactly reproduce
the first three-site cost-only control with the shared training API. It is
one additional diagnostic head,not one of the864 crossed heads. A200-update
crossed pilot resumes to its fixed2000-update budget. Run support checks first.
No downgrade for slowness. Preserve10GiB disk and all existing assets.

Apply the identical parent cost-only two-slope readout to every cell and keep
raw identities. It is frozen,not refit to a new target or held labels. This
tests readout transport,not optimal cell-specific recalibration. Do not mix
denominator columns from different cells. Keep all four predicted moments;
the native three-site control must match its stored scores exactly.

## Evaluation

Primary targets use the unchanged outer three-site easy cut. Score the same
frozen predictions against two-site easy-cut labels as a clearly separated
diagnostic only; never use those held targets for training or selection.
Retain expected easy-harm MSE on positive envelopes and guards:top10 harm
capture,coverage-log error,all-envelope H_all MSE. These are not FDE/ADE,
easy degradation,conformal coverage or physical safety.

For raw and scaled modes report: cut3 vs cut2 at two and three row sites;
three vs two row sites at cut2 and cut3. Also report scaled vs raw in all
four cells. For each replica compute each paired metric contrast first;
average the three replicas within held locality,then the three seeds.
Use3000 paired resamples of four held localities per assignment. Six source
assignments overlap;CIs are descriptive and unadjusted. No replica/window
pseudoreplication,no missing-cell deletion,no favorable-assignment selection.

Report cut-by-regime difference-in-differences for easy-harm MSE:
100*((MSE_two_cut2-MSE_two_cut3)-(MSE_three_cut2-MSE_three_cut3))
/MSE_three_cut3,computed per replica before averaging,for both raw/scaled.
A positive value means switching to cut3 helps the two-site regime more.

## Decision Rules

This is a mechanism diagnostic,not an advancement/deployment trial. A cut
explanation requires all six full-input primary-cut CIs favorable at both
row regimes without negative/missing registered guards. A fitting-regime
effect requires all six CIs favorable at BOTH fixed cuts; it still includes
preprocessing/sampling. Size-matched magnitude transport is supported only
if native two-site scaled/raw primary has6/6 positive CIs with no guard harm.
An interaction requires six same-signed nonzero CIs;otherwise report mixed or
unresolved. No single failed screen proves absence of useful information.
No mechanism result promotes a policy or opens independent roles.

If effects remain mixed,retain that result rather than attributing the
original auxiliary failure to a single cause. A positive cost-only mechanism
is not proof of the auxiliary's failure mechanism. Stage5C and SMC stay off.
