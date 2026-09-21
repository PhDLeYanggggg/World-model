# Geometric Risk Factorial: Cost Fit Improves, Policy Does Not

## Material Passport

2026-09-21. `fresh_run`: 48 registered Torch fits, source evaluation and replay.
`cached_verified`: frozen native forecasters, gain heads, old risk scores and
nested producer views. `not_run`: independent calibration, untouched confirmation,
new deployment. Registration `91581b9a` was pushed before fitting. All four source
scenes remain design-exposed; no original val/test/main/external/bookstore role
was opened. This is a completed development experiment, not a publication gate.

## Experiment Completed

Three seeds, four excluded physical scenes, four feature/loss arms. Each has
23,233 parameters and 3,000 updates: 144,000 updates and 36,864,000 row draws.
No unknown-risk labels were sampled. The 100-update pilot is included once.
Summed cached-head fitting time is 115.896 seconds; no new forecaster or latent
extractor was trained. All 48 endpoints finished before new held-source readout.

The model learns P(CV_ADE=0|past). Native protected cost is this probability times
the known distance between candidate and CV rollouts. The geometric-loss arm
adds native protected-cost MSE; the feature arm adds four past-only CV-consistency
statistics. Candidate-agreement rows supervise baseline predictability, while
their intervention risk remains exactly zero. Labels with incomplete futures
stay unknown. The identity is elementary, not a novel safety theorem.

## Fixed Strict-Rule Comparison

All gains are versus causal CV using the registered equal-scene native ADE
summary. Positive-easy q25 and hard q75 remain frozen diagnostics. Zero-CV counts
are repeated query/seed instances. They must not be called independent events.

| Rule | ADE Gain | Positive-Easy Degradation | Hard Gain | Switch Rate | Zero-CV Harmed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing MSE strict | 1.29194% | 0.58569% | 1.73959% | 2.79270% | 33 |
| Existing MSE strict + past-stop veto | 1.29190% | 0.60943% | 1.73956% | 2.76444% | 0 |
| Previous protected head + MSE strict | 1.29071% | 0.60950% | 1.73841% | 2.76273% | 5 |
| Base/event geometric factorization | 1.29084% | 0.60567% | 1.73842% | 2.75970% | 3 |
| Base/geometric-loss factorization | 1.29067% | 0.60567% | 1.73820% | 2.75913% | 3 |
| Kinematic/event factorization | 1.29061% | 0.60567% | 1.73821% | 2.76178% | 3 |
| Kinematic/geometric-loss factorization | 1.29063% | 0.60567% | 1.73822% | 2.76178% | 3 |

Every new strict arm harms one complete zero-CV query in each seed, all in
deathCircle with a zero last observed displacement. Maximum harm is 12.5623px
ADE. Thus the rare moving-zero-CV support problem is not the sole explanation
for strict-rule failure. Neither new loss nor history features solve protection.

Without the existing MSE strict gate, the four new factorized rules have
5.5720%-5.5728% ADE gain but 7.1521%-7.1533% positive-easy degradation and 22
zero-CV harmful instances each. Retaining average benefit is not safe deployment.

## Useful Simple Control, Still Not Certified

The pre-registered past-stop-plus-MSE-strict control gives 1.29190% ADE gain,
conditional scene CI [0.51102, 2.30072]. Seed gains are 1.52165%, 1.29118% and
1.06287%. Mean gains by scene are coupa0.23511%, deathCircle2.80531%,
gates1.34024%, hyang0.78694%. No observed complete zero-CV query is harmed in
any seed; positive-easy diagnostic degradation is0.60943%.

This is a useful nonzero-intervention development reference, not a guarantee.
It still selects 283 query/seed instances without any ADE label support and
3,064 without complete twelve-step risk labels. These outcomes remain unknown.
Its hard gain is only1.73956%, not the historical10% target, and it has no
independent risk calibration or confirmation. It is not promoted to deployment
or presented as the paper's new learned contribution.

## Matched Coverage and Factorial Effects

All 4,437 requested query/seed switches are retained at common0.84151% coverage,
unlike the previous broad-risk control's collapse to eight. Ordinary MSE strict
net ranking gives0.951225% ADE gain; base/event, previous-risk and stop controls
give0.951214%; other new arms give0.951089%. All matched arms have zero observed
complete zero-CV harm, including the simple MSE control without the new risk
head. The latter still selects120 unknown-ADE and1,181 incomplete-risk instances.

New factorial strict-rule ADE contrasts are all smaller than0.00024percentage
points in magnitude. Base geometric-minus-event is-0.000169pp; kinematic-minus-base
under geometric loss is-0.000035pp. The tiny positive geometric-minus-event effect
with kinematic inputs is+0.000027pp. Matched geometric and kinematic effects are
zero or slightly negative. Their conditional bootstrap records are retained in
analysis.json; these are not meaningful practical lifts or independent tests.

## Cost-Fit Success and Its Limit

All48new endpoints have lower held-source protected-cost MSE than their
corresponding prior direct-magnitude risk head. Equal-view mean MSE ratios to
that old head are0.5810,0.5779,0.5423,0.5398 for base/event, base/geometric,
kinematic/event,kinematic/geometric. These ratios are descriptive paired summaries,
not pooled native costs or confidence intervals. The old head differs in target
and parameterization; only the new2x2 factorial is capacity/budget matched.

Within the factorial, adding geometric loss improves cost MSE in7/12base views;
adding kinematic inputs improves it in9/12geometric-loss views. Those improvements
do not translate into a useful accuracy-harm frontier. Global severity fit and
event AUROC must not substitute for reliable decisions in the selected region.

## Verification and Next Action

424 source bindings pass. All48checkpoints reproduce2,109,072probability rows;
36paired sampler comparisons pass. Independent scalar verification checks240
selection hashes,1,440scene reductions,6,327,216repeated training-target rows and
96event metric records.123relatedtests pass. Required processes all exit normally.

Retain the simple causal-stop/strict rule as a research control and all failed
learned arms. Do not spend the next round on another nearly identical risk-head
loss by default. First verify the control's conditional/missing-outcome support
and scene-query alignment, then register any source-only joint-versus-independent
comparison or independently calibrated extension with clean producer lineage.
No threshold sweep on these displayed outcomes and no relaxation of strict
zero-reference protection. The long-term goal and contribution remain unmet.

Eight observed/twelve predicted sampled annotation steps, native SDD pixels;
raw-frame t50 supplementary. No seconds, metric, true3D, foundation or human-gold
claim. Stage5C/SMC off. See [full table](results.csv), [failures](failure_analysis.md),
[reproduction](execution_notes.md) and [independent checks](independent_verification.json).
