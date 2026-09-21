# Protected-Risk Learning: Completed, Not Deployable

2026-09-21. `fresh_run`: 24 real Torch risk-head fits and frozen source readout.
`cached_verified`: native forecasters, nested producer views and MSE gain heads.
`not_run`: independent risk calibration, untouched confirmation or deployment.

## What Was Tested

The fixed experiment separates gain prediction from protecting queries that CV
predicts exactly. General-harm and zero-reference-harm heads have the same 357
causal inputs, 23,042 parameters, initialization, complete-label training rows,
scene-uniform batches and training budget. Future group membership is a label,
not an input. All 24 endpoints were completed before source readout. The fixed
0.01 score cut is diagnostic, not a calibrated risk bound or a tolerated error.

Four excluded physical source scenes, three seeds, 3,000 updates per head:
72,000 updates and 18,432,000 row draws. No unknown-risk label was sampled.
Summed fitting time is 56.279 seconds on cached head inputs, not full-model
training time. The 100-update real pilot is included once in the budget.

## Results

ADE is in annotation pixels within each scene; reported gains are equal means
of within-scene gains versus causal CV after averaging seed errors. Easy here is
the frozen positive-CV training-q25 diagnostic, not a newly chosen gate. Harmed
counts are repeated query/seed instances, not independent events.

| Row-Local Rule | ADE Gain | Scene CI | Hard Gain | Positive-Easy Degradation | Switch Rate | Zero-CV Harmed |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Native predictor without guard | 7.6331% | [5.9632, 9.3022] | 10.6575% | 21.7097% | 88.4135% | 9,000 |
| Frozen MSE net gain > 0 | 5.5738% | [3.3982, 7.8069] | 7.7625% | 7.1111% | 30.2463% | 79 |
| Net gain + past-stop veto | 5.5746% | [3.3976, 7.8083] | 7.7622% | 7.1619% | 30.1767% | 17 |
| Net gain + general-harm guard | -0.0003% | [-0.0012, 0.0004] | 0.0002% | 0.0000% | 0.0015% | 0 |
| Net gain + protected-risk guard | 5.5723% | [3.3950, 7.8076] | 7.7598% | 7.1601% | 30.1651% | 25 |
| Previous MSE strict rule | 1.2919% | [0.5110, 2.3009] | 1.7396% | 0.5857% | 2.7927% | 33 |
| Previous asymmetric strict rule | 0.3404% | [0.0188, 0.8676] | 0.4015% | 0.1015% | 0.8415% | 1 |

The protected-risk rule retains accuracy but fails strict zero-CV protection in
all three seeds: 8, 8 and 9 harmed instances. Worst zero-CV harm is 15.7288px ADE.
It selects 2,176 instances with unknown ADE and 30,511 with incomplete twelve-step
risk labels. Neither group is counted as safe. The simple past-stop control has
fewer zero-reference harms and slightly better ADE, although it too is unsafe.

## Matched-Count Result and Power Loss

The registered common-count comparison requests 4,437 query/seed interventions,
but the general-harm pool admits only eight. Common coverage becomes 0.001517%,
not the requested 0.8415%. These are eight repeated decisions, not eight verified
independent situations. The protected-risk, net-only and stop-veto controls pick
exactly the same eight rows. Their ADE gain is 0.01669%; the protected-minus-general
contrast is +0.01701pp, conditional scene CI [0.00164, 0.03464].

**That contrast does not establish useful protected-risk learning.** It disappears
against the two simple matched controls, comes from almost complete abstention
in the broad-risk arm, and is not positive in every scene. Four selected instances
have incomplete risk labels. Zero observed zero-CV harm at this tiny count is
not a safety certificate. No rejected row was forced into a pool to fill a quota.

## Learning and Verification

Protected-event AUROC ranges from 0.8658 to 0.9853 across scene/seed views.
Its event Brier improves over the fitted constant in 10/12 views, but its expected
cost MSE improves in only 4/12. General-harm Brier improves in 12/12 and cost MSE
in 9/12. High ranking quality does not certify the small low-score intervention
region. The two event tasks differ; their probabilities are not interchangeable.

Mean first-to-final logged batch loss: general harm 1.4331 to 0.5390;
protected harm 1.5110 to 0.05546. Final protected BCE is 0.05395 and normalized
cost MSE 0.001515. These are minibatch optimization diagnostics, not population
error, independently calibrated probabilities or proof of safety.

402 dependency bindings pass. All 24 checkpoints replay 1,054,536 risk-score
rows exactly; all twelve paired samplers and preprocessing views match.
Independent code checks 3,163,608 repeated training-target rows, 132 policy
choice hashes, 792 scalar scene reductions and 48 event-metric records.
116 related tests pass. All required processes have exited normally.

## Research Decision

Keep the native forecast as a developmental predictor, not a newly deployed
model. Do not claim that a separate rare-event probability head solves selective
prediction. The next source-only experiment should first diagnose conditional
low-score errors and training support, then test a registered conditional
harm-severity/support-aware objective against the simple past-motion control.
Keep the predictor fixed, preserve all seeds and avoid a post-readout threshold
sweep. Independent calibration and confirmation need a clean upstream lineage;
they cannot be created by renaming these explored source scenes.

This is useful negative mechanism evidence, not a standalone major result,
CCF-A candidate or evidence that Transformer/JEPA composition is novel. Primary
task remains 8 observed/12 predicted annotation steps. Raw-frame t50 remains
supplementary. No metric, seconds, true-3D, foundation or human-gold claim.
Original val/test/main/external/bookstore roles remain closed. Stage5C/SMC off.

See [all results](results.csv), [failure taxonomy](failure_analysis.md),
[reproduction](execution_notes.md), [replay](verification.json), and
[independent check](independent_verification.json).
