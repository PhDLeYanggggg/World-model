# What This Controlled Training Round Established

## Result
I completed 72 new Torch risk-head fits, 144,000 optimizer updates and 432
frozen source-held decision views. This round does not justify a deployment
change. It supports query-level allocation as a useful development direction,
but does not support the added selected-group loss as an accuracy improvement.

The forecasts and old utility scores were fixed. The two objectives shared
initialization, training examples, preprocessing, component scales and budget.
Three seeds were averaged within locality before 3,000 locality-bootstrap
resamples. Each source assignment has four held localities; the six assignments
overlap, and these localities were historically opened development data.

## Positive Evidence, With Its Comparator
- Full-pair selected-joint versus selected-dual improves all ADE by
  **1.4882% to 2.8571%** across the six three-seed source assignments. All six
  intervals are positive. Hard-subset gains are **1.5225% to 3.6262%**, also
  with six positive intervals. This compares joint allocation with independent
  all/easy gates, not with the strongest old policy.
- Against a control with exactly the same number of interventions per query,
  full-pair selected-joint gains **0.0214% to 0.8179%**: four intervals are
  positive and two overlap zero. This is partial ordering evidence. The control
  matches counts, not predicted risk, and does not isolate neural scoring from
  the rest of the policy.
- Full selected-joint preserves net easy error in all 18 settings. However,
  complete observed risk passes only **9/18** settings. Net preservation does
  not establish control of the positive harm accumulated on worsened examples.

## Negative Evidence
- Selected-joint versus mean-joint has all-ADE changes **-0.4997% to -0.0324%**.
  Five intervals are negative and one overlaps zero. The same comparison on
  motion-only forecasts is **-0.7078% to -0.0752%**, again five negative
  intervals. The added selected-group objective is not an accuracy contribution.
- Full selected-joint versus the frozen old neural rule changes all ADE by
  **-1.1316% to +0.3148%** after three-seed averaging. There are no positive
  intervals, three negative intervals and three overlapping intervals.
- Full selected-joint has 14 easy-event positive-harm violations among 72
  dependent locality views, despite zero all-event violations and no net easy
  degradation. Motion-only selected-joint has three such violations.
- The conservative motion-only selected-dual passes 18/18 observed settings,
  but loses **2.1849% to 3.6292%** all-ADE accuracy versus its old raw neural
  rule, with all six intervals negative. Its small positive gain versus R is
  a control result, not new neural dynamics or independent safety evidence.

## What Was Learned During Fitting
The added loss lowers fixed-batch selected-group error in 26/36 matched pairs
(median selected/mean ratio 0.9585), but lowers ordinary moment error in only
16/36 (median ratio 1.0047). These are fitting diagnostics, not validation.
On the full source-C pair, the median predicted/actual selected-harm ratio is
0.3979 for the old rule, 0.8911 for mean-joint and 0.8453 for selected-joint.
The action sets differ, so this is descriptive rather than an isolated
calibration effect. Selected-joint still underestimates harm in 46/72 views.
Posthoc [budget accounting](query_budget_audit.md) isolates a worse conditional
defect: at the worst locality, supported selected easy harm is underpredicted
about tenfold, while unknown-future mass is only 1.4% of the easy budget.

## Evidence and Boundaries
[Full results](results.md), [paired contrasts](source_contrasts.svg),
[fitting traces](training_loss.svg), [failure analysis](failure_analysis.md)
and [machine-readable gates](gates.json) retain the negative arms and controls.
Registration was pushed as `ef83ba79`; trained decisions were frozen and pushed
as `522131d7` before source-C readout. No threshold changed after outcomes.

Fresh_run: new risk-head training, frozen readout, diagnostics and replay.
Cached_verified: parent forecasts, source data and old utility/risk heads.
Not_run: new trajectory training, the six selection-locality readout, reserved
calibration, confirmation, remote M3W asset inventory and full legacy tests.
Targeted regression evidence is recorded separately in verification.json.

No deployment is promoted. The main protocol remains obs8/pred12 annotation
steps at raw stride12, image pixels and detector-derived labels. No metric,
seconds, human-gold, physical-safety, true3D or foundation claim. Stage5C and
SMC remain off. This is not yet a submission-ready main-method result.
