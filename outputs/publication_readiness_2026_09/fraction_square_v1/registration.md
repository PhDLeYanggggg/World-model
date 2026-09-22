# Forest-Objective Neural Cost Control

2026-09-22. Register before fresh fitting/readout. Four SDD development sites
are already design-exposed. This is not independent confirmation and does not
replace any earlier failed primary gate. No new scientific roles are assigned.

## Hypothesis and Scope

The previous equal-count risk comparison gives forest3.53029% versus
neural2.80697% ADE gain, but the neural head uses compositional log loss while
forest uses two-output fraction squared error. Test that remaining objective
confound before claiming a neural architectural deficit or proposing another
large model. Standard regression losses and trees are not a new contribution.

Train12 native Torch cost heads (four sites, seeds17/29/43), using only ramp
forecasts. Reuse the same356 causal inputs, complete fitting rows, nested
source-excluded scoring/target producers, fitting preprocessing, fixed region
weights, initial parameters and exact sampler stream as the prior ramp log head.
Keep128 hidden units,45,954 parameters,12,000 updates,batch256,AdamW lr.001,
weight_decay.0001,clip5 and all checkpoint/heartbeat settings unchanged.

For benefit/harm target y and causal forecast disagreement D, q=y/D for D>0;
zero disagreement must have zero cost. Roundoff-only simplex normalization
matches the frozen forest target constructor. Network qhat uses the unchanged
softplus-pair/(1+sum softplus) bounded parameterization. New batch loss:

`mean_i(region_weight_i * D_i/train_cost_scale * mean_k((qhat_ik-q_ik)^2))`.

There is no residual-third-component term, asymmetric harm multiplier, log loss
or fitted epsilon. D=0 carries zero mass. Unknown targets are not sampled.
Over the exact prior sampler draws, this is the forest's empirical weighted
two-output squared objective up to a fixed positive normalization constant and
float32 neural arithmetic. Tree leaves use float64 target averages; SGD order,
optimization, capacity and regularization still differ. Do not claim fully
matched estimators or a single isolated architecture effect.

## Fixed Evaluation

Primary contrast: new fraction-square neural strict minus old log neural strict
in equal-site ADE gain. Primary conjunction requires its3000-site-bootstrap
CI low>0, positive gain eachseed, aggregate and eachsite/seed positive-easy
degradation<=2%, and complete exact-zero-CV added error0. Keep all failures.

Secondary comparisons at the frozen forest count per site/seed: square neural
relative-risk rank, square neural net-gain rank, old log neural ratio/gain and
forest ratio/gain. No counts, threshold(.1harm/benefit), margin, seed or model
are selected from held outcomes. Samecount is not matched displacement mass.
Freeze choices before future-label readout; retain incomplete/unknown outcomes.
Report ADE/FDE, hard/easy, worstscene/tails, interaction/smoothness proxies,
conditional gain/harm fit, forecast-disagreement bounds, runtime and all seeds.
Only fitting rows may inform training diagnostics; no early stopping on held
results. Evaluated checkpoint is always the fixed final budget.

Three-seed errors are averaged, then physical sites weighted equally. Bootstrap
uses four physical sites,3000resamples,seed38113, not overlapping windows as
independent. All intervals are conditional development evidence, not independent
calibration or a multiplicity-adjusted superiority guarantee. The earlier tree
versus original-neural primary gate remains failed regardless of this trial.

## Runtime and Boundaries

Native arm64 `.venv-pytorch`, CPU4/interop1,workers0. Full-row100-update pilot,
then resume its exact checkpoint to12k if runtime is healthy. Do not reduce the
registered budget due to slow progress. Source hashes, PID/heartbeat, atomic
checkpoints, sampler/optimizer/RNG resume, all-state replay and separate loss/
target/choice arithmetic checks required. No Torch import-only success claim.

Local first: prior24small heads trained in minutes; this12head trial does not
justify a remote transfer. CREATE connection/project/queue unverified, not
claimed absent. If resource needs change, inventory scheduler and assets first.

Observed8/predicted12 annotation steps at raw stride12; SDD annotation pixels.
Raw-t50 supplement, external confirmation, independent calibration and full
forecaster retraining not_run. This is a cost-head fit, not new trajectory
network training. No metric/seconds, true3D, foundation or human-gold claims.
Deployment unchanged. Stage5C and SMC remain off. Submission readiness unmet.
