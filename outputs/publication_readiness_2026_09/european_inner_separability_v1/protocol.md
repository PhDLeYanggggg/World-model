# Internal Locality-Held Gain/Risk Separability

## Material Passport

Registered development training experiment. Parent: the verified fitting-switch
diagnostic. Existing 12 opened localities, obs8/pred12, raw stride12, image-local
detector-silver. This is not independent selection/calibration/confirmation and
does not change the existing primary risk estimand, source roles or deployment.

## Falsifiable Question

Can a nonlinear causal cost head improve gain/all-risk/easy-risk prediction and
same-count protected selection over an affine head when the evaluated fitting
locality supplies neither labels nor preprocessing to that head?

The latest diagnostic placed75.33% of available benefit in the all-risk-rejected
partition. That partition also contains harm. Another occurrence-freezing or
threshold sweep is not the proposed intervention. Earlier utility-only, tail-loss,
signed-risk and descriptor/capacity studies remain negative or safety-incomplete.
This experiment changes capacity within a paired single-source cost-learning
design, not the frozen forecaster. It does not claim architecture novelty.

## Source Protocol and Deduplication

Keep each registered4/4/2/2 producer/controller/fitting/held assignment. Within
the two fitting localities, train on one and evaluate on the other, then reverse.
All preprocessing, support cutoff, cost scales, loss scales and initialization
priors use the internal training locality only. No previously learned two-source
utility/risk scores, normalizers or descriptors enter the new model.

There are18 frozen producer/seed/controller contexts and6 fitting pairs each:
216 directional evaluations. A head trained on the same one locality is identical
across the three contexts where that locality is an internal training source.
Train it once and reuse the checkpoint:72 unique fits x2 arms =144 heads, not432
independent trainings.288,000 planned updates. Preserve the two outer held
localities of each directional view; no outcomes from them enter that view.
Roles rotate across views; this is cross-fitted development, not new unseen data.

## Single Paired Factor

- Same380 raw causal features, rollout envelope, protected floor and forecasts.
- Affine logits versus one32-wide SiLU hidden layer. Both use the same bounded
  five-moment decoder and identical initial predictions. Parameter counts differ.
- Output expected benefit B, positive harm H, floor reference error R, easy-event
  reference ER and easy-event harm EH. B+H <= causal envelope, ER <= R, EH <= H.
- Train on native cost labels normalized by the internal training mean floor
  cost. The registered CV-defined easy event remains unchanged.
- Proper squared moment loss plus squared signed decision-score loss on
  (B-H, H-.02R, EH-.02ER), weighted1/2 each. Fixed train-only RMS per component,
  floor.01. Equal query weights; unknown labels never sampled. These are constant
  loss weights, not sample-dependent ratio/ranking labels.
- AdamW lr.001, decay.0001, clip5,16 queries/update,2000 final updates. Same sampled
  queries, no checkpoint selection, no hyperparameter/threshold search.
- Training-only99th-percentile support guard, as in the prior support definition.

## Frozen Actions and Readout

After all training, freeze causal scores and actions before computing directional
outcome metrics. Store score hashes and small action masks, not large score caches.
Compare floor, unprotected neural, intercept-only decoder, affine independent,
nonlinear independent, and their same-count controls. For each query, matched
count is the minimum of the two independent admissible counts. Each arm ranks
its own admissible pool by expected gain, ties by row ID. Both all/easy predicted
risks must be nonpositive per selected row; no joint MILP or relaxed budget.

Primary developmental contrast: nonlinear minus affine normalized signed-score
MSE averaged over gain/all-risk/easy-risk. Lower is better. Secondary same-count
ADE gain, FDE, hard/easy, tail, harm ratios, intervention and unknown/undefined
coverage remain mandatory. Report initial-decoder skill, each locality and three
forecaster seeds. Use3000 paired draws over12 locality means after averaging
dependent views. Nominal intervals only; no independent confirmation or
multiple-comparison/selection-adjusted claim. Do not drop undefined views.

A useful capacity result requires a negative primary interval and no safety
regression on the matched decision readout. Even a positive internal result does
not promote deployment or solve the original selected-risk/independent-calibration
blockers. Failure localizes limitations; it does not prove all causal features
are incapable of forecasting useful intervention.

## Execution and Recovery

Native arm64 CPU4/interOp1/workers0. Run a measured100-update paired pilot; project
full compute and storage, keeping10GiB free. Local first; use CREATE only if the
pilot justifies it. Fresh CREATE queue inspection found no M3W jobs; no remote
job is submitted by this registration. Save compact checkpoints every250 updates,
heartbeat every250 updates and per source, with an exclusive process lock.
Resume must reproduce uninterrupted weights and sampler state in tests. Verify
one full paired training replay, all216 causal action replays and the numerical
readout. No restart on a mere observation timeout; no slow-run scope reduction.

No Stage5C, SMC, metric/seconds, true3D, foundation, human-gold or safety-certificate
claim. Publish only code/config/aggregate evidence. Keep row data and checkpoints
private; preserve unrelated staged changes.
