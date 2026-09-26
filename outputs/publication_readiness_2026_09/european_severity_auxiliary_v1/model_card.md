# Model Card: Severity Auxiliary Cost Head

## Scope

A source-development cost estimator, not a new trajectory forecaster, latent
rollout model, deployment policy or foundation model. Completion, measured
performance and decision are recorded in conclusions.md and gates.json;
this card describes the registered model rather than asserting success.

## Architecture and Outputs

383 causal features feed a width64 GELU shared encoder. Four nested nonnegative
moments model reference error D, all-harm H, easy reference error D_E and
easy-harm H_E. A separate auxiliary logit uses the shared hidden vector.
24,901 parameters, identical to the matched ordinary-auxiliary/cost-only
models. H is bounded by the causal forecast-disagreement envelope; nested
fractions keep H_E<=H and D_E<=D. This is an error bound, not physical safety.

Only H/H_E replace original readouts in the comparison. D/D_E stay from the
frozen original. The auxiliary output is never multiplied into those costs.
It targets a harm-weighted fraction, not calibrated P(easy|features).

## Fitting

The sole change is H/mean_training_H-weighted BCE, with coefficient1, beside
the unchanged four normalized MSEs. Same initialization, equal-locality
known-row sampler, seed, width, 2,000 updates, AdamW, batch256 and clip5.
Three fitting localities determine all normalizers and loss weights. Future
errors/easy labels are detached supervision only. No future target is accepted
by predict(model,x,env,pr). No clipping of severity weights or threshold search.

Six source assignments, three seeds, two feature pairs, four excluded-locality
folds:144 new fits. Source producers and comparator heads are cached_verified;
new fitting/readout status must come from the receipts, not this design card.

## Intended Use and Limits

Assess whether magnitude-aligned auxiliary supervision improves expected-harm
estimation under the fixed exploratory protocol. Do not deploy the auxiliary
probability, claim a new ADE/FDE gain from an MSE contrast, or infer independent
generalization from dependent views. Heavy tails, label noise, finite locality
support and many development iterations limit interpretation. Frozen controls
remain required even if a weaker failed model is beaten.

Private checkpoints are under
`data/stage_cvpr2027_experiments/european_severity_auxiliary_v1/heads/`.
Source and output hashes are recorded per head; raw data and model files are
not published to Git. Native arm64 CPU4, interop1, workers0, checkpoint/resume
and heartbeat are enabled. No deployment alias is changed.

Pixel coordinates, detector-derived labels and annotation-step horizons only.
No metric/seconds, human gold, true3D, foundation, physical-safety or submission-
ready claim. Stage5C and SMC remain off.
