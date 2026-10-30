# Matched Frozen-Bank Cost Heads: Model And Data Card

## Intended Use

Research diagnosis of whether continuous gain/harm learning can safely use
a changed neural trajectory candidate. Not a deployable predictor, a
physical-safety certificate or an independently validated paper result.
The complete new-policy risk gate fails; no deployed model is replaced.

## Model

- 54 real Torch cost heads: six original fitting contexts, three head seeds
  (17,29,43), and three bank variants (physical, old neural, new neural).
- Each head has 30,405 trainable parameters, width64, 2,000 updates,
  batch128, learning rate0.001 and original AdamW.
- 462 causal fields: 380 candidate-conditioned motion/rollout, 18 static
  region/rollout, 64 past-only supplied-detector-history features.
- Seven action slots; six physical paths stay fixed. They are constant
  position, causal finite-difference CV, damping0.90, damping0.97,
  history-OLS4 velocity and history-OLS8 velocity. The floor is the original
  frozen causal CV/damping0.97 rule, not a newly selected baseline.
- The seventh slot uses either the old actual dimensionless grouped
  partial-neighbor producer or its frozen full-detector-history version.
  Producer seed43 is fixed, not selected from scores.
- Five nonnegative continuous moments: benefit, positive harm, reference
  error, easy reference error and easy positive harm. Asymmetric regression
  (under-harm weight4) plus within-row ranking (weight0.5), gradient clip5.
  This is neither expected-FDE regression nor best-class classification.
- Prediction-time decoder and both all/easy tolerances remain unchanged.
  No future label or mask is accepted by the causal candidate builder.

## Inputs, Labels And Roles

Cached_verified: original SOURCE exports, row IDs, source splits, trajectory
producers, detector-history cache, old policy controls and fitting normalizers.
Fresh_run: three matched past-only proposal caches, pilot, 54 full cost-head
fits, 72 readouts and a separate exact inference replay. The latter reruns
head inference, actions and metrics on the frozen proposal caches; it does
not retrain every producer or create an independent dataset.

The original controller-fitting cohort contains 149,510 unique queries,
74 recordings and 12 localities. Its 897,060 repeated context/seed row views
must not be treated as independent samples. Some future labels are partial;
3,163 unique queries have no requested future labels. Known-only supervised
costs enter the loss; unknown rows remain in the frozen causal sampler with
zero observed-cost loss. Future labels are used only for supervision and
evaluation, including explicitly future-informed oracle diagnostics.

No controller-held outcome, independent selection/calibration/confirmation
outcome or DroneCrowd outcome was opened. No test statistics, endpoint goals,
central velocity, remaining-track feature or future latent enters inference.
SOURCE has historical development exposure; the study is fitting
resubstitution, not whole-pipeline OOF or independent generalization.

## Evidence And Limits

All36 old full scientific training states reproduce. The first new-bank head
also has an independent exact 2,000-step full refit. All72 old predictions,
actions and required-recording scores agree exactly. See [verified
evidence](evidence.json), [fixed result](readout.md) and [failure analysis](failure_analysis.md).

The new policy has 1.893446% known ADE gain over the frozen floor, compared
with 1.968657% for the old bank. The new-minus-old interval crosses zero.
Only37/444 required recording-risk views pass. The same-head actual
neural-use benefit on known labels does not override either failure.
Missing-outcome bounds, native-coordinate tails, normalized cost residuals
and finite/smoothness proxies are included, not called physical guarantees.

All54 private checkpoints total23,058,125B. Public receipts, actual training
traces and aggregate metrics allow hash checks; raw data and weights are not
in Git. Reproduction needs the licensed local assets and exact recorded
environment. The public checkout alone is not a complete anonymous release.

## Scope Restrictions

Observation8/prediction12/stride12 raw frames; detector-silver bbox midpoint
pixels, not historical SDD t50, human gold, verified live sensor-as-of,
metric, seconds, true3D or foundation evidence. No scene-joint decision model,
new image encoder, latent rollout, Stage5C execution or SMC in this study.
No promotion to deployment or CCF-A submission readiness.
