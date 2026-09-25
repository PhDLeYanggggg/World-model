# Model Card: Dual-Event Policy Bridge

This is a learned intervention controller over frozen forecasts, not a newly
trained world-dynamics backbone or a generative model. R is producer A's fixed
easy-event old_stop policy; P is the same producer's all-event old_stop policy.
Their fallback forecasts may differ. All three cost heads describe P relative
to the actual delivered R, avoiding the mismatch in conjoining old scores.

Inputs are 383 past-only geometry, causal rollout and policy-bit features.
Outputs are positive gain/harm moments plus all-event and CV-positive-easy
reference/harm moments. A maximum rollout-separation envelope bounds predicted
gain and harm independently of label visibility. Learned switching also requires
strictly positive predicted net gain, nonidentical rollouts and a moving latest
step. Each risk constraint uses the fixed 0.02 predicted-harm ratio.

Eighteen producer/controller/seed settings contain one utility and two risk
heads each: 54 native Torch fits, 108,000 updates, and 54 ridge controls. Utility
uses width 64 GELU and 24,706 parameters; each hurdle/ranking risk head has
24,771 parameters. Each fit has 2,000 updates, batch 256, AdamW 0.0003, gradient
clip 5, source-balanced sampling, checkpoint and heartbeat every 200 updates.
The 100-update pilot resumes inside the first utility head's fixed budget.

No best epoch, seed or threshold is selected on readout labels. No new backbone,
image encoder, latent rollout, correction head, simulation curriculum, Stage5C
or SMC is trained or executed. Existing model artifacts are unchanged.

Predicted dual constraints are not conformal guarantees. Accuracy, inherited
reference error, easy degradation, zero-CV damage and realized all/easy risk
must all be read from the complete results, including negative controls.
This artifact is research-only; independent calibration and confirmation have
not run, and no deployment is promoted.
