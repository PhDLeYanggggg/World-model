# Causal Kinematic Risk Representation: 5 November 2026

This addendum records a failed model repair. It does not replace or rebuild
the main-paper PDF, reopen independent H, or change any evaluation rule.

We trained six neural risk heads in three fixed source-separated contexts.
An explicit 64-field causal motion basis is compared with a masked-branch
continued-training control. Both start from the same warm risk model, with
zero new-branch weights, identical reset AdamW, query sampling, bound-aware
targets and 2,000 updates. The common encoder is trained in both arms. The
added fields describe backward displacement, speed, acceleration, turning,
observed-neighbor approach and causal baseline rollout diagnostics. They are
deterministic transformations of existing inputs, not a new modality. Both
arms have 32,130 nominal parameters; the masked arm has less effective capacity.

All six fits were frozen before held-source readout. Full fitting-data MSE
reductions are 72.99%, 58.10% and 30.13% with the features, compared with
34.86%, 36.28% and 14.38% without them. These losses use the original empirical
query weights and missing-label supervision. Yet fitting policy utility is
0.5288 percentage points worse with the features. This disconnect occurs
before held-scene evaluation, so distribution shift is not the sole explanation.

On seven already exposed held SOURCE units, known-label ADE improvements
over the same floor are 4.3233% for the earlier affine control, 3.9625% for
continued training, and 3.9418% for the kinematic arm. The matched feature
contrast is -0.0207 points, with missing-outcome completion bounds
[-0.9431, +0.9017]. Against affine, it is -0.3816 points, bounded by
[-1.1439, +0.3808]. These bounds are not sampling confidence intervals.

Only one of three fixed contexts meets the prospective expansion condition.
Neither new arm passes complete original risk/support on any of 17 recording
views. Relative to the continued-training control, kinematic all-risk violations
increase from 13 to 14. Easy violations decrease from 16 to 14, but two
selected-easy views become undefined. Worst defined signed recording easy-net
degradation remains 37.3451%. These counts cannot establish preserved safety.

The model overestimates easy risk across the proposal pool but underestimates
it on admitted actions, including on the fitting data. Thus lower average
regression loss does not establish decision-conditional calibration. Causal
kinematic engineering alone does not repair the failure. Extra training of
the original representation also loses held utility, so the new branch is
not the sole explanation. This experiment neither proves causal features
uninformative nor identifies a general impossibility result.

The real resume pilot, full independent refit, and all three inference,
decision and metric blocks reproduce exactly. Seventeen focused tests pass.
These are actual neural risk-encoder updates, not new trajectory training,
multi-seed evidence, independent confirmation or a novel dynamics contribution.
No favorable context is promoted. The original 24-group primary stays NA.
The original 2% harm, easy-net and support rules remain unchanged; no identity
certificate or deployment/submission-ready claim is introduced.

Evidence: [complete report](../moving_kinematic_risk_v1/README.md),
[failure analysis](../moving_kinematic_risk_v1/failure_analysis.md),
[full fitting losses](../moving_kinematic_risk_v1/full_fitting_loss.json),
[protocol](../moving_kinematic_risk_v1/protocol.md),
[replay](../moving_kinematic_risk_v1/verification.json).

Native obs8/pred12 image-pixel development only. No verified seconds, metric,
human-gold, true 3D or foundation evidence. Stage5C and SMC remain disabled.
