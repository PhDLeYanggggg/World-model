# Fixed-Floor Tail Loss: Completed Negative Contrast

## Material Passport

Fresh_run: 216 Torch risk-head fits, 432,000 gradient updates, inference, frozen
decisions, source readout and 3,000 locality-bootstrap draws. Cached_verified:
nine forecasters, protected-damping producer chain, ridge utility and fitted
preprocessing. No new trajectory or JEPA training. Twelve development-exposed
localities only; independent selection/calibration/confirmation remain closed.

## Answer

**Training completed; the registered risk repair did not pass. Deployment is
unchanged.** This is not submission-ready or independent safety evidence.

Ordinary moment-MSE neural risk heads improve ADE by **0.1544%
[0.1154%, 0.1952%]** over the protected floor. Tail-weighted heads improve
**0.0670% [0.0448%, 0.0925%]**. Both are smaller gains than the frozen ridge
screen's 0.5233%; the ridge itself remains over the positive-harm budget.

The primary comparison is not merely one headline gain. Tail-weighted heads
intervene on 4.9228% of rows versus MSE's 7.8187%. At EXACTLY the same number
of interventions per current recording/frame, tail versus MSE ADE gain is
**0.0033% [-0.0025%, 0.0121%]**. Better ordering is not established.

The registered harm-ratio primary is **undefined** because three held views
in locality112 have no interventions and no selected reference denominator.
The fixed roster is retained; empty coverage is not assigned zero risk and
undefined views are not silently discarded.

## Positive Evidence and Failed Gates

- Real native-arm64 training and checkpoint resume work: all216 fitting monitors
  improve; no unknown-label supervised draws.
- All three forecaster seeds show small incremental ADE gain over the floor.
- Tail easy net error is preserved in every observed held view; worst easy gain
  over CV is +0.3857%, with no zero-CV-reference harm.
- Yet95/216 tail views exceed the2% selected positive-harm budget, with3 empty
  views. The ordinary MSE model violates110/216; ridge violates171/216.
- Tail versus unconstrained-count MSE loses0.0879% ADE [-0.1113%, -0.0662%].
- Tail versus ridge loses0.4607% ADE [-0.6320%, -0.3126%].
- Within213 defined tail views, the worst harm ratio is26.9994%. This is a
  descriptive tail statistic, not a primary estimate after dropping empty views.

## What the Controlled Experiment Establishes

Both new arms use the same architecture, initialization, mini-batch draws,
source roles, training budget, future-label exclusions, utility, floor and
thresholds. Only fitting-only tail loss weighting differs. This supports a
negative conclusion about this registered weighting change, not about all
neural risk estimation or all multimodal world models.

The removed failure mechanism was unconstrained negative harm clipped to0.
The new nonnegative bounded heads avoid that mechanism but do not remove
conditional moment bias. MSE's predicted selected harm is0.7169%, observed
harm3.1126%; its selected reference cost is overpredicted3.6947-fold on the
locality-averaged ratio. A positive network output is not calibrated risk.

## Next Research Action

Do not sweep thresholds or deploy the least-bad action. Test signed
`positive_harm - 0.02 * floor_reference_cost` directly, with the SAME frozen
floor and utility producers used here and matched training controls. Diagnose
conditional cancellation, error-scale dependence and source shift before
opening independent data. Earlier CV-referenced signed-excess results are
negative context, not an equivalent fixed-floor test. This is a new hypothesis,
not a promised improvement or authorization to read reserved roles.

Observation8/prediction12, raw-frame stride12; image-local detector silver.
Not historical Stage37t50, metric, seconds, human gold, physical-safety
certification, true3D, foundation or submission-ready evidence. Stage5C and SMC
remain disabled. The long-term research goal remains active.
