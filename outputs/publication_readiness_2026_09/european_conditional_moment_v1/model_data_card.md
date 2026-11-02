# Conditional-Cost Model And Data Card

Status: completed real SOURCE fitting and full readout; numerical replay is
recorded separately. Not deployable or independently confirmed.

- 36 real Torch cost heads, 31,381 active parameters each; marginal and identified
  arms, six contexts, three cost seeds17/29/43 sharing frozen forecaster43.
- Same warm own-choice encoder and equal new fitting-prior output initialization.
  Not an exact prediction-preserving continuation of the old critic.
- Shared predicted easy probability and conditional B/H/R magnitudes compose
  the full seven-candidate coherent B/H/R/ER/EH bank. These are cost factors,
  not future trajectory branches or a generative world model.
- Identified supervision adds occurrence BCE and conditional magnitude loss.
  It reuses the earlier easy-hurdle idea, whose old negative results remain.
- 462 frozen causal fields and initial causal selected-set context. Actual
  future easy status, endpoints and validity are loss/evaluation labels only.
  Unknown outcomes remain NaN; zero unknown supervised labels.
- 208,320 main updates plus14,336 exact full-budget refit updates; every original
  query at least four visits. Actual400-update recovery pilot completed separately.
- Frozen TDEF ranking, forecast paths, original all/easy2%constraints and roles.
  No new trajectory, JEPA, Transformer, residual or latent-generative training.
- 149,510 unique targets, 74 recordings, 12 exposed SOURCE localities, 72 groups.
  444 repeated record/context/seed risk views are not independent recordings.
- 3,000 locality-bootstrap draws after averaging repeated contexts and cost
  seeds give conditional development intervals, not independent confirmation.
- Identified known ADE gain2.830236% versus matched marginal6.909619%; primary
  completion-lower effect -5.321489pp,95%CI[-7.088381,-3.692526].
- Worst known easy net degradation1.264269% does not certify positive-harm risk:
  150known easy-risk violations remain and only55/444risk views are supported.
- EuropeanSquares oblique detector-silver/local-image obs8/pred12/stride12rawframes.
  Partial-mask ADE is not fully observed twelve-step ADE. Missing final labels
  keep full-roster FDE not estimable. No metric/effective-seconds/human-gold claim.
- Native arm64CPU4/inter-op1/workers0. Atomic checkpoints, heartbeat and explicit
  resume. Raw data, features, weights, checkpoints and third-party assets private.
- Public outputs are code, config, protocol, scalar curves, aggregate metrics,
  diagnostic intervals, reports and a derived vector figure.

Decoder coherence, lower fitting error, repeatable inference and a completed
report do not certify population safety, independent generalization, true3D,
foundation capability or submission readiness. Deployment is unchanged;
Stage5C and SMC remain off.
