# Causal Geometric-Envelope Cost Heads

## Material Passport

Code experiment, opened-source development. Registered before fitting new
heads or evaluating their outcomes. The producer-transport diagnostic rejected
a simple smaller-predictor replacement. The subsequent frozen-head check did
not support constant-feature amplification as its root cause. This experiment
changes the output parameterization, not forecasts or scientific data roles.

## Hypothesis and Bound

Let b_t and p_t be the fixed causal baseline and candidate forecasts on twelve
requested steps. Let D=max_t ||p_t-b_t||_2, computed without any future target
or future-validity mask. For every nonempty observed subset S, the triangle
inequality gives |ADE_S(p,y)-ADE_S(b,y)| <= D. Thus realized positive gain G
and harm H satisfy G+H <= D. This also bounds endpoint-error difference when
that endpoint is observed. D is deliberately a maximum: mean rollout distance
does not bound ADE on arbitrary partial future support.

Utility head: (predicted G,predicted H)=D * softmax(z_G,z_H,0)[:2].
Event-risk head: predicted reference mass=softplus(z_B), predicted event harm=
D * sigmoid(z_H). The reference mass must NOT vanish when p=b; harm must.
The risk target remains E[baseline_error*event] and E[positive_harm*event], not
a directly regressed random ratio. All outputs train with the same native-unit
MSE and fitting-only CV scale as their matched original heads.

The bound constrains the magnitude of predicted costs; it is NOT a guarantee
that expected harm is estimated correctly, that the2% gate transports, or that
neural forecasts are accurate. The triangle inequality is standard mathematics,
not claimed as a novel forecasting contribution.

## Fixed Matrix

54 new heads:3 folds x3 seeds x2 candidates x3 tasks,2,000 updates each.
Same64-wide MLP parameter count,355 causal features,train-only preprocessing,
balanced-locality draws,AdamW settings and108,000 total update budget as the
completed nested-calibration heads. Reuse the exact18 inner OOF producers,
9 final predictors and54 original score heads. Four fitting localities and
eight excluded readout localities per fit; complete producer-chain exclusion.

Read all144 views: neural/damping x3folds x3seeds xall/easy event x4 fixed arms:
old,utility_only,risk_only,both. Compare each replacement with its old head pair,
and neural against matched protected damping. No best arm or threshold chosen
from these outcomes. Hold2% predicted risk budget fixed. Do not reuse previous
calibration maps as certificates for new scores. No reserved data are opened.

Retain all/easy/hard ADE, FDE, worst locality,zero-CV harms,unknown labels,
switch rates,relative gains,conditional3,000-resample locality CIs,raw forecast
controls and score reliability/opportunity diagnostics. All views share the
same twelve opened development localities; neither windows nor144 views are
independent replications. Preserve every negative or undefined result.

## Runtime and Validation

Native arm64 Torch,CPU4,inter-op1,workers0. Run a100-update pilot, then resume
within the fixed2,000-update budget, not an extra budget. Atomic checkpoints
every200 updates; record PID,heartbeat,loss,gradient norm,draws and elapsed time.
Use local resources if the measured workload fits. Existing CREATE read-only
queue observation is dated05:50UTC on25 September2026,not a new queue query or
proof of M3W remote assets. No login-node training or unrelated remote changes.

Before readout require all54 fits complete, original draw counts identical,
54 exact4,096-row score replays and matching input/target hashes. Recompute all
metrics and reproduce old policies exactly. Unit tests include the partial-label
bound,zero-delta event mass,causal translation invariance and exact resume.

Source-only detector tracks,image pixels,observed8/predicted12 rawstride12.
Not historicalt50,seconds,metric,human-gold,physical safety,true3D,foundation or
independent confirmation. No deployment promotion,Stage5C execution orSMC.
