# Explicit Scale Removal: Prospective Retrained Ablation

## Question

I test whether the risk head uses explicit absolute pixel scale as a source
shortcut. A causal-only coverage audit finds that 57.05% of proposed rows in
the largest negative source078 have log-scale outside their own fitting range.
This is a descriptive association, not a demonstrated cause. Sources051/062
have much more feature clipping yet improved under the previous objective;
clipping alone cannot explain that result. Near-constant fitting dimensions
make some standardized mean shifts enormous and unsuitable as causal evidence.

## Fixed Intervention

Remove only common input column354, the explicit log causal-past pixel scale,
by setting its standardized value to zero in both fitting and inference.
Keep all other492-input positions, normalized motion, neighbors, rollout
geometry,64kinematic features, target labels, source roles and preprocessing.
No new forecast, teacher, threshold, risk definition or calibration rule.
The removed column is inactive; the nominal32,130-parameter architecture stays
the same. Warm weights originally saw scale: this is a continuation ablation,
not a claim that the full system has never used scale.

Frozen proposers, support costs, forecast candidates and positive score
rescaling remain original. Thus this is NOT full-policy coordinate invariance.
No future endpoint, validity target, central velocity or test goal is an input.

## Controls And Training

Six fresh fits: three fixed source-separated contexts times two objectives.
Main comparison: masked continued MSE versus its verified unmasked MSE control.
Secondary: masked signed-cost logistic versus its verified unmasked counterpart.
The six unmasked controls are cached_verified from moving_signed_decision_v1;
they are not newly trained here. Identical original warm weights, optimizer
reset, source-balanced query sampler, seed17,72queries/batch,width64,lr0.001,
clip5 and2,000updates per fresh fit. Unknown-outcome bounds remain unchanged.
Targets, proposals, sampling draws and every unmasked input must match controls.

Each objective gets50+50resume versus100direct;400pilotupdates including repeats.
A complete first-context masked logistic refit adds2,000updates to12,000primary.
Every model freezes before exposed V evaluation. Checkpoint/heartbeat every100,
CPU4/interop1/workers0,nativearm64,12hguard,5GiBdiskreserve,512MiBprivatecap.
Healthy slow training is not downgraded. No reserved data or new H readout.

## Evaluation

Retain all seven exposed SOURCE units,17recording views,one producer seed.
Fresh new predictions/actions and metric recomputation; verify old actions by
hash and all old metrics by exact recomputation. Exact full prediction replay.
Report both objectives and all sources; do not substitute the better objective
after viewing V. Main expansion screen: positive paired missing-label completion
lower contrast in every context against unmasked MSE, nonzero actions, no increase
in known all/easy violations or undefined selected/full easy support, and positive
aggregate completion lower versus affine. Report secondary screen separately.
This is a development expansion screen, NOT the original deployment safety gate.
Missing-label completion bounds are NOT sampling confidence intervals.

Original2%positive-harm,easy-net and support rules remain unchanged. No identity
fallback certificate. Empty matched utility controls remainNA. Original24group
primary remainsnotestimable, old failures remainfailures. No new source/epoch/
threshold selection on V. Reserved20Moving,DUT,DroneCrowd,Hremainclosed.
Deployment unchanged, submission_readyfalse, Stage5C/SMCoff.

Native observation8/prediction12 imagepixels, no verifiedseconds/metric/true3D/
foundation/human-gold claims. This study trains neural risk heads, not new
trajectory dynamics. Repeated development evidence is not independent confirmation.
