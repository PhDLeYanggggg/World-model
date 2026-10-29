# Isolated Harm Learning: Fixed TRAIN-Internal Continuation

## Hypothesis and Boundary

The completed decoder-by-risk factorial found better signed-score MSE without
safer intervention when risk penalties were removed. It did not prove gradient
conflict. This experiment tests whether freezing an accurate utility-trained
benefit/reference representation while training an independent harm branch
preserves prediction quality and improves safe selection.

Use only original European source TRAIN packets, the same six contexts, four
sources/context, whole-recording fit/held partitions and seeds17/29/43. Held
recordings are already exposed exploratory material. No outer development,
independent selection, calibration or confirmation data is opened. Forecast
producers, targets, features, observation8/prediction12 and raw-frame stride12
are unchanged. Dataset-local coordinates remain unverified for metric/time.

## Matched Continuation

Start each three-arm comparison from the same frozen shared-utility final from
the preceding factorial. Copy its primary encoder into an independent harm
encoder and initialize a two-output branch at zero. The branch changes overall
and easy harm by bounded odds shifts. It preserves nonnegativity, benefit+harm
envelope and easy-harm nesting. Initial predictions match to float roundoff.

1. `utility_continue`: train base and harm branch, with zero risk multipliers.
2. `joint_risk`: same architecture, train base and harm branch with risk duals.
3. `isolated_risk`: freeze every base weight; train only the separate harm branch
   with the same risk duals. Benefit, reference and easy-reference outputs must
   remain bit-exact to the anchor. Net gain is NOT frozen: it is benefit-harm.

All arms have the same total architecture, source priors, preprocessing, pooled
RMS, initial predictions, sampling continuation and fresh AdamW state. Each
receives exactly2,000 additional updates,16queries/source/update, width32,
lr.001, weight decay.0001, gradient clip5. Risk multipliers start at zero; their
learning rate1 and cap100 are unchanged. Auxiliary heads remain unused/frozen.
Expected total parameters25,407; joint/utility trainable24,615; isolated12,258.
Freezing changes effective trainable capacity. This is a protected-branch test,
not a capacity-matched or direct-gradient-conflict causal proof.

Total54 fresh continuations,108,000 new updates. Each continued model has4,000
effective updates including its2,000-update anchor. The primary matched-budget
controls are utility continuation and joint risk. Old local and old risk heads
are2,000-update secondary controls; anchor is the no-continuation control.
No early stopping, checkpoint search, threshold sweep or source selection.

## Evaluation and Advance Rule

Freeze all54 finals before any new fit or held readout. Score144 source/seed/
scope views,72 per scope, including unknown outcomes and empty-action views.
Replay all three old controls exactly. Every isolated output must preserve
benefit/reference/easy-reference, and resumed inference must match exactly.

Compare isolated risk with anchor, utility continuation, joint risk, local and
old risk. Also show joint versus utility continuation and continuation versus
anchor. Report signed-score MSE; paired completion-lower utility normalized by
full-known floor reference mass; query-wise same-count utility; intervention;
known and completion-upper all/easy positive risk; net easy degradation;
undefined support; per-recording risk; own-selected harm predictions. Add harm
diagnosis on the identical frozen anchor-selected cohort for every arm so that
selection changes cannot masquerade as improved harm calibration.

Average repeated views within12 localities before3,000 paired bootstrap draws,
seed20261029. These are nominal exposed-TRAIN intervals, not independent,
simultaneous, population-safety or official-test guarantees. Repeated rows and
head seeds are not new independent recordings or forecast seeds. Lower-bound
utility is not raw ADE/FDE; do not subtract pair-specific lower bounds to claim
an interaction effect.

Isolated risk can motivate later development only if every full and same-count
held lower-utility CI exceeds zero against all five specified controls, all72
full-policy views meet the unchanged2% risk/easy/completion requirements, and
protected outputs remain exact. Prediction MSE alone cannot pass the gate.
Same-count risk is reported but not substituted for full-policy safety. Even a
pass does not authorize deployment or turn exposed diagnostics into confirmation.

## Execution and Recovery

Commit code, config, protocol, tests and registration before a real100-step
pilot. Pilot compares exact interrupted/resumed isolated training, all-arm
sampling and protected outputs; it is an engineering check, not model evidence.
Run native arm64 CPU4, interop1, workers0; stream hash-verified existing CREATE
TRAIN data. No new remote job. Keep10GiB disk reserve and256MiB checkpoint cap.
Atomic checkpoints/heartbeats every100updates, process locks, explicit resume,
12hour invocation boundary. Interruption resumes the last consistent checkpoint;
do not save a partially applied update as complete.

No new forecaster, trajectory residual, JEPA or generative model is trained.
Stage5C and SMC stay disabled. No foundation, true3D, human-gold, metric, seconds
or physical-safety claim. Frozen legacy deployments are not promoted here.

## Numerical Amendment Before New Fits

The float32-only precursor stopped after37 completed fits when joint-risk
backpropagation produced nonfinite gradients. A fork of the saved1600-update
checkpoint reproduced NaN in the bounded odds update; no held readout occurred.
This revision computes only the odds-ratio arithmetic in float64, retains the
original float32 denominator floor and returns float32 harm outputs. Network
weights, optimizer, losses, data, budget and fixed evaluation remain unchanged.
Because rounding changes training, all54 fits restart from the original frozen
anchors. No precursor fit is mixed into this new experiment. Preserve the
failed precursor and its registration. Float64 division is CPU-safe; no MPS
claim is made. Near-zero-envelope backward regression tests are required.
