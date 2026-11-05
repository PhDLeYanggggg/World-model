# Causal Kinematic Risk Representation: Fixed Model Repair

## Material Passport

Run/validate; real model fitting on already exposed MovingDroneCrowd++ SOURCE.
Question: does an explicit past-motion basis improve conditional risk prediction
beyond equal continued training of the original flat-history neural critic?
This is a standard feature-basis control, not a claimed new architecture.

## Diagnosis And Intervention

The original 380 active fields contain full ego and nearest-eight neighbor
history, masks, floor/candidate/CV rollouts and disagreement summaries. They
do not explicitly encode backward velocity differences, speed, acceleration,
turn geometry or relative closest approach. The existing critic has a shallow
64-unit encoder and query-mean pooling. Its frozen affine and nonlinear
readout controls improve fitting error without repairing held admission risk.
That motivates testing representation learning, not asserting missing data.

Add 64 deterministic causal features: ego backward differences, speeds,
acceleration norms, turn sine/cosine, path/net length and straightness;
observed-neighbor support, relative radius/closing motion and closest approach;
and floor/candidate rollout path, speed and turn diagnostics. Closest approach
uses observed relative motion, capped at 12 native steps and divided by 12.
It is not verified collision TTC, physical safety or seconds. Future forecast
diagnostics are legal causal predictions, not future ground truth.

Keep all original 428 columns; the final 48 shared-motion columns remain zero
in both arms because their earlier controlled addition failed. Both models
start from the identical completed direct-mean neural checkpoint, with a new
zero 64-to-64 linear branch. All common weights continue learning. The matched
control masks only the new 64 inputs. Initial predictions must equal the warm
model exactly. AdamW is reset identically in both arms; no optimizer mismatch
is interpreted as a feature effect. Both have 32,130 parameters, though the
masked branch has no effective learning capacity.

## Unchanged Scientific Contract

Use the first already registered context per original critic fold, seed17,
router part1, fitting part0 and held part2. No source reassignment or additional
holdout access. Frozen forecasts, physical floor, router, original mean/quantile
heads and exact affine readout remain controls. New preprocessing uses only
causal fitting rows, balanced by fitting source, never held statistics or future
validity. Original preprocessing remains exact. Unknown labels retain the
original upper-bound harm and zero lower reference supervision.

Six real neural fits, 2,000 updates each; original source-query coverage,
sample sequence, signed-budget MSE, optimizer settings and 2% admission rule.
No threshold search, outcome-driven stopping, favorable context selection or
identity certification. Complete all six models before held-source readout.
Keep the original easy-net, positive-harm and support checks. Empty matched
controls are NA. The original 24-group primary remains unavailable, not passed.

## Runtime And Reproduction

Native arm64 .venv-pytorch, CPU4/interop1, workers0; no resource probing or
multiprocessing. Atomic checkpoint and heartbeat every100updates. Real pilot:
both arms50+50updates versus100uninterrupted; exact model/optimizer/sampler/
prediction equality. Repeat the first full2,000update fit independently.
512MiB private artifact cap and5GiB free-disk reserve. No cache/checkpoint in
Git. Healthy training may continue up to12h; interruptions resume exact state.

## Analysis And Expansion Rule

Report all fitting and held-source policy metrics, action flow, admitted risk
reliability, matched utility, all/easy violations, undefined support and worst
defined recording easy-net degradation. Completion bounds are not sampling
CIs. One producer seed and seven exposed SOURCE units are a mechanism pilot,
not multi-seed evidence or independent confirmation.

Expand only if the kinematic-minus-masked completion-lower utility is positive
in every fixed context, with nonzero intervention and no increase in known
all/easy violations in each, and the source-macro kinematic-minus-affine
completion lower is positive. This is a compute-allocation condition, not
deployment certification. Any successful model still needs the full original
risk/support contract and independent evidence. Do not claim failure proves
that causal information is exhausted.

Reserved Moving groups, DUT, DroneCrowd and H remain closed. No metric,
seconds-level, human-gold, true3D, foundation or submission-ready claim.
No Stage5C execution and no SMC.
