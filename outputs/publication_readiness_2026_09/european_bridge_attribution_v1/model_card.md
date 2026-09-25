# Model Card: Policy-Bridge Attribution

## Scope

This is a fixed attribution study of causal cost scoring, not a new trajectory
backbone or a deployable world-model release. Full pairs reuse the preceding
producer-excluded bridge. Motion-only pairs remove neural trajectory candidates
and their policy bits, then refit utility and all-risk heads. Learned motion
floor scorers remain part of both systems.

## Training

Each new utility head has 24,706 parameters; each new risk head has 24,771.
The 383-input, width64 networks use bounded geometric utility and a hurdle risk
model with moment, conditional, occurrence and pair-ranking objectives. There
are 18 source-role/seed groups, two heads per group, 2,000 updates per head,
batch256, AdamW learning rate0.0003 and gradient clip5. Three seeds are17/29/43.
All72,000 updates and36 ridge fits completed. The first head resumed after a
100-step pilot. No unknown-label row was sampled. All18 new risk heads reduced
their fixed training-batch loss; this is a training check, not calibration or
generalization evidence. Random-batch losses need not decrease monotonically.

The full-pair36 neural and36 ridge fits are cached_verified, not retrained.
Motion-only fits match their training rows, source-balanced draws, known-label
support and CV cost scale. Feature mean/std differ legitimately because the
pair changes. Ridge uses alpha0.01 and the same pair-specific inputs/targets;
the comparison confounds the learned function family and its loss design.

## Actions

Ordinary policies require current motion, different causal rollouts, positive
predicted utility and predicted positive-harm/reference error ratio at most0.02.
The matching control uses common positive-utility support and selects the same
number of agents at each past query. It compares ranking, not a shared estimated
risk budget. Hash ranking is diagnostic and need not satisfy either risk model.

Predictions and decisions for all396 views were frozen before new outcome
readout. See `results.md` for all views, `conclusions.md` for interpretation and
`world_model_gate.md` for the distinction between code completion and evidence.

## Restrictions

No deployment candidate is selected by this study. These six reused opened
model-selection localities are not independent confirmation. Predicted risk
thresholds are not physical-safety or population-risk certificates. Partial
future support follows the existing protocol; unknown outcomes never select
inference rows. No future targets or target latent at inference, central
velocity, test endpoint goals, metric/seconds claim, human-gold label claim,
true3D/foundation claim, Stage5C or SMC. Checkpoints remain local, outside Git.
