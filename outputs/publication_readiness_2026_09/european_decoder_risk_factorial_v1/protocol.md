# Source Decoding by Risk Objective: Fixed TRAIN-Internal Factorial

## Material Passport

Fresh planned experiment, motivated by the completed shared-risk failure.
Cached_verified source packets, recording partitions and frozen forecast models
are reused. Prior shared models fit their training costs worse than local heads
and fail recording transfer. This experiment distinguishes output sharing from
empirical-risk penalty interference. It is not a new official split or an
independent confirmation; the inner-held recordings have already been exposed.
No outer development or independent selection/calibration/confirmation is read.

## Four Fixed Arms

Use the same six contexts, four sources/context, head seeds17/29/43 and original
whole-recording TRAIN partition. Refit preprocessing identically on inner-fit
only. Keep source normalization, pooled priors/RMS, width32 encoder, AdamW,
2,000updates,16queries/source/update, exact sampling histories and fixed actions.

1. Shared risk: fresh exact replay of the18 prior shared-source-risk models.
2. Shared utility: identical shared model, with risk multipliers held at zero.
   Primary moment/score loss and decision regret remain unchanged.
3. Source risk: shared encoder with a separate five-output decoder per source,
   otherwise the same source-risk constraints and loss as1.
4. Source utility: the same separate decoders, with risk multipliers at zero.

All decoders start as exact copies of the shared output layer. Encoder, decoder
initial predictions, normalized inputs, targets, priors and sample draws match.
Source decoding adds495 parameters at width32:13,644 versus13,149 total, including
the unchanged unused792-parameter auxiliary head; active primary parameters are
12,852 versus12,357. This is a small source-specific decoding-capacity change,
not a parameter-matched proof of source semantics. No extra source prior or
source-dependent RMS is added. Source identity is available at inference in this
recording-transfer design; unknown sources are rejected, not mapped silently.

All72 fits are fresh;72 source-local heads remain cached_verified controls.
Each shared-risk final must exactly reproduce its prior weights, optimizer,
duals, trace and sample draws before evaluation. New arms must support exact
resume. No auxiliary target is optimized, and no new JEPA, trajectory predictor,
residual, latent rollout or stochastic proposal is trained.

## Frozen Evaluation

Freeze all72 finals before any fit or held readout. Score all72 source/seed views
in each scope,144 total, retaining all unknown outcomes and empty comparisons.
Source-local and fresh shared-risk inference must reproduce old aggregate
policies and score errors. No threshold sweep, early stopping or best-checkpoint
selection is permitted. Source priors and cost labels use only inner-fit data.

Prespecified pairwise comparisons: source-risk versus shared-risk; source-utility
versus shared-utility; risk versus utility within each decoder architecture;
and both source-decoded arms versus local controls. Report signed-score MSE,
full and query-wise same-count paired completion-lower utility, intervention,
known and upper risk, undefined support, easy preservation and source slices.
The decoder-by-risk difference-in-differences is valid for MSE point quantities;
do not subtract pair-specific lower utility bounds to invent an interaction.

Average repeated views within12 source localities before3,000 paired bootstrap
draws. Intervals are nominal TRAIN-internal, not simultaneous, population safety
or independent confirmation. Cost-head seeds are not new forecast-producer seeds.

A source-decoded arm can support later development design only if its fit and
held MSE intervals are lower than its same-objective shared control, full and
matched held utility intervals exceed both that shared control and the local
control, and every72 full-policy view meets unchanged2% all/easy positive-risk,
easy-preservation and completion-support requirements. Same-count risk is
reported, including undefined, but not an extra deployment safety requirement.
No favorable comparator, source, zero-action view or seed is removed.

All arms still face the same2% evaluation budget. Removing a training penalty
does not relax deployment requirements. Routing eligibility is only a research
design decision, never automatic deployment or independent-role authorization.

## Execution

Native arm64 CPU4, interop1, workers0; stream verified TRAIN from existing CREATE
storage, no new remote job. Prior complete72-model paired training took minutes
locally; a real100-step pilot confirms exact control/resume and resource cost.
Atomic checkpoints every100updates, heartbeat, process lock, explicit resume,
10GiB free-disk reserve,256MiB checkpoint cap and12hour continuation boundary.
Code/config/protocol/readout are committed before numerical execution.

Observation8/prediction12, raw-frame stride12, dataset-local coordinates; historical
t+50 remains a separate exploratory protocol. No metric/seconds, physical safety,
human-gold, true3D or foundation claim. Stage5C and SMC stay disabled.
