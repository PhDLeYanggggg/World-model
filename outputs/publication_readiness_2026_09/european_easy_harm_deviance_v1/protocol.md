# Paired Easy-Harm Cost-Deviance Experiment

## Material Passport

Type: registered real-data code experiment. Role: source TRAIN and already
exposed source-development readout only. Independent confirmation: closed.
Primary purpose: test one objective change suggested by the completed 216-head
TRAIN false-safe diagnosis, not introduce a new architecture or risk threshold.

## Fixed Comparison

Train 72 original source/head-seed identities (17/29/43), with two arms each:
the exact original no-auxiliary quadratic objective, and a replacement of only
its fifth moment term (easy positive harm) by a Poisson-style cost deviance.
The unchanged target is continuous nonnegative cost, not a Poisson count claim.
For normalized prediction u=p_easy/RMS_easy and target v=y_easy/RMS_easy,
the replacement term is 2*(u-v+v*log(v/u)), with the correct v=0 limit.
Compute log(u) directly from the existing bounded decoder logits; do not clip
probabilities to create a zero-gradient floor. Zero causal envelope means exact
zero attainable harm and zero term. Unknown labels are never imputed or sampled.

The four other moment quadratics and three signed gain/all/easy-risk quadratics
are unchanged. The fifth term retains its original 1/10 total-loss weight.
Architecture, 32-wide encoder, initial weights, TRAIN preprocessing/RMS,
16-query batches, sampled identities, learning rate .001, AdamW, clip5, and
2,000 updates are identical. The original unused zero-weight auxiliary graph
is retained so the quadratic optimizer and states can replay exactly.
The nominal auxiliary_weight field stays .1 for parent-config compatibility;
its effective weight is exactly zero in both arms, as in the parent `none` arm.

## Pilot And Full Execution

The first manifest identity is chosen without label inspection. Run both arms
to100 updates. Check deviance50+50 versus uninterrupted100 exactly, and new
quadratic100 versus the original trainer100 exactly. Four hundred updates
including these two replays are engineering work, not full training. Probe
real TRAIN decoder gradients on the fixed128-query monitor; do not choose
hyperparameters or an arm from pilot outcomes. Admission is finite arithmetic,
exact controls/resume, peakRSS<14GiB and an extrapolated per-shard time<12h,
with a fixed1.5 runtime margin. A lower pilot loss is not a scientific gate.

If admitted, run all144 fixed-final fits in four disjoint shards of18 identities
and36 fits, resuming the pilot pair in its assigned shard. Save every100 steps;
CPU4, interop1, workers0; no accelerator/resource probing. Only the registered
owned M3W roots and runtime are used. Existing24 TRAIN packets remain read-only
at their parent path, not copied. Retain10GiB reserve, shared parent+child256MiB
checkpoint cap and2MiB atomic headroom. A join verifies all144 hashes/steps and
all72 exact original quadratic controls before releasing any readout.

## Prespecified Readout And Decision

Use the unchanged moving/support/gain/all-risk/easy-risk eligibility rule at2%.
Unknown outcomes remain in the inference population. Freeze every final head
before any new development prediction. Compare the deviance arm with quadratic
and the previously frozen original/additive/Poisson/cost controls, not only a
weak neural comparator. Retain full and per-query same-count paired completion
utility, harm/reference decomposition, known and completion-upper risk, undefined
support, intervention counts, tail/recording/locality results and all negative arms.

Primary paired quantities: full and same-count lower-utility differences versus
quadratic and the original forest control; aggregate head seeds/views inside
each of12 source localities, then use3000 paired locality bootstrap draws.
Report nominal development intervals, never independent confirmation. Advancement
requires positive lower CI for both full and same-count utility versus both
comparators, no known or completion-upper risk violations, and nonempty finite
complete risk support for every view. Undefined support is failure, not zero risk.
The other strong controls remain visible and must not be selectively omitted.
Even passing this screen only permits subsequent transfer design, not deployment.
No checkpoint, threshold, coefficient, seed or easy definition is selected on
this readout. Do not sweep the loss after an unfavorable result.

Past-only380 causal features, frozen trajectories, detector-silver image-local
obs8/pred12 at stride12 raw frames. No new forecaster, metric/seconds claim,
physical-safety certificate, true3D, foundation or submission-ready claim.
Stage5C and SMC remain off. Simulation and clinical assets are not used.
