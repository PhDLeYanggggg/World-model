# Nested full-grid advantage: fixed 2x2 factorial

Registered before training. Exposed OpenDD development, not new confirmation.
The initial four-arm registration in opendd_nested_advantage_v1 encountered a
concurrent five-arm draft at its protocol path. The pre-training hash check
stopped that run with zero updates. Preserve that draft and its old registration;
use this distinct protocol, runner, data helper, model and cache namespace.
This experiment does not implement or approve the separate five-arm proposal.

## Question and four arms

Prior policy-matched selectors fit training donors but overestimate held-location
chosen success: 67.20% versus 20.79% original-dual observed. Full-grid passes are
52 against incumbent55. Compare capacity and target choice without changing the
outer roster, forecast bank, intervention budget or original risk tolerance.

linear_absolute and mlp_absolute fit full-grid binary cross entropy including
keep. linear_advantage and mlp_advantage fit squared error between
tanh(score(option)-score(keep)) and full-grid(option)-full-grid(keep), in -1/0/1.
The small MLP has 48/32 GELU hidden layers; linear uses a single affine score.
Both use the existing keep row and aligned causal exchange features.

Fixed training: AdamW lr.001, default weight decay, gradient norm cap5, batch32,
400 updates, seeds17/29/43. Sample donor sites equally; options average within
slate then slates equally. No checkpoint or held-threshold selection. Each arm
uses the same valid-training-option mean/std, std floor.1, standardized clip8.
No validation/outer statistics enter scaling. Relative scores are not calibrated
probabilities. Shared context cancels for linear score differences. Comparisons
to the old raw-feature/original-label model change multiple factors; only the
new factorial cleanly compares its specified target and capacity factors.

## Nested location roles

For outer H, validation V, training donor D, all donor producers exclude H/V/D.
A teacher target T for that count producer is nominated by a ranker excluding
H/V/D/T. Reuse verified old double/triple rankers and double count producers.
New fits: 105 quadruple-excluded ranks x700,105 triple-excluded counts x500;
420 teacher caches. Every target-derived cutoff uses only producer fit sites.
Outcomes are consulted only after nominees, counts, anchor, exchanges and causal
features are fixed. Unresolved allocations are omitted and reported.

For each excluded pair and seed fit four selectors on the other five donors:
21x3x4=252 inner fits. The same fitted models serve both orientations of the
excluded pair because their training data and seed are identical. Validate on
each omitted V separately using the other H as outer exclusion; V's upstream
producers exclude H/V. Fit four final arms on six donors per H/seed:84 outer fits.
Total updates:126000 producer +100800 inner +33600 outer =260400.

Inner selection: compute full-grid gain over keep within each validation site,
average seeds, then weight the six sites equally. For each H select the largest
strictly positive mean; ties follow linear_absolute,linear_advantage,
mlp_absolute,mlp_advantage. Nonpositive best selects keep. Empty validation folds
block execution rather than count as success. No other metric selects the arm.
Write the selection before outer choices. Freeze all21 choices before scoring.
Inner slates use available donors and at most32 label-blind sampled exchanges,
not the outer101 eligible mask; this population mismatch remains explicit.

## Fixed readout and primary

Preserve101 eligible views,780 forecast edits,202 recordings,7 locations,606
seed views, original2% risk, full-grid support, empty-support treatment and all
count/movement/token budgets. Score ties retain keep. Report all four arms,
inner-selected policy, own-switch-mask uniform, incumbent, aligned, uniform,
hindsight oracle and prior paired. Oracle remains diagnostic.

Primary: inner-selected full-grid pass difference has strictly positive lower
95% paired-location bounds against incumbent, own-mask uniform and prior paired.
3000 draws, seed817, first average seeds within location. Factorial comparisons
are secondary, not additional primary opportunities. Report original/full/material
passes, rescue/destruction, risk undefined/violation, switching, location and seed.

These are exposed development data, not independent external confirmation.
Recording-wide observed-history aggregation supports retrospective allocation,
not a streaming claim. No future input/test goals/central velocity, external
confirmation access, author Word change, new release, Stage5C or SMC.

## Runtime and original literature

Native arm64 CPU4,interop1,workers0. Pilot one producer and one inner cell,
then complete the full matrix if local memory/time fit. Save optimizer/weights/
RNG every100 updates, immutable packets, heartbeat and lock. Preserve10GiB free;
12h checkpoint boundary then resume the same command. Never score a partial fit.

Mandi et al., ICML2022 Section4.3/Eq13 already regresses feasible-solution
objective differences. This bounded binary keep-relative diagnostic does not
claim a novel loss or inherit their theoretical guarantees.
[Original paper](https://proceedings.mlr.press/v162/mandi22a/mandi22a.pdf).
Official CVPR dates checked2026-11-10: registrationNov10/mainNov16/supplementNov23
AoE. No portal action is taken here.
[Dates](https://cvpr.thecvf.com/Conferences/2027/Dates).
