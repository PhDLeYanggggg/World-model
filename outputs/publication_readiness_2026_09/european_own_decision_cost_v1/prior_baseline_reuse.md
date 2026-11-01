# Existing Published-Loss Baseline: Reuse Rather Than Recreate

## Material Passport

fresh_run:public artifact and source-lineage inspection on1November2026.
cached_verified:the existing `european_regression_deferral_v1` registration,
registered executable bindings,complete/summary/replay hashes and144-group
scalar archive. This is not new training or fresh inference of that baseline.
not_run:adaptation and evaluation of that head on the current updated bank.

## What Already Exists

The project already implements a disclosed application of Mao,Mohri and Zhong's
[ICML2024Eq3 logistic multi-expert deferral surrogate](https://proceedings.mlr.press/v235/mao24d.html).
It uses continuous expert ADE costs rather than one-hot oracle identities.
The old experiment completed54heads/108,000updates across three bank families,
six contexts and three cost seeds,with exposed fitting and inner-held readouts.
Its original bounded-loss theorem assumptions were not verified for this native
partial-mask ADE application;it is not a reproduction of the original paper.

The cached fitting mixed-bank known ADE gain is21.735065%,but original recording
risk support is0/444. Its exposed inner-held mixed gain is8.018405%with0/186risk
support. The latter186views represent31unique recordings repeated over head
and context views,not186independent videos. None is independent confirmation.
Those old results remain unchanged and are not new results of the current trial.

## Why These Scores Do Not Enter The New Primary Table

The current pipeline follows `run_m3w_active_bank_cost.prepared_context`,which
replaces the neural trajectory candidate and recomputes its causal features.
The earlier published-loss runner follows the older multiaction bank and its
own fitted preprocessing. Exact current-bank feature/prediction equivalence
has not been established. Same context names and observation lengths alone do
not prove that two controllers saw the same experts or normalized features.

Reuse `src/world_model/m3w_regression_deferral.py` and its valid surrogate tests
for a prospectively registered current-bank baseline. Bind the current forecast
hashes,causal feature schema,normalization,role assignments and full target IDs.
Disclose any floor-token construction and capacity/budget differences. Re-evaluate
the complete policy under the same2%risk contract,including unknown outcomes.
Do not relabel old-bank predictions as a fresh or matched current-bank baseline.

No new loss-baseline training is reported here. The active own-decision trial
continues unchanged;this inspection prevents redundant work and an invalid
cross-bank comparison,not a change to its registered primary or data roles.
