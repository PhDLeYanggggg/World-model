# Neural Conditional Risk Transport

## Question And Scope

Does adjusting a reference outcome for the causal query-reference mismatch
improve risk prediction and admission decisions? The preceding fixed-reference
attribution finds larger rollout/history mismatch among harmful source076
admissions, particularly lateral forecast departures. Other sources differ.
This is descriptive evidence for a controlled check, not proof of causation.

This experiment changes the model, not the original risk contract. It is a
neural risk head, not neural trajectory training or a generative world model.
It uses the same48,061 known own-B proposals, eight source-held folds and
three exposed contexts. The1,274 unknown proposals remain explicitly excluded
from supervised scoring, not made safe. No new nested-C/H, Moving20, DUT or
DroneCrowd. No independent confirmation claim; original24-group primary stays NA.

## Fixed Estimators

Fit f(x) on donor sources only. Retain all120 original causal features, their
fitting-only standardization clipped to[-8,8], original signed all/easy risk
targets and fitting-only target RMS. Reference IDs and weights are frozen from
moving_track_analogue_v1: nearest32 distinct source/agent buckets, with the
whole query source excluded. No new reference search uses outcomes.

Plain prediction: f(query).
Transport prediction: f(query) + weighted_mean(donor_target - f(donor)).

The correction is to a risk estimate, never to a trajectory. Only fitting
donor labels enter transport; the query label enters evaluation afterward.
The plain and transported estimates use exactly the same fitted model. A
fixed ridge regression (penalty0.01) and a small MLP provide linear/nonlinear
controls. The cached shared64 forest and cached track32 risk predictions remain.
No model branch is selected after source-held results.

MLP:120 ->64 SiLU ->32 SiLU ->2; seed17, AdamW lr0.001, decay0.0001,
gradient clip5, batch512,32complete epochs. Output weights initialize to zero,
bias to the fitting prior. Every fitting row appears once per epoch, with
fixed-seed shuffling. The weighted normalized two-channel MSE is unchanged.
No validation-selected checkpoint claim: this is a fixed-budget information
experiment, not a deployment training run. Report loss traces and every fold.

Primary: neural_transport must reduce source-macro joint decision cost by>=1%
versus shared64, not increase its harmful-admission cost, and have lower cost
than neural_plain in>=2/3 contexts. This internal follow-up screen is not the
original scene-level deployment safety gate. Keep the decision boundary at0.
Ridge, plain neural and track32 are controls, not posthoc replacement primaries.

## What Stays Unchanged

Original2% all/easy positive-harm and signed easy-net requirements, reference
support and empty-control NA. No identity-fallback certification. No future
endpoint, future mask, oracle choice or easy truth in inference features.
Native obs8/pred12 image coordinates with silver labels, not metric/seconds,
true3D, foundation or human gold. Stage5C/SMC stay off. Deployment unchanged.

## Execution

Register/commit before real fitting. Native arm64 CPU4/inter-op1/workers0.
Pilot two epochs on a full donor fold, with exact one-epoch resume and a fresh
two-epoch repeat. Then all eight complete neural fits and eight fixed ridge
fits; no reduction of samples or budget. Save each epoch and every128steps,
including optimizer, shuffle state, current permutation/cursor and visits.
Heartbeat per epoch. Respect12h phase limit,5GiB disk reserve,512MiB private cap.
Measure pilot before selecting local versus CREATE. Do not duplicate jobs.

Recompute all eight neural/ridge fits and all predictions; replay with fitting
disabled. Keep checkpoints/cache private. Publish code/config/light metrics,
negative results, failure analysis and reproduction guide. Exact repeatability
is not independent scientific confirmation. No submission-ready claim.
