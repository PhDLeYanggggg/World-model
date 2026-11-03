# Model And Data Card: Observation-Support Costs

## Status

Complete real Torch experiment with negative primary and failed full risk.
Use for diagnosis,not deployment or independent confirmation. A better released-
label probability score is not a new world-model contribution.

## Model And Targets

Two matched new model families,18nested jobs each,36heads total. Each width64
query-context MLP has29,318parameters: the original29,253conditional cost head
plus one shared65-parameter observation logit. Both start with the same weights,
fixed.95observation prior,normalization,query draws and full budgets. Comparison
to the old direct head is not capacity/initialization matched.

Inputs:444past-only geometry,rollout and detector fields plus exact causal
candidate/floor disagreement. Outputs: expected conservative B/H/R/ER/EH and
probability that at least one released requested future label exists. Compose
known-conditional costs with explicit unknown-bound contributions. No probability
threshold is fitted. The original2%predicted guard controls interpolation or
exact fallback; it does not certify actual risk. The marginal versus identified
arms differ only in the observation/conditional auxiliary objective.

Future masks and labels are targets only. Inference rejects additional target
fields and receives no future endpoint,central velocity,eventual track length,
test normalization or endpoint goals. Nu=1does not mean complete12future labels,
human visibility or a baseline-failure event.

## Data And Roles

149510SOURCEwindows,74recordings,twelve exposed EuropeanSquares localities.
For each nested job,four producer sites,four other critic sites and four remaining
evaluation sites are disjoint. All fitting preprocessing and original baseline/
easy/hard definitions retain their fitting roles. Seeds17/29/43are actual fits.
444recording/pipeline views are not444independent scenes. No calibration private
arrays,final confirmation or DroneCrowd are opened by this study.

Observation/prediction is8/12,raw stride12,image-local pixels,detector/tracker
silver. No verified time/metric/depth or human-gold interpretation. Available-
label ADE is not full12ADE; bounds cover wholly unknown rows,not partial steps.

## Results And Compute

Known gain versus floor:0.865652%,versus frozen direct3.430437%. Paired known
effect-2.564786pp [-3.740926,-1.480043]. Primary completion-lower effect
-3.133282pp [-4.410254,-1.987128]. Full-risk support160/444,worst signed easy-net
8.319583%; no policy passes. Availability AUROC0.636791is not failure AUROC.
CI unit:twelve localities,3000resamples after excluded-view/seed averaging.

116160primary optimizer updates;4000full-refit and1200pilot updates counted
separately. Real six-cell recovery and both first full refits exact. Nativearm64
Torch CPU4/inter-op1/workers0,atomic checkpoints,heartbeat/resume. Training
404.408s,peak9.230GB. Complete replay131.106s,including77.438s full prediction
inference. All36cost/probability/amplitude triplets and72metric groups exact.
Local resources suffice; no new HPC job or MPS success is claimed.

No new trajectory,JEPA,Transformer,generative rollout or correction training.
Public artifacts reconstruct numerical reports; complete anonymous cold-start
data/model preparation and independent method confirmation remain incomplete.
Deployment unchanged,Stage5C/SMCoff,not submission-ready.
