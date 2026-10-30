# Detector Tokens In Trajectory Dynamics: Full Matched Experiment

## Material Passport

Fresh_run:72real Transformer trajectory producers,288,000primary updates,
4,000additional full-refit updates,900pilot updates,72fixed readout groups
and a separate complete inference replay. Cached_verified: original18
producer designs/checkpoints,318,969SOURCE rows and163record past-input
store. Fresh legacy inference uses exactly the same allowed IDs/batches.
Not_run: learned routing on the new proposal bank,controller-held outcomes,
independent selection/calibration/confirmation,DroneCrowd or deployment.

Training PID47833: 5669.635506s, peak RSS2,399,272,960B.
Readout PID54383: 1153.103515s; exact replay PID55783: 1181.451332s.
All18full disabled states and predictions reproduce the original model.
First full history refit is exact. Native arm64CPU4/workers0,no resource
probing,NumPy training fallback or reduction in original4,000step budgets.

## Matched Trajectory Accuracy

Relative to each producer own-fit-selected strongest causal baseline.
Primary is single-fold history,the original controller-consumer design.
Complement designs are retained secondary controls,not substituted winners.
Intervals use3,000locality resamples after fold/seed averaging over all12
SOURCE localities. These are exposed development outcomes,not independent
confirmation. ADE retains the original available-label mask;partial rows
do not reveal full-grid ground truth. FDE needs the last requested label.

| Producer design | Detector input | ADE gain %, CI95 | FDE gain %, CI95 | Hard ADE gain %, CI95 | Easy ADE gain %, CI95 |
|---|---|---|---|---|---|
| single | disabled | +4.922961 [+2.257335, +7.980410] | not_estimable | +4.077200 [+0.842297, +7.615753] | -0.047786 [-1.729433, +2.492386] |
| single | current | +5.961379 [+3.215498, +8.617917] | not_estimable | +4.953185 [+2.342621, +7.688788] | +1.431874 [-0.890607, +4.887632] |
| single | noise | +4.840526 [+2.221206, +7.809316] | not_estimable | +3.948929 [+0.738372, +7.395531] | -0.142711 [-2.098960, +2.587325] |
| single | history | +8.811113 [+6.289303, +11.505005] | not_estimable | +7.508666 [+3.611652, +11.165178] | +0.551800 [-4.154079, +6.630546] |
| complement | disabled | +4.512785 [-1.723068, +10.094058] | not_estimable | +4.278224 [-2.378210, +10.368277] | -1.453815 [-6.406158, +2.648069] |
| complement | current | +5.694572 [+1.676231, +9.855988] | not_estimable | +5.593256 [+2.116992, +9.660120] | -0.054788 [-4.459628, +4.871222] |
| complement | noise | +4.711686 [-1.288497, +10.145283] | not_estimable | +4.529053 [-1.839845, +10.464658] | -1.213084 [-5.766936, +2.674391] |
| complement | history | +5.744188 [-2.655877, +11.877062] | not_estimable | +5.165318 [-3.054440, +11.358627] | +2.119550 [-2.863166, +7.684458] |

## Label Support And Repeated Views

These counts repeat original producer-fold/seed views,not independent
samples. The recording union is reported separately.

| Design | Unique recordings | Indexed row views | Complete future | Partial future | Missing future | Stationary observed |
|---|---:|---:|---:|---:|---:|---:|
| single | 74 | 897060 | 549654 | 328428 | 18978 | 0 |
| complement | 74 | 448530 | 274827 | 164214 | 9489 | 0 |

The inherited zero ego-motion budget prevents stationary-to-moving
corrections. Known stationary misses remain counted for every input arm;
see support_profile in evidence.json. No future label determines inputs.

## All Causal Physical Baselines

These are descriptive paired comparisons,not selection on readout labels.

| Producer design | Physical baseline | ADE gain %, CI95 | FDE gain %, CI95 |
|---|---|---|---|
| single | constant_position | -117.340885 [-163.448859, -72.934336] | not_estimable |
| single | constant_velocity_causal_fd | +2.751314 [-4.525851, +11.032768] | not_estimable |
| single | damped_velocity_090 | +1.115445 [-1.282809, +3.112073] | not_estimable |
| single | damped_velocity_097 | +6.436269 [+0.816192, +12.629448] | not_estimable |
| single | history_ols4_velocity | +1.279565 [-4.335622, +8.835853] | not_estimable |
| single | history_ols8_velocity | -20.656348 [-35.445527, -7.245246] | not_estimable |
| complement | constant_position | -130.257592 [-198.726219, -71.926923] | not_estimable |
| complement | constant_velocity_causal_fd | +1.408967 [-3.678202, +8.094203] | not_estimable |
| complement | damped_velocity_090 | -1.520985 [-9.429409, +4.981871] | not_estimable |
| complement | damped_velocity_097 | +4.922653 [+1.114682, +9.910877] | not_estimable |
| complement | history_ols4_velocity | -1.134896 [-11.042086, +8.230687] | not_estimable |
| complement | history_ols8_velocity | -25.754706 [-53.014138, -6.559581] | not_estimable |

## Supplementary Paired Fixed-Baseline Comparison

Added descriptively after the complete fixed readout,not used to select
a model,reference or primary gate. Difference is history gain minus each
fixed baseline gain,in percentage points of the original own-fit reference
mass;it is not a newly normalized percent improvement over that baseline.

| Producer design | Fixed physical baseline | History-minus-baseline ADE pp, CI95 | FDE pp, CI95 |
|---|---|---|---|
| single | constant_position | +126.151997 [+80.068994, +174.356706] | not_estimable |
| single | constant_velocity_causal_fd | +6.059798 [-1.074082, +12.028285] | not_estimable |
| single | damped_velocity_090 | +7.695667 [+4.255479, +11.979613] | not_estimable |
| single | damped_velocity_097 | +2.374844 [-2.730947, +6.709627] | not_estimable |
| single | history_ols4_velocity | +7.531547 [+1.124459, +12.893691] | not_estimable |
| single | history_ols8_velocity | +29.467460 [+16.417413, +44.224801] | not_estimable |
| complement | constant_position | +136.001781 [+80.389030, +197.744799] | not_estimable |
| complement | constant_velocity_causal_fd | +4.335222 [-4.673836, +11.420252] | not_estimable |
| complement | damped_velocity_090 | +7.265174 [+2.953925, +12.058920] | not_estimable |
| complement | damped_velocity_097 | +0.821535 [-6.505038, +6.050071] | not_estimable |
| complement | history_ols4_velocity | +6.879084 [+0.815678, +12.205280] | not_estimable |
| complement | history_ols8_velocity | +31.498894 [+16.168511, +51.094428] | not_estimable |

## Primary History Minus Controls

| Control | Known ADE change pp, CI95 | Completion-lower pp, CI95 | Completion-upper pp, CI95 |
|---|---|---|---|
| disabled | +3.888152 [+2.245557, +5.655328] | -11.355480 [-27.808700, -0.343903] | +19.131784 [+9.639000, +34.365542] |
| current | +2.849733 [+1.869917, +3.784299] | -9.769184 [-23.054478, -1.320447] | +15.468651 [+7.592041, +27.788024] |
| noise | +3.970587 [+2.403991, +5.668633] | -11.351879 [-27.866828, -0.358578] | +19.293052 [+9.703351, +34.685748] |

Completion bounds keep causal forecast-disagreement mass for wholly
unknown future outcomes;the denominator remains fixed known reference
mass,not an identified full-population percentage. Hard/easy membership
uses observed evaluation errors and is a diagnostic subset,not missing-
membership identification. Required undefined views are not dropped.

## Actual Training Loss

| Input | Models | First median minibatch objective | Final median minibatch objective |
|---|---:|---:|---:|
| disabled | 18 | 1.05657 | 0.979434 |
| current | 18 | 1.05657 | 0.972704 |
| noise | 18 | 1.05657 | 0.981463 |
| history | 18 | 1.05657 | 0.944793 |

Losses are real training minibatches,not validation/independent losses.
The inherited trace key mean_past_normalized_ADE is a legacy name:here
its raw error is restored pixel ADE before own-fit loss factors,not metres.
Neither lower loss nor oracle physical-plus-neural opportunity constitutes
a safe intervention policy. The approved2%complete policy requirement is
unchanged and not evaluated by this raw-proposal experiment.

## Proposal Gates

- full72groups: true
- disabled_fresh_legacy_exact: true
- history_minus_disabled_known_ADE_CI_positive: true
- history_minus_current_known_ADE_CI_positive: true
- history_minus_noise_known_ADE_CI_positive: true
- history_minus_disabled_completion_lower_CI_positive: false
- history_minus_current_completion_lower_CI_positive: false
- history_minus_noise_completion_lower_CI_positive: false
- proposal_input_evidence_gate: false
- learned_policy_safety_evaluated: false
- deployment_changed: false
- submission_ready: false

## Limits

The branch bundles detector shape/confidence,body-relative motion and IoU;
it does not isolate a confidence-only mechanism. No new scene images,JEPA
or generative model is trained. SOURCE is supplied detector-silver oblique
midpoint pixels,obs8/pred12/stride12 raw frames,not human gold or verified
live sensor-as-of. No metric/seconds,true3D,foundation,Stage5C or SMC claim.
Independent and controller-held outcomes remain closed. Config/checkpoints
and exact local replay do not mean a public from-scratch anonymous asset
package is complete. See model_data_card.md and reproduction_zh.md.

[Loss and accuracy figure](figures/summary.svg), [actual traces](training_traces.json),
[machine-readable evidence](evidence.json), [failure analysis](failure_analysis.md).
