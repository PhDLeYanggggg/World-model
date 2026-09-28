# Paired Risk-Priority Auxiliary Training

Status at registration: not_run. Scope remains the twelve already-opened
European source-training development localities. Roles per context remain
4 producer /4 controller /2 fitting /2 held-development. This is a controlled
repair within development, not independent confirmation.

## Single Changed Factor

The verified fitting diagnostic measured auxiliary gradient norm dominance
and some direct-risk conflicts. Compare equal-weight supervision (uncapped)
with g = g_risk + alpha*g_aux, alpha = min(1, .5*||g_risk||/||g_aux||).
Use the Euclidean norm over all model parameters, detached alpha, zero auxiliary
update at zero risk gradient. Both arms retain AdamW, global norm clipping5,
the same model, initial state, source-balanced query draws, normalization,
easy labels, causal envelope and objectives. There is no weight/threshold sweep.
The fixed .5 cap preserves at least half the direct-risk projection before
AdamW in real arithmetic; it is not an AdamW or held-risk guarantee.

Auxiliary tasks are occurrence BCE and conditional cost MSE; direct risk is
the unchanged row/query signed-risk loss for E*(H-.02R). No prediction target,
observation8/prediction12, raw-frame stride12, coordinate scale or .02 budget
changes. Forecasters, reference floor, all-risk score and utility stay frozen.

## Training and Runtime

Retain all108 packet groups, all three original seeds, two arms, 2000updates
per head:216heads/432000updates. Existing fitting-only packets are hash-verified
on CREATE and not copied locally. CPU4/16G/2h, one process, workers0. A first-pair
100-update real pilot checks finite loss/gradients, resource envelope and matched
sampling; it is continued to full training, not selected for score. Checkpoints
every500updates, heartbeat every500 and per group; exact resume guards.
Both arms share initialization and query sample hashes. Uncapped full states
must reproduce the existing supervised states excluding identity and elapsed
time. Replay the first full pair in the same runtime before scientific readout.

## Development Readout Fixed Before Training

Freeze all model states and causal actions before opening new development
outcomes. Policies:floor, common_anchor; raw,uncapped,risk_priority each with
independent, own-count joint and common-count matched action variants. Common
anchor is the intersection of all three causal feasible anchors. Joint actions
maximize the same frozen utility with both signed-risk constraints and node
budget256. Primary repair contrast:risk_priority_matched versus uncapped_matched
ADE at the identical number of interventions per query. Secondary:each matched
arm versus raw_matched; each own-count arm versus raw_joint. Do not combine old
and new common-count sets as if identical. Also compare head signed MSE/bias,
occurrence Brier/logloss, conditional costs, benefit lost/harm avoided,
selected risk including zero-denominator abstentions, easy/zero-CV harm,
worst locality, tail, seed and count behavior. Report all groups and controls.

Use the existing locality-level paired 3000-resample seed101531 bootstrap;
intervals are nominal and exploratory after repeated development, not formal
independent confirmation. The original incomplete selected-risk primary is not
replaced by accuracy, fitting loss or a full-floor-denominator diagnostic.
Any accuracy gain with failed risk/easy screens remains nondeployable. No
checkpoint, threshold, cap or new normalization is selected on held results.

## Evidence Boundaries

Source result tags:fresh_run for new training and readout; cached_verified for
hash-checked packets, forecasters and controls; not_run for unexecuted steps.
Independent selection/calibration/confirmation remain CLOSED. No future
endpoint, future latent, central velocity or test-endpoint goal in inputs.
Image-local detector silver, not human gold. No metric/seconds, physical-safety,
true3D, foundation, deployed improvement or submission-ready claim from training.
Gradient norm balancing is established methodology, not a novelty claim.
Stage5C execution and SMC stay disabled.
