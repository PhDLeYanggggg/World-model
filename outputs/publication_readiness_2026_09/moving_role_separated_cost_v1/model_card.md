# Role-Separated Cost Head Model Card

## Intended Use

Research-only source-held cost prediction for selecting among a causal motion
floor and six frozen neural-forecast amplitudes. This head is not a new trajectory
forecaster, JEPA encoder, Transformer dynamics model or safe deployed policy.
The complete original recording-risk contract fails. Deployment is unchanged.

## Model And Outputs

Each head has 13,286 parameters. A 120-feature conditional MLP has hidden widths
64/32; a separate 50-feature reference MLP and 50-feature label-support MLP have
width32. The first50 raw features are action-invariant past motion/neighbor
descriptors. Candidate-dependent rollout diagnostics enter only the conditional
cost branch. Reference and support have exactly zero candidate-field Jacobian.

The coherent decoder produces benefit, harm, reference, easy reference and easy
harm. Benefit/harm are envelope bounded; benefit <= reference, easy reference <=
reference, easy harm <= harm. Composed contributions mix conditional costs and
the original unknown envelope using learned released-label support. The support
probability is not physical visibility, baseline failure or full12-label quality.
See method_boundary.md for the statistical target and asymmetric loss limits.

The two matched arms use composed-moment loss versus observed-only conditional
loss plus complete-pool support BCE. Harm-underprediction weighting is fixed at4;
there is no claim of unbiased conditional means or calibrated confidence bounds.
Fitting-only normalization, target RMS and causal per-row scale must be applied
with each checkpoint. Checkpoint weights alone are not an inference API.

## Training And Provenance

Two arms x eight source-excluded folds x three risk seeds17/29/43 = 48 primary
models, 4,096 steps each; 48 complete repeat fits reproduce exactly. Each arm
uses the same initial state, complete source/query draws and 50/50 own-B/A fitting
mass. Final checkpoints are fixed, not selected by outcomes. Teacher seed17 is
unchanged. This is a source-specific cross-fit collection of 24 heads per arm,
not one universal deployment checkpoint or three independent forecast runs.

Evaluation uses 62,466 already-exposed own-B rows in eight source components and
24 recordings. All374,796 legal own candidates remain, including10,590bound-only;
producer-A training banks contain753,240candidates. Future labels and validity
supervise loss/evaluation only, never query admission, causal inputs, goal
construction or preprocessing. Parent lineage and source exclusions are verified;
annotation-generation causality remains unverified. No new held C/H is opened.

## Results And Limitations

Registered paired completion-lower improvement versus the matched anchor is
+0.445303pp, exploratory source interval[+0.090835,+0.930796]. All three risk
seeds have positive means; two source lower contrasts are negative. The separated
joint policy passes only23/72 required repeated recording views, with27known-easy
harm violations. It is not a safe candidate. The model can overestimate reference
and underestimate selected harm even on fully observed selected rows.

The cost decoder, full-action coverage and five-target task differ from older
two-signed-output heads. Only the two new arms isolate the specified loss role.
Solver timeout fallback preserves predicted feasibility, not realized risk.
Available-label native8/12image-coordinate ADE does not become full12ADE, metric
distance, verified seconds, true3D, a foundation result or human-gold supervision.
Original24-source-group main remains NA. Stage5C and SMC remain off.

## Storage And Reproduction

Public code, config, registrations, scalar reports and hashes are in Git. Private
checkpoints/arrays remain under
`data/stage_cvpr2027_experiments/moving_role_separated_cost_v1/` and are not
published. Reproduction needs the authorized parent caches and local native
arm64 environment. Read operation_zh.md and verification_report.md; replaying
fixed decisions is not independent validation or a rerun of timed integer search.
