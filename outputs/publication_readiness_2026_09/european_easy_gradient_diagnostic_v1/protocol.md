# Fitting-Only Loss-Gradient Diagnostic

Status at registration: not_run. Parent easy-risk supervision experiment is
sealed negative; this diagnostic does not modify it or create a new policy.

## Question and Fixed Scope

Does explicit occurrence/conditional supervision dominate or oppose direct
signed-risk learning on fitting sources? Are label support and realized easy
harm sparse or heterogeneous enough that mean loss is a poor decision proxy?

Inspect all 108 paired fits, both arms, initial and final states. Four fixed
source-balanced batches of 32 queries per pair use seed = fitting seed + 47113.
Use the same batches for both arms and both states. Compute loss component
gradients for marginal signed risk, occurrence BCE and conditional-cost MSE;
report norms, cosines with risk and projections on risk, for all parameters,
the shared first layer and the output layer. Zero-norm comparisons are undefined,
not zero conflict. Initial paired gradients must match exactly. No optimizer
step, parameter change, new checkpoint selection, threshold or deployment.

Fitting targets retain the registered easy label, normalization and 2% budget.
Report query-balanced prevalence, easy cost/harm masses and realized within-
budget/zero-harm fractions separately for each fitting source. These are label
diagnostics, not a causal-input oracle, deployable policy or irreducible-risk bound.

The old packet whitelist, fitting IDs, source roles and hashes must verify.
Read fitting-only packets and frozen checkpoints already on CREATE; no held
actions or held labels are opened. Allocate CPU4/16G, single process, workers0.
No login-node computation, no simulation-project changes, no large local copy.
Checkpoint diagnostic progress per group; resume only matching immutable groups.
Replay all groups and compare scalar results exactly in the same runtime.

## Interpretation Before Results

Gradient norms are not loss magnitudes. Negative dot products indicate local
Euclidean gradient conflict, not proof of worse AdamW updates or future policy
accuracy. Report all arms, states and parameter blocks, not only a selected
negative slice. Source-role/seed repetitions are dependent; descriptive counts
and medians are not independent sample sizes or inferential p-values.

Large gradient conflict would motivate a controlled objective/representation
repair. Little conflict with persistent signed-risk bias would instead favor
target/feature/support diagnosis. Neither result authorizes tuning on the
previous held readout. A remedy must be separately registered before running.

No replacement of the original incomplete primary. Independent selection,
calibration and confirmation remain closed. Obs8/pred12, raw-frame stride12,
image-local detector silver only; no metric, seconds, human-gold, physical-safety,
true3D, foundation or submission-ready claim. Stage5C and SMC disabled.
