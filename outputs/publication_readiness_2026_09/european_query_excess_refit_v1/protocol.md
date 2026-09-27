# Matched Query-Excess Training

## Material Passport

Fresh training of 108 paired risk heads (216 heads, 432,000 updates) against
the verified frozen query-utility experiment. Forecasters, protected floor,
utility, causal descriptors, fitting preprocessing and source roles are frozen.
Twelve already-opened development localities; no independent selection,
calibration or confirmation access. Registration precedes fitting.

## Hypothesis and Matched Design

The parent obtains better mean ADE by pooling predicted slack, but actual
risk fails. Does training at the current-query aggregation unit improve
out-of-source signed-risk ordering compared with individual supervision?

Both arms use the same 25,028-parameter descriptor head, initialization, fitting
rows, current-query grouping, equal-source sampler, AdamW settings, fixed
2,000-update budget, checkpoints and frozen utility-based allocation. Each
batch draws 32 queries, half from each of the same two fitting sources. A query
is a locality/recording/current-frame group. All known fitting-agent labels in
a selected query are used; unknown future labels contribute no supervised loss.
They remain in causal inference and action counts. Neither arm receives a
future coordinate, future mask, future endpoint goal or central velocity input.

For each all/easy signed excess error e_i, compare:

- Pointwise: average over queries of mean_i(e_i squared).
- Query: average over queries of (mean_i e_i) squared.

Both arms weight each sampled query equally. Dividing by its known-agent count
avoids letting crowded queries dominate. The loss is on a mean, whose sign
matches the sum, not a calibrated population-risk guarantee. Pure aggregate
loss permits canceling individual errors; this is a known failure possibility
to test, not a property to conceal. Single-agent queries give identical losses.
Report singleton fraction, fit and held-query loss, coverage and sparse risks.

Both heads are freshly trained because the new query-balanced sampler differs
from the previous row sampler. A cached old head is not a matched control.
No encoder or additional context feature is introduced. The four outputs
remain score bases, not separately identified expected cost moments.

## Decisions and Primary

Freeze all trained checkpoint identities before inference; freeze all actions
before readout. Keep the parent's floor, independent, joint and unconstrained
top-k masks as cached_verified controls. For each new head, retain independent
admission and utility-aware joint allocation under the same predicted 2%
all/easy constraints, fixed 256-node solver limit and checked fallback.

Primary score-ordering diagnostic: rank eligible agents by max(all,easy)
predicted signed excess, using exactly the parent's independent count in every
current query, for each new head. Compare query versus pointwise selected
positive-harm ratio; require a positive lower 95% locality CI and defined
ratios on the full roster. Also require positive same-count ADE advantage.
These rank arms are not certified budget-constrained deployment policies.

Compare actual joint policies at identical nominal risk budgets, with explicit
intervention-rate differences. Do not present this contrast as rate matched.
Retain all/easy/hard, FDE, tail, unknown interventions, complete/partial labels,
per-locality and three-seed breakdowns. The deployment screen still requires
all 216 views to have defined selected positive harm <=2%, easy degradation
<=2% versus CV, no zero-CV harm, nonempty locality coverage and positive joint
ADE advantage. Undefined ratios fail; no changed budget or selective deletion.

Use 3,000 paired locality-bootstrap draws, seed101531, after averaging dependent
views within each of the twelve fixed localities. These are development CIs,
not IID-query or independent-confirmation claims. No held checkpoint selection,
threshold tuning, independent calibration, deployment upgrade, Stage5C or SMC.

## Resources and Reproducibility

Native arm64 CPU4/interop1, workers0; first paired100-update pilot is included
in the resumed budget. Atomic lossless compressed model/optimizer/RNG/query-draw
checkpoints every500updates, per-group heartbeat and resume. Keep10GiB free.
Run locally if the measured pilot fits; otherwise use approved scheduled CREATE
resources, never login-node compute. Full prediction/action replay, first paired
fit replay and independent accounting verification follow. No new large feature
or latent bank. Image-local detector silver, obs8/pred12 raw-frame stride12;
no metric, seconds, human-gold, physical-safety, true3D or foundation claim.
