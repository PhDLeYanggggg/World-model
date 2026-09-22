# Fixed Compositional Log-Loss Test

## Material Passport

2026-09-22, registered before fitting or reading outcomes from this new head.
This is a source-development loss experiment motivated by previous negative
results, not independent confirmation or a new-method novelty claim. All four
sites are design-exposed. Closed data roles remain closed; independent-scene
acquisition/calibration decisions remain pending. Stage5C and SMC are disabled.

## Hypothesis and One-Factor Change

Frozen-region square-loss heads underestimate harm on proposed switches. Updating
their fitting region failed; cross-objective review preserved easy but gave up
most gain. The next hypothesis is optimization sensitivity near a simplex face:
with a bounded output and square loss, a wrongly near-zero harm fraction can have
a small corrective gradient. Test a log score on the same cost composition.
This is a hypothesis about this fixed estimator, not a claim that square loss is
improper or that log loss guarantees generalization/calibration.

For complete training labels let b and h be positive and negative parts of
CV-ADE minus neural-ADE; D is past-only forecast-to-forecast mean distance.
Triangle inequality gives b+h<=D. For D>0, q=(b/D,h/D,1-(b+h)/D). Existing head
logits give u=softplus(z) and p=(u_b,u_h,1)/(1+u_b+u_h). The forward expected
costs remain D*p_b,D*p_h. Replace weighted squared cost error divided by D with
L = w*D*(-sum(q*log(p))). Use training-only native cost scale for D and costs.
D=0 has zero cost and zero loss; it cannot trigger intervention. Only float32
roundoff at a simplex face may be repaired (eight machine epsilons), not large
target truncation. Stable log(softplus) uses its asymptote below -20.

Because D and the frozen weight w depend on causal inputs, the ideal conditional
minimizer is E[q|inputs] (linearity in q; cross-entropy minimization). That is a
derivation for cost fractions, not a failure-event probability or calibrated
upper bound. Finite capacity, source shift and selection may still defeat it.

Prior basis: Gneiting and Raftery (2007), section3 example3, connects logarithmic
scoring and KL divergence. Section8.2 also cautions about sensitivity to extremes.
Reading scope is these passages, not an asserted full-paper read.
[Author PDF](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).
The log score is established. This trial tests task-specific value, not novelty.

## Matched Conditions

Twelve fresh heads: four outer-source sites x seeds17/29/43. Same model (width128,
two outputs), initialization, 355 causal features, nested producer exclusion,
training-only preprocessing, complete-label support, 12,000 updates, batch256,
AdamW rate.001, gradientclip5, frozen4xfit-region weights and sample draws as
`conditional_cost_v1`. Same fixed neural forecasts. No refreshed regions, new
scene split, threshold search, output clipping or fitting-budget expansion.
Loss curvature/optimization dynamics are the changed factor; numerical gradient
scales need not match squared loss. No post-readout rate sweep is allowed.

Primary policy remains strict positive net benefit, harm<=.1benefit, nonzero D,
past-stop veto. Net-only and pre-existing anchor-count ranking remain diagnostics.
No secondary winner promotion. Primary comparison is the completed frozen-region
square-loss strict policy (4.09764% developmental ADE gain), not the weaker review.
Retain tempered/native/fraction references and existing fixed-count controls.

## Gate and Data Contract

Primary paired equal-site ADE-gain difference must have positive lower 3,000-site
bootstrap bound. Also require every seed positive vs CV, easy degradation<=2%
both aggregate and every scene/seed, and zero harmed complete exact-zero-CV paths.
All conditions are required. Report hard/easy, tails, worst site, switch counts,
fitting/held conditional costs, unknown outcomes and full-grid gain bounds.

8 observed/12 predicted annotation steps, stride12, SDD annotation pixels;
4sites/33recordings/175756pasteligiblequeries. Seeds average errors, not forecasts.
Overlapping windows and repeated seeds are not independent sites. The four-site
interval is developmental. No seconds/metric/true3D/foundation claims. The
offline-annotated contract does not prove sensor-time annotation causality.

## Execution and Verification

Nativearm64 .venv-pytorch, CPU4/inter-op1, workers0. Checkpoint every500updates,
heartbeat every100; exclusive lock, PID/events and exact resume. First test a
100update pilot, then resume the same checkpoint to the fixed budget. Local cost
is expected small based on prior12headfits; check pilot rather than assume.
No new CREATE job is needed unless actual resource evidence changes.

Test simplex constraints, gradients at near-zero harm, no label gradients, zero-D
behavior, unchanged forward scores and exact interrupted resume with identical
sampling. Verify frozen source bindings before fitting. After all fits, freeze
decisions before label readout, replay all checkpoints and separately recompute
labels/weights/choices/metrics. No incomplete experiment is marked complete.

```text
.venv-pytorch/bin/python scripts/run_m3w_log_cost.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_log_cost.py --view coupa_seed17 --stop-at 100
.venv-pytorch/bin/python scripts/run_m3w_log_cost.py --resume
.venv-pytorch/bin/python scripts/run_m3w_log_cost.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_log_cost.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_log_cost.py
```
