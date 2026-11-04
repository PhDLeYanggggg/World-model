# Matched Cross-Amplitude Convex Cost Decoder

## Hypothesis And Scope

The completed frozen fitting audit finds material cross-amplitude H/EH shape
violations despite convex true labels. Test a decoder that respects this
structure against the verified unconstrained amplitude-bank head. This is a
single prospective development repair, not a threshold search, new backbone,
independent confirmation, new data role or relaxation of the 2% contract.
The original 24-group primary remains not estimable. All 23 supported groups,
195,286 past-eligible rows and 348 repeated recording views remain.

## Matched Control And Actual New Training

Reuse all 18 original amplitude-bank heads as cached_verified unconstrained
controls, not binary-only heads and not 18 newly fitted controls. Fit 18 fresh
convex heads, three actual seeds 17/29/43, 2,000 updates each: 36,000 new primary
updates. Both arms have the same 428 inputs, seven candidate paths, 28,229
parameters, width 64, physical fitting normalizers, raw initial parameters,
source-balanced query72draws, optimizer, gradient clip and learning rate.
No warm start. Compare all 18 initial states, sampler/draw hashes, IDs,
preprocessing, budgets and RNG states to their cached controls.

One original control is fully refitted to verify equality to its saved
scientific state. The first convex head also has a separate full refit.
Neither verification refit counts toward primary updates. Three critic-fold
50+50/direct100convex pilots verify exact resume and original100-step query
draw equivalence. Pilot updates are separate. Measure runtime/memory before
the full matrix; keep CPU4/interop1/workers0 and 12h/checkpoint/heartbeat/resume
boundaries. This is locally feasible from the measured parent ~50min/9GB study;
use authorized HPC only if the real pilot contradicts that assessment.

## Decoder, Not A New Convex-Network Claim

For fixed alpha grid {0,1/32,1/16,1/8,1/4,1/2,1}, use six nonnegative hinges:
phi_j(alpha)=max(alpha-t_j,0)/(1-t_j), t_j in {0,1/32,1/16,1/8,1/4,1/2}.
The unchanged network supplies per-action benefit/harm/easy logits and shared
reference logits. Full harm cap c equals the original full-action harm
softmax fraction times the full causal discrepancy. Nonnegative simplex
weights w_j come from the six harm logits. Set

H(alpha)=c*sum_j w_j*phi_j(alpha)
EH(alpha)=c*sum_j w_j*sigmoid(easy_logit_j)*phi_j(alpha).

H, EH and H-EH are then nonnegative convex curves with zero at alpha0.
They are nondecreasing and H<=alpha*full_discrepancy; EH<=H. Original shared
R/ER are unchanged. Benefit is the original predicted benefit clipped by R
and the remaining causal discrepancy envelope, preserving B+H<=envelope.
Actual labels, masks, true CV easy event and unscaled references stay unchanged.

Encoder/softmax remain FP32. Linear hinge evaluation and weight renormalization
use float64 for the small grid; training casts the final moments back to the
original FP32 loss/reduction, while inference retains stable hinge precision.
This numerical decoder is disclosed, not a separate precision ablation or a
claim that every effect can be attributed to convexity alone. Initial raw
parameter state is identical; initial full-action moments agree to floating
tolerance. Smaller-action initial curves intentionally differ. Loss, sample
weights, fitting coverage, feature schema and update budget do not change.

This is an elementary nonnegative hinge mixture. Input-convex/monotone neural
models are established prior work, not claimed invented here. Correct shape
does not guarantee correct magnitude, selected-set calibration, reference
support, population risk or generalization. The preceding shape audit is
descriptive association, not proof that shape causes the parent failure.

## Fixed Evaluation

Finish all 18 fits and verification refits before new held evaluation. Use
floor, uncontrolled neural, independent, query-uniform, count-matched joint
and free joint policies. Both use the same candidate paths and physical floor.
Freshly infer all 36 held cost outputs. Cached control costs, scalar metrics
and saved allocation bindings must exactly match the complete parent study.
Reuse its solver choices as cached_verified, not new control optimization.
New convex solves use the same 0.25s/query budget, with the original predicted-
feasible parent fallback on timeout/post-check failure. Joint means predicted
budget pooling, not optimized geometric collision constraints.

Fixed supported-group development secondary: convex-minus-unconstrained
joint-free paired completion-lower utility. Also show known/upper contrasts,
every policy, complete original recording risk, easy/hard, tails, endpoint
diagnostics, selected amplitudes, switch rates and same-count utility.
Empty utility comparisons stay not estimable; counts are not amplitude mass.
All original unsupported views remain. Separate replay validates all outputs,
scalars and saved solver feasibility, not bitwise MILP reoptimization.

Use the unchanged 3,000 source-group resamples after excluded-producer and
actual-seed averaging. These are exposed-source development intervals, not
independent final tests or a multiple-search correction. No new held threshold,
seed or checkpoint selection. No future endpoint, central velocity, future
mask/latent input, test normalization or test-endpoint goal is admitted.

Pixels/native8/12annotation steps are not verified metres, seconds or SDDt50.
Partial-label ADE and wholly-unknown bounds do not identify missing steps of
partial futures. No new trajectory dynamics/JEPA/latent rollout training,
physical safety, true3D, foundation, deployment or submission-ready claim.
Stage5C execution and SMC remain disabled. Reserved roles remain closed.
