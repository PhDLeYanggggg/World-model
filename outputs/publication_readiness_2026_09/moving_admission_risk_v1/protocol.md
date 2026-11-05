# Fixed Warm-Error Priority: Model-Side Objective Comparison

## Evidence And Question

The preceding kinematic-basis trial trained six real neural risk heads but did
not improve held SOURCE utility or pass original risk/support. Full fitting
MSE improved while admitted easy risk remained underestimated, including on
fitting data. A fresh fitting-only sign decomposition finds that observed
positive easy budget excess predicted nonpositive contributes 0.925%, 1.221%
and 1.041% of that output's complete weighted MSE in the three kinematic
contexts. Corresponding absolute output-gradient shares are 1.187%, 1.623%
and 2.103%. These are output-level loss diagnostics, not parameter gradients,
independent samples, causal identification or proof that weighting will work.

Question: does prioritizing the warm model's fixed fitting errors improve
conditional risk decisions compared with equal continued training?

## One Changed Training Factor

Keep both original signed budget-excess outputs, query-context encoder,
kinematic64 basis, causal input schema, source roles, candidate forecasts,
frozen router and support costs. Start BOTH arms from the same completed
kinematic model, not from separately initialized heads. Reset AdamW identically;
same 2,000-update budget, 72-query cyclic sampler and source weights. Both
arms train all 32,130 parameters. All input normalizers stay frozen fitting-only.

On fitting data only, freeze the warm model's raw signed risk predictions.
For each legal proposed action and output channel, mark an error if the warm
prediction is <=0 and the observed signed target is >0. This mask is a training
label, never an inference feature. Wholly unknown labels retain the original
bound-aware anchor and are never called observed errors. No hard mask is
recomputed from the changing model; no future label chooses an inference action.

Control: original normalized signed-risk MSE.
Treatment: the identical MSE, with marked channels receiving weight10 instead
of1. Equivalently add9times their original squared-error contribution. Keep
original per-query/row/channel denominators; no oversampling, learning-rate
change, gradient-norm balancing or model-dependent threshold. Ten is one fixed
training multiplier, not chosen on held scores; no multiplier search. Record
anchor and priority losses separately. The weighted output is a learned risk
score, not a calibrated estimate of expected harm or a confidence bound.

The 2% risk budget, zero predicted-excess admission threshold, reference
support and signed easy-net evaluation are unchanged. Empty matched utility
controls stay NA. Original24group primary remains NA, not a pass.

## Earlier Negative Controls Are Retained

This is standard fixed hard-example replay, not a new method contribution.
Earlier European online admitted-score penalties used changing model choices
and a nested five-moment output; their norm-parity variants reduced utility
without full safety. Fixed subset/record aggregation, proposal-conditioned
critics, mean-versus-quantile objectives, and frozen affine/nonlinear readouts
also failed. The current contrast tests fixed warm-error emphasis within a
direct signed-risk encoder and explicit causal basis. It does not erase or
claim to dominate those experiments. A negative result must not be relabeled
as a successful new safety mechanism.

## Scope, Readout And Fixed Expansion Condition

Use the same three seed17 contexts, proposer A=1, risk-fitting B=0, held V=2.
V is already exposed SOURCE development, not an independent test. Freeze all
six fits before V readout. No reserved Moving20, DUT, DroneCrowd or new H.
No early stopping, held-driven checkpoint selection or favorable-context subset.

Recompute all previous policies exactly and compare continued_mse,
false_safe_priority, warm kinematic, earlier affine and old causal controls.
Report fitting versus held utility, hard utility, switches, all/easy positive
harm, signed easy-net, undefined support, admitted risk and matched utility.
Decisions are computed before loading evaluation targets. Missing-outcome
completion bounds are reported, not mislabeled as sampling confidence intervals.
No new multi-seed or independent-generalization claim from this small contrast.

For expansion only, require positive priority-minus-control completion-lower
utility in EACH of three contexts, nonzero intervention in each, no increase
in known all/easy violations OR undefined selected/full easy views in each,
and positive source-macro priority-minus-affine completion-lower utility.
This is a compute-allocation criterion, not a substitute scientific gate.
All original risk/support checks must remain visible. No deployment changes.

## Runtime And Verification

Native arm64 CPU4/interop1/workers0, local-first, no resource probing or
multiprocessing. Real 50+50 versus100-update resume pilot for both arms;
all six primary fits total12,000updates plus an independent first2,000-update
refit. Atomic checkpoint/heartbeat every100updates. Same-binding explicit
resume, process lock,12h healthy runtime allowed,5GiBdisk reserve,512MiBprivate
artifact cap. Compare initial weights, preprocessing, masks, draw hashes,
query coverage, optimizer steps and exact independent inference/readout replay.
Engineering verification does not establish scientific success.

Native obs8/pred12 image pixels, not SDD t50 or verified seconds/metric,
human gold, true3D or foundation evidence. No Stage5C or SMC. Code/config/
reports/light metrics only in Git; no data, caches or weights.
