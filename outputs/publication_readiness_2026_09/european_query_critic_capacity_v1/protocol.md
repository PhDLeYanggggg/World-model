# Training-label query units and protected-moment geometry

## Material Passport

Post-training diagnostic,registered after36 full risk fits and their telemetry,
while the72-group new policy readout is running. No partial new policy outcome
is read to choose this diagnostic. No model,objective,threshold,decoder,risk
budget or data role is changed. SOURCE fitting labels only; no independent claim.

## Two distinct questions

1. The actual selected-query gradient is small relative to all-action fitting.
   Compare the registered query-score RMS with the source-balanced RMS of true
   signed scores on nonempty,fully known ORIGINAL selected queries. Report both
   channel scales,counts and absent support. Do not treat unknown/empty subsets
   as zero-score supervision. A changed inverse-squared scale would alter each
   channel gradient algebraically,not guarantee a better combined direction.
2. Could even a label-aware oracle fit the targets while benefit and references
   remain frozen? The risk decoder has0<=easy_harm<=harm<=envelope-benefit
   (with its exact numerical anchor-preserving maximum). Project each known
   legal action target onto this triangle using the actual fitting RMS weights.
   For a complete selected query,add the frozen-minus-true reference offsets
   to its true signed target;project that desired total onto the aggregated
   triangle. Compute the minimum label-fitting loss and overlapping causes:
   negative desired harm/easy harm,capacity excess and easy>all coupling.

The weighted projection first clips independently to[0,cap]. If easy>harm,
project onto their shared edge using the ORIGINAL targets' weighted mean,
then clip to[0,cap]. Exhaustive grid tests check this formula. No solver or
learned cost head is fitted by the diagnostic.

## Interpretation limits

These are finite training-label interpolation limits,not proof that conditional
expected costs are misspecified. Realized future harm may exceed a mean-based
capacity even when a conditional mean lies inside it. A negative desired harm
is a conservative-side target mismatch; it is not automatically an unsafe
underprediction. Counts overlap and do not identify a unique causal root cause.

Use all72 original context/source/seed groups from the frozen parents,not new
selected policies. Future costs enter this explicitly label-aware diagnostic
only,never inference. The original selected-set hashes and old full-policy
controls must reproduce. All source roles and unknown masks remain unchanged.

Execute on native arm64 CPU4/workers0 with a separate lock/log/heartbeat,not the
running policy readout's files. Publish aggregates/code/receipts only. A separate
complete diagnostic replay must agree. No new raw/history/feature/latent cache,
model training or policy score. The current risk heads continue unchanged.

No metric/time/human-gold/true3D/foundation upgrade,Stage5C execution or SMC.
