# v6: Past-Only Joint-Context Input Conditioning

This is a versioned numerical-conditioning repair under the user's delegated
obs8/pred12 development task. It is not a new independent-confirmation protocol.
The raw-frame t+50 supplement remains separate. No metric, seconds, physical
safety, foundation or true-3D claim is introduced. Stage5C and SMC remain off.

## Failure Motivating the Change

v5 seed17 completed four 10,000-update forecasters, both OOF heads and development
evaluation. One fold required explicit CPU recovery after MPS step89 overflow.
The next full-fit seed29 failed with nonfinite loss on both MPS and CPU before
its first 50-step heartbeat. No fold/seed is silently excluded to declare v5 a
complete paired study. Its artifacts remain preserved; source snapshot
`73b30e6a` retains the original bound training implementation.

The fit-only input audit found normalized ego histories within unit radius but
neighbors up to about 16,447. The failed seed17 batch had finite labels of size
2.64 but a float64 forward of the failed weights reached 3.90e24; float32 forward
was nonfinite on both MPS and CPU. These observations support testing numerical
conditioning. They do not yet prove a performance benefit or that every failure
is solely caused by neighbors. The v5 seed17 development export is preserved;
its scores were not inspected to choose this repair.

## Exactly What Changes

Before either predictor, compute r = max(1, maximum observed radial norm across
ego history and complete aligned neighbor histories). It uses the same eligible
past context already admitted to the predictor, without future coordinates,
future-validity masks, goal labels, test statistics or baseline outcomes.

Divide all coordinate inputs, including the causal baseline query, by r. Preserve
times, masks, agent membership and network parameters. Multiply the forecast by
r before the existing loss, OOF features/cost targets, scene reconstruction and
development metrics. Thus training inputs are conditioned but the original
past-normalized evaluation scale and original targets are not redefined.
No trainable parameter, clipping of forecast error, sample filtering or narrower
neighbor membership is introduced. This is not claimed as novel normalization.

Models: the same fixed-head author-core EqMotion and local Transformer, both
receiving this wrapper. Preserve batch32, learning rate0.0003, Smooth-L1, all
10,000 updates, seeds17/29/43, full + three physical-fold fits, identical source
caches/splits, both OOF cost heads and the frozen policy grid. The public model
remains an adapted K=1 control, not published minADE20/minFDE20 reproduction.

## Execution and Falsification

1. Run input/output/gradient-unit, future-input rejection and unchanged-parameter
   checks. Preserve the failed v5 artifacts instead of rewriting their hashes.
2. Replay a 100-update fit-only pilot for the previously failing seed29, using
   the new full-budget identity; if numerically finite, resume that same fit to
   10,000 and complete every registered seed/fold and matched local model.
3. Keep all negative results. Finite training only clears a runtime gate;
   predictive usefulness and easy protection require the complete development
   evaluation. No test threshold search or bootstrap of overlapping windows.

The new protocol binds this decision and current source/configuration hashes.
Old protocols require their saved source snapshot; do not modify old digests to
load a new implementation. The completed v5 seed17 can provide only a diagnostic
comparison, not a three-seed complete single-factor accuracy study. One University
development site still cannot supply meaningful independent-scene uncertainty.
