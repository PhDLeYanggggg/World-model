# Locality-Excluded Harm Heads

## Purpose

This diagnostic separates easy-case harm ranking from expected-harm magnitude
prediction. It is not a new trajectory model, deployed intervention rule,
probability calibrator or physical-safety guarantee.

There are 144 cold-start Torch mean-moment heads: six ordered source roles,
three seeds, two frozen forecast pairs and four held controller localities.
Each original EventMomentHead has width 64 and 24,836 parameters. It receives
2,000 AdamW updates with batch 256, learning rate 0.0003, weight decay 0.0001
and gradient clipping 5. No head is chosen by its held-locality results.

The outputs are reference error and positive excess error, for all cases and
for the diagnostic easy event. The training loss is the original mean MSE.
The plotted fixed-batch loss is a fitting diagnostic, not validation accuracy.

## Inputs and Boundaries

Frozen causal inputs and forecast-disagreement envelopes come from source A.
Only three B localities fit normalization, cost scale and the easy-event cut.
The fourth B locality contributes no fit labels or learned preprocessing.
Future coordinates are supervision/evaluation only. No central velocity or
test-endpoint goal is introduced. A B-fitted selector eligibility mask is not
used; the secondary subset is positive causal forecast disagreement.

The fraction score divides predicted harm by forecast disagreement and uses
zero when disagreement is zero. Neither score is an event probability.
Top-tail capture is a label-based evaluation diagnostic, never an inference
rule. Score-bin boundaries are fitted on the three training localities only.

## Use and Limitations

All heads and preprocessing state remain in the ignored private experiment
directory with optimizer/RNG checkpoints. Public receipts bind their hashes.
These heads are not deployed. Existing forecast, threshold and fallback
artifacts are unchanged; no new Stage37 performance claim is made.

The three-site easy definition varies across folds. Original full-B models
use a different, whole-B definition for B/C transport diagnostics. These are
not target-matched populations. Four held localities per ordered role and
overlapping role assignments limit the uncertainty analysis. Low event
support is reported, not treated as evidence of safety.

Current scope is image-pixel top-down trajectory research with detector-derived
labels, not human gold, metric/seconds prediction, true 3D or a foundation
world model. Stage5C and SMC remain off. See results and failure analysis for
the numerical conclusion; completion of this diagnostic is not promotion.
