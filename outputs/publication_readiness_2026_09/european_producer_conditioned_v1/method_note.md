# What The Producer-Conditioned Experiment Tests

I test whether the gain/harm controller benefits from knowing the identity of the
model bundle that generated its candidate and fallback trajectories. This is a
controller-learning experiment, not new trajectory-dynamics training.

## Source Exclusion

Each outer development fold has four fitting localities and eight excluded
localities. The fitting localities are partitioned into two fixed pairs. A fitting
row receives forecasts only from the opposite pair's trained neural forecaster
and protected CV/damping controller. Neither producer has trained on that row's
locality. The two bundles are then applied separately to the eight excluded
localities. Both branches are reported; I do not select the better bundle or
average their outputs after seeing results.

The earlier two-source neural forecast banks are reused after artifact hashes,
source lineage and checkpoint-prefix replay have been verified. The two-source
protected-floor predictions on excluded localities are computed afresh. The
four-source protected floor and stopping controller remain common evaluation
references. This separates changing the fallback from improving the controller.
The cached trajectory forecasters retain their parent's masked-loss sampler,
which included unknown-future draws with zero supervised loss. Those draws are
not counted as labeled examples. The newly trained controller sampler instead
excludes unknown targets entirely; this does not retroactively change the
forecaster training protocol.

## Matched Heads

All three arms have 382 inputs and a width64 hidden layer. The first 380 inputs
are the same causal history and forecast diagnostics. The final two inputs are:

- Global: zeros.
- Producer: the one-hot identity of the bundle making both forecasts.
- Placebo: a deterministic row-key hash, independent of outcomes. This is a
  noise/capacity control, not a recommended deployment feature.

The controller predicts expected benefit, positive harm, and baseline-error
risk moments. Supervision uses future labels only in the loss. Utility is bounded
by a causal rollout envelope. Risk uses occurrence/severity learning and a
cross-moment ranking loss with a fixed, fitting-only denominator. The all-event
and easy-event targets are retained as separate registered configurations.
Preprocessing is fitted only on the fitting rows. Unknown future labels are
excluded from supervised sampling, not assigned zero error.

Within each fold, seed, event and head type, the arms have identical parameter
counts, supervised draws, labels, loss settings, optimizer, number of updates,
causal inputs and ranking denominator. The registered final checkpoint has
2,000 updates; no checkpoint is selected using the development readout. There
are 108 new heads and 216,000 optimizer updates, including the resumed pilot.

## Decision And Outcome Separation

A neural intervention requires a nonzero latest observed displacement,
predicted benefit exceeding predicted harm, and predicted positive-harm moment
at most 2% of the predicted reference-error moment. Otherwise the branch's
protected motion baseline is used. This is a prediction-based rule, not an
independently calibrated guarantee.

All three arms, an inference-only wrong-tag diagnostic and the earlier
380-feature controller produce frozen decisions before new outcome readout.
The primary contrast uses exactly the same candidate/fallback forecasts for
producer versus global/placebo controllers. Wrong-tag sensitivity helps show
whether the tag is used, but sensitivity alone is not predictive usefulness.

## Evidence Boundary

Producer identity is confounded with the two fitting-source cohorts. A positive
result cannot by itself establish a causal effect of producer shift. Differences
from the old four-source stopping controller also include producer-quality changes;
only the identical-forecast comparisons isolate the new controller input.

The 180 views are correlated combinations of three folds, three seeds, two
events, two bundles and five policies. Each view has eight excluded development
localities drawn from twelve already-opened localities. Bootstrap resampling is
at source-locality level, not overlapping-window level. Interval counts are
descriptive, not independent replications or multiplicity-adjusted discovery.

Released detector tracks are not human gold. Coordinates remain image pixels;
obs8/pred12 uses raw annotation stride12, not seconds or historical raw-t50.
Only eligible indexed agent histories are represented, not all visible agents.
Independent selection, calibration and confirmation remain closed. Stage5C and
SMC remain off; no true-3D, foundation or physical-safety claim is supported.
