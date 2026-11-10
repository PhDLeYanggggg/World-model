# Compare exchange candidates with the incumbent allocation

The previous listwise experiment forced an exchange on the frozen aligned
ranker's 101 eligible views. Its strongest new pointwise arm reached 53
full-grid dual passes, below 55 for leaving every allocation unchanged.
This experiment tests whether representing the incumbent in each training
slate improves admission and avoids destroying existing passes.

Keep the prior exposed OpenDD development roster: 202 recordings, seven
locations, three seeds (17, 29, 43), eight observations, twelve predictions,
780 forecast edits, original movement caps, chronological credits, and
original 2% checks. Nothing uses a new external label or changes release.
This does not become independent confirmation by withholding location labels
inside a fit; the development results have already informed the design.

Reuse the exact donor features, exchange slates and original dual-pass labels
from the prior listwise experiment. Donor forecast producers and cutoffs exclude
both their donor and the outer scored location. Recover each incumbent's donor
label from the archived transition and option outcome, checking consistency
within its slate. These are targets only, never features.

Represent the unchanged allocation with the existing shared recording-context
block, zero source/destination exchange features, and an explicit incumbent flag.
Each option has the same existing causal features and a zero flag. No new future
or test-derived input is permitted. Existing map and simultaneous peer features
remain unchanged.
The inherited recording-wide context makes this a retrospective batched
allocation study, not a claim of stream-time deployment. The feature path
never reads held future targets, but it can pool the observed histories of
multiple queries from one recording.

Train two equal-width, identically initialized 48/32 MLP rankers:

1. Pointwise BCE over incumbent and options, with equal slate weights.
2. The same BCE plus pairwise logistic loss against the incumbent only when
   their donor pass labels differ. Homogeneous comparisons contribute zero.

Use 400 AdamW updates, the existing learning rate 0.001, gradient clip 5,
32 slates per update, equal donor sampling and identical batches per seed.
Save optimizer, sampler and model state every 100 updates. Use four Torch
threads, one interop thread, zero workers and a PID heartbeat. A one-cell local
pilot measures real compute cost and is resumed into the full 21-cell run.
There is no held-location early stopping, threshold tuning or model selection.

For each model freeze two choices before outcome scoring: best exchange only,
and best of incumbent plus exchanges. Ties favor the incumbent. Both can act
only within the same 101-view eligibility mask; optional admission may reduce
allocation exchanges but must preserve all 780 forecast edits and movement caps.
The incumbent is not CV: it is the unchanged existing forecast allocation.

Report original/full-grid/material (>0.10 pp against both CV and OLS4) dual
passes, risk-undefined/violation counts, rescues, destroyed passes, and number
of allocation exchanges. Include unchanged incumbent, archived aligned ranker,
previous pointwise/listwise controls, and exact uniform exchange on each new
model's own switch mask. Also report uniform choice among incumbent and options
on the fixed eligibility mask. Random comparators are expectations, not learned
or deployable policies; no-edit choices do not establish a risk pass.

Primary contrast: pairwise optional versus both the unchanged incumbent and
same-switch-mask uniform. Mechanism controls: optional versus forced using the
same network, and pairwise optional versus pointwise optional. Report 3,000
paired location bootstrap draws after seed averaging. A selection contribution
requires positive lower interval endpoints against both primary controls, not
merely fewer exchanges or recovery to the incumbent. Report all arms regardless
of sign. No finding alone authorizes changing any frozen external policy.
