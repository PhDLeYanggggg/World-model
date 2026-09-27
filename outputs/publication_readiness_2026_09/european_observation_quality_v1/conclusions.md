# Findings and Research Decision

## What Changed
The audit reproduced all 318,969 target-query histories in 163 admitted source
recordings, covering 12 localities. All 2,551,752 observed boxes match raw
records, and the original packed geometry matches exactly. The source archive
and member hashes are verified. A second complete raw pass reproduces every
diagnostic array and aggregate exactly.

There is a large, concrete context restriction: 282,529 target queries
(88.58%) have at least one partial-history agent among the nearest eight
currently visible neighbors. The old packer omits those agents; its neural
attention and conditioner also require complete aligned neighbor histories.
This restriction was explicit in the old architecture, not a corrupted cache.
The new packer retains their observed samples and valid masks. Mean neighbor
count rises from 6.7624 to 7.3045. The per-locality affected fraction ranges
from 30.19% to 98.30%; this is not only one dense locality.

The versioned neural adapter consistently changes attention eligibility and
past-only input scaling. It adds no trainable parameters, retains the initial
causal baseline forecast and the same bounded output rule. Tests show partial
tokens can influence the model and masked garbage cannot. Existing source
models and deployment are untouched.

## What Did Not Improve Yet
This is input coverage, not prediction lift. There was no research training,
no future outcome scoring, and no independent evaluation. Synthetic gradient
checks are not neural training evidence. The cap experiment remains negative.
Source observations are detector-derived, not human gold.

Missing raw frames alone are not a supported dominant explanation: average
presence within the observed raw-frame prefix is 99.16%. This does not certify
tracking identities. Substantial direction reversals and box-size variation
may reflect detector noise, actual motion, pose or occlusion; they are proxies.
Half-pixel quantization is uncommon in these released coordinates.

OLS4 has slightly lower pooled within-prefix error than finite difference,
but improves only five of twelve localities. Its locality-relative changes
range from -11.67% to +18.95% (positive means lower prefix error). The equal-
locality average is +0.71%, with no independent interval. Two localities account
for 83.40% of indexed queries, so pooled summaries are not evidence of uniform
benefit. OLS6 worsens the pooled check. No smoothing switch is justified by
these observed-prefix diagnostics alone.

## Next Discriminating Experiment
Compare legacy and partial-neighbor context with a matched neural refit, not
another threshold or cap sweep. Freeze original future labels, sampling,
baseline family, loss, parameter count, producer/controller locality exclusions,
three seeds and update budgets. Verify a matching legacy endpoint first; save
new weights separately and freeze predictions before source-held scoring.
Report forecast and gain/harm prediction tradeoffs, not just attention access.
If the additional observations do not help, retain that negative result and
investigate event-bearing support rather than claiming interaction value.

The sparse event-label quality split remains not_run in this input-only audit;
earlier cap-event results remain cached evidence. Independent selection,
calibration and confirmation stay closed. No metric/seconds, true-3D,
foundation-model or physical-safety claim. Stage5C and SMC remain off.
