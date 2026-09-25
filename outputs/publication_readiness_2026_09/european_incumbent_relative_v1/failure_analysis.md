# Failure Analysis

## What The Registered Test Repaired

The preceding matched-producer controller had three negative all-ADE intervals
versus the original stopping controller; all three lost more useful original
switches than they gained by removing harmful ones. This round adds the causal
incumbent choice to both matched arms and tests incumbent-relative supervision.
The worst group-mean regression narrows to 0.00893%, and most incumbent-relative
views improve. This supports the value of preserving useful original decisions,
but does not isolate the regression loss from the changed reference-based gate.

## Remaining Negative All Case

`fold2_seed17_all_controller0` retains a negative all-ADE interval:
[-0.016592%, -0.000806%]. The source-equal error-change terms, divided by each
locality's original-policy error before averaging, are:

| Term | Percentage points of incumbent error |
|---|---:|
| Added intervention harm | 0.07039203 |
| Added intervention benefit | 0.07782204 |
| Removed useful-intervention loss | 0.01730810 |
| Removed harmful-intervention benefit | 0.00094353 |

Added actions net -0.00743001 percentage points of error, but removals net
+0.01636457. Together they produce +0.00893456% degradation. The accounting
matches the independently recomputed metric, not a posthoc model change.

## Why Add-Only Still Is Not A Deployment Certificate

In the same group's hard subset, added harm0.05559073 exceeds added
benefit0.03343662. Add-only therefore degrades that hard view by0.02215411%, with
interval[-0.038711%, -0.005146%] for improvement. Avoiding removals cannot repair
these incorrectly added actions. Across all views, add-only hard gain ranges
from-0.033642% to+1.372422%; there are20 positive and one negative intervals.
All-subset positivity does not imply every hard source/subset improves.

Remove-only has30 negative and zero positive all intervals. It is not a useful
standalone update in this registered comparison. The matched-input floor head
still wins some views: incumbent-relative versus that control has19 positive and
eight negative all intervals. We cannot claim universally better cost estimation.

## Risk Estimation Still Fails

The incumbent-relative gate admits overrides whose predicted harm/reference
ratio is at most2%, but65/139 supported selected locality/views exceed2% in
realized labels.110 underpredict the ratio. Five of144 locality/views lack a
positive realized event-reference denominator; they are not zero-risk passes.
Reference costs differ between the two arms, so raw ratio comparisons are not
calibration improvements on the same estimand. Net easy preservation is real in
this readout, but is not calibrated positive-harm control or physical safety.

Fixed training-batch risk loss decreases for34/36 incremental heads and31/36
floor heads. That does not establish validation improvement, calibrated
uncertainty, convergence or better dynamics. The utility traces use changing
training batches and cannot be compared as validation curves.

## Changed Next Action

Preserve the incumbent and investigate the newly added interventions, especially
the failing hard-source slice. Register a same-forecaster comparison of individual
versus scene-level joint additions with matched counts/risk and outcome-independent
controls before running it. Do not select a seed, source rotation or new removal
threshold from these outcomes. Independent calibration/confirmation remains
closed; new dynamics training and paper readiness remain unproved.
