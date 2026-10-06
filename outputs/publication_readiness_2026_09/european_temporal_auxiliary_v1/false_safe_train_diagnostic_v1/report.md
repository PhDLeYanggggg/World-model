# TRAIN False-Safe Component Diagnosis

Fresh inference on the frozen 216 heads and their original 24 TRAIN packets.
This is resubstitution, not generalization or independent confirmation.
Source/seed views and selected occurrences overlap; they are not independent samples.

| Arm | Risk-defined views | Known easy-risk violations | Median risk (%) | Unknown selected occurrences |
|---|---:|---:|---:|---:|
| none | 72/72 | 58 | 5.371021 | 2496 |
| rowmean | 72/72 | 57 | 4.567470 | 1911 |
| temporal | 72/72 | 52 | 4.304727 | 1828 |

## Signed Decomposition

Equal-locality averages of source/seed-view components, divided by actual
selected known easy reference mass, in percentage points. Negative terms
are retained. Undefined support invalidates an aggregate, rather than dropping views.

| Arm | Predicted slack | Harm underprediction | Budgeted reference overprediction | Realized excess |
|---|---:|---:|---:|---:|
| none | -1.352692 | +10.951187 | -0.440167 | +9.158328 |
| rowmean | -1.308834 | +9.146102 | -0.430146 | +7.407122 |
| temporal | -1.275936 | +9.197277 | -0.457331 | +7.464010 |

The first three columns sum to realized excess above the unchanged 2% budget.
These are signed error contributions, not FDE improvements or physical safety probabilities.

## Interpretation Limits

- Failure on TRAIN would rule out pure unseen-domain shift as the sole explanation.
- The decomposition locates numerical error; it does not prove a repair will generalize.
- A reduced global quadratic loss does not guarantee calibration after action selection.
- The frozen full development readout remains failed. No threshold or budget is changed.
- Unknown outcomes remain in inference. Known risk is not a complete-support safety guarantee.
- Detector-silver image-local raw-frame data; no metric, seconds, true-3D or foundation claim.
- Stage5C and SMC remain off.

Job: 37812850; elapsed: 93.706 seconds;
independent scalar sum-identity assertions: 432.
The raw per-head field `scalar_checks` counts scalar summands plus one identity,
not that many independent assertions. This report distinguishes those counts.
