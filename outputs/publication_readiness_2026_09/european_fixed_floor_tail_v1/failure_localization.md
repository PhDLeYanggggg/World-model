# Post-Readout Failure Localization

## Material Passport

Fresh analysis of frozen development readout and decisions. No refitting, threshold change, candidate selection or independent-role opening.
Primary fixed-roster intervals remain unchanged. Defined-view medians below exclude empty views and are ONLY descriptive, not repaired primary estimates.

## Empty Coverage

Tail-weighted and matched-count policies have 3 empty held views; all are at eu-locality-112.
The registered harm-ratio primary is undefined, not zero. Locality-average intervention may still be positive because other dependent views intervene.

| Policy | Defined /216 views | Median observed harm % | Maximum observed harm % | Median predicted harm % | Median predicted/actual reference | Unknown-label interventions |
|---|---:|---:|---:|---:|---:|---:|
| mse | 216 | 2.0663 | 18.2159 | 0.7139 | 3.0367 | 5177 |
| tail4 | 213 | 1.7229 | 26.9994 | 0.7631 | 4.0339 | 3349 |
| mse_matched_count | 213 | 1.7136 | 26.9994 | not_applicable | not_applicable | 3325 |

Counts above repeat windows across fitted heads and seeds; they are not independent people or scenes.

## Does Count Matching Permit a Ranking Test?

- query_views: 747900
- nonempty_query_views: 170883
- rankable_query_views: 160865
- different_rank_selection_query_views: 28834
- tail_interventions: 307827
- common_interventions: 273245

Rankable means0<K<eligible current-query agents. If K=0 or K=all eligible, matching fixes the selection and cannot test ordering.
A weak matched contrast can reflect little within-query freedom as well as weak ranking. Neither explanation establishes better selection.

## Diagnosis

1. All216 fitting monitors improve: optimization ran, but training loss is not conditional risk calibration.
2. Bounded nonnegative outputs remove the former negative-output clipping failure, yet MSE predicts selected harm below its observed value and inflates the reference denominator.
3. Tail weighting shrinks coverage and net gain. The equal-count ADE interval crosses zero; no stable ordering advantage is demonstrated.
4. Empty views invalidate the registered fixed-roster harm primary;95 tail views still exceed the2% budget. Net easy preservation does not certify positive-harm safety.
5. This contrast freezes the utility head and forecasting bank, so it does not isolate errors of a newly trained trajectory model or prove representation impossibility.

Next: directly model fixed-floor signed budget excess using the SAME repaired producer chain and non-clipped utility floor. Prespecify a matched objective-only contrast and test cancellation/denominator bias; do not open reserved sources or sweep held thresholds.
Only registered future fitting experiments can test that next hypothesis. Existing signed-excess experiments used a different CV-reference/producer setting, so they are relevant negative controls, not equivalent evidence.
Image-local detector silver, obs8/pred12 rawstride12. No metric/seconds/physical-safety/true3D/foundation claim. Stage5C/SMC remain disabled.
