# Internal Locality-Held Cost Learning

Fresh training, causal decisions, internal development readout and independent accounting checks. Inputs/checkpoint ancestry are cached_verified. Independent confirmation and deployment are not_run.

72 unique paired fits,144 heads,288,000 updates; training 252.88s. 216 directional views are repeated reuse of those heads, not216 independent experiments. The evaluated fitting locality contributes no statistics or labels to its head; roles rotate across views. All12 localities were already development-exposed. Outer-held and independent roles remain closed in this experiment.

## Registered Primary

Nonlinear minus affine normalized signed-score MSE: 0.07709021; nominal95% locality interval [-0.10208869181626862, 0.30087182198670326]. Lower is better. The affine arm has affine logits and the same nonlinear bounded decoder, not ordinary linear regression.

Same-count nonlinear versus affine ADE gain: 0.656286%; nominal95% interval [0.342668823911117, 1.0055798237422657]. Both arms rank only their own admissible pools; the common count does not make the pools or predicted risks identical.

| Policy | ADE gain over floor (%) | FDE gain (%) | Easy gain (%) | Hard gain (%) | Intervention (%) | All-risk violations / defined | Undefined all / easy |
|---|---:|---:|---:|---:|---:|---:|---:|
| affine | 0.470499 | 0.673949 | 2.341345 | 0.404704 | 13.458670 | 162/216 | 0/0 |
| affine_matched | 0.433867 | 0.623301 | 2.051815 | 0.374054 | 11.453256 | 158/216 | 0/0 |
| floor | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0 | 216/216 |
| intercept | 0.345374 | 0.546324 | 4.194903 | 0.002406 | 20.557815 | 94/189 | 27/27 |
| neural_unprotected | 7.730662 | 12.282454 | -15.521219 | 12.990197 | 100.000000 | 216/216 | 0/0 |
| nonlinear | 1.907151 | 2.729208 | 1.641166 | 2.416606 | 23.214260 | 184/216 | 0/0 |
| nonlinear_matched | 1.081715 | 1.547036 | 1.239072 | 1.291170 | 11.453256 | 168/216 | 0/0 |

Table percentages are equal-view descriptive means; bootstrap first averages within locality. A missing denominator stays undefined. Conditional means, unknown-label interventions, per-locality scores and seed breakdowns are in summary.json; none is silently promoted to a full-roster result.

## Statistical Scope

3,000 paired bootstrap draws over12 locality means, after averaging dependent role/seed views. Three forecaster seeds are reported but share data. Intervals are nominal developmental summaries, not selection-adjusted or independent confirmation. Accuracy and positive-harm risk are distinct. The selected-floor risk denominator and2% budget have not changed. Zero intervention is not risk certification.

## Verification

72 training-source preprocess/checkpoint pairs verified; 37,800 independently reduced metric values, 648 query-balanced score vectors and 747,900 matched query counts checked. 5,614,596 known and 126,846 unknown repeated occurrences; 318,969 unique row IDs, still overlapping and not independent. One full paired fit replays exactly; all216 decisions and the complete numerical readout replay exactly. Only the first pair was independently retrained, not all144heads.

## Scope Limits

This trains conditional cost heads, not a new world-dynamics forecaster. Observation8/prediction12, stride12 raw frames, image-local detector-silver. No metric/seconds, human-gold, true3D, foundation, deployment safety or submission-ready claim. Stage5C/SMC remain off.
