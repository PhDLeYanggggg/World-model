# Fitting Failure Taxonomy

This analysis does not change the registered objective, final checkpoints,
thresholds, outcome definitions or scientific split. It uses frozen TRAIN
scores and checkpoint metadata; new inner-held and independent roles are unopened.

|Question|Evidence|Conclusion|
|---|---|---|
|Is there useful fixed-action training headroom?|Previous exact label-aware oracle is positive and budget-feasible in 146/148 record/context views.|A blanket absence of useful actions is false; causal learnability is not established by the oracle.|
|Did the new primal loss learn more benefit?|81.552% versus 25.998% retained eligible benefit; matched utility lower bound +1.824042 pp against exact-gradient.|Yes, on these fitting data. This is not generalization or FDE lift.|
|Did record safety improve?|84/444 supported record views versus 178/444; known all-risk violations 346 versus 216.|No. Better utility and aggregate source coverage do not certify the full policy.|
|Is failure only missing easy support?|Undefined easy views decrease from 55 to 14; 346 known all-risk violations remain.|No. Most failure is observed positive harm, not merely uncertainty.|
|Does preserving net easy error suffice?|Net easy degradation is within 2% in every record view; known easy positive-harm violations are 54 and upper-bound violations 70.|No. Signed net change and positive harm/reference are different constraints.|
|Is dual clipping the primary all-risk problem?|No all-risk dual reaches cap100; maximum25.8845; 344 positive excesses just before final update.|Raising the cap alone is not a supported explanation or repair.|
|Did the saturation mechanism change?|Wrong saturated row occurrences fall from443,891 to25,925; model-specific cost-weighted wrong mass falls.|Consistent with the intended optimization change. Own-final-dual derivatives do not isolate the cause of every prediction error.|
|Where do known violations concentrate?|154/168 locality008 and189/192 locality074 record/head views violate known all-risk.|343/346 violations lie in the two large multi-recording sources. Investigate shared representation, record weighting and constraint convergence.|
|Can all records pass with current candidates?|locality082/51 has no admissible eligible known easy anchor in two contexts.|No under the current supported-selection contract. It remains visible; no record exclusion or budget relaxation.|

## Narrow Repair, Not Another Architecture Search

Keep the useful cost-sensitive objective and freeze the same feature/action
interface. Diagnose whether per-record violation gradients are diluted by
source/row weighting, whether fixed dual steps converge too slowly, and whether
the shared small head can express the required distinctions. A next registered
comparison should change one of these mechanisms, not all at once. Report
per-record positive harm, unknown-label bounds and utility jointly; preserve
both supported and unsupported records. A weight/cap heuristic is not a theorem.

Do not open independent confirmation to choose this repair. Do not repair a
known infeasible candidate set by changing the pass definition after seeing
outcomes. Candidate forecasting/support repair is a separate TRAIN-only task
with its own fixed comparison and provenance.

## Research Gap

The experiment establishes a useful fitting improvement over the previous
decision surrogate, not a deployable world model or a novel cost-sensitive
reduction. Complete-policy constraints, unseen-scene generalization, scene-joint
decision value and independent confirmation remain unproved. Three head seeds
do not replace independently trained forecasters or independent scenes.
Historical failed results stay failed; Stage5C and SMC remain off.
