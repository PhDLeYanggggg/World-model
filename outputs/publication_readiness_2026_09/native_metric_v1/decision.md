# Native-Coordinate Evaluation Amendment

Recorded 2026-09-21, before the new paired readout. Status: adopted through
explicitly delegated author choice, not a retrospectively preregistered study.

## Decision and Authority

After asking what the two options meant, the author instructed: "哪个更容易发论文拿到大结果就用哪个".
I select option 1 because the intended question concerns forecasting overall
motion, not almost exclusively departures from exactly static histories.
Neither publication nor a large positive result is guaranteed. An outcome-driven
metric search is not authorized by this choice. The earlier proposal and every
negative result remain unchanged and available.

The old metric divided each query's displacement error by its past motion scale,
with a 0.001 floor. The source audit exposed a resulting emphasis on static starts.
This amendment follows those observed results and must be disclosed as post-hoc
protocol development. Previously exposed recordings do not become independent
test data by changing the score.

## Fixed Rules

Eight observed annotation steps and twelve predicted steps remain the task.
Raw-frame t+50 stays supplementary, t+100 diagnostic. Membership, split boundaries,
data roles, K=1 prediction budget and leakage restrictions do not change.

ADE is the primary displacement measure; FDE is secondary. Use native coordinates
separately by dataset: SDD annotation pixels and external dataset-local units.
Do not pool raw errors across incompatible units or infer meters or seconds.
Numerical input conditioning is separate from scientific error weighting.

Within each prespecified physical scene, compute
100 * (1 - mean model ADE / mean reference ADE), then average those percentages
equally across scenes. Report every scene's raw ADE/FDE and the worst scene.
Reference CV is fixed. A train-selected stronger reference is a separate control;
its selection must never read the evaluated scene's outcomes.

Compare exactly matched row identities, target masks and budgets. All past-eligible
queries remain indexed. The present auxiliary-source readout reports supported,
masked ADE first and complete-future ADE separately, plus endpoint-supported FDE.
Unknown labels stay unknown. No nonfinite prediction may be hidden by dropping a
row. A missing scene or zero-error reference makes its percentage and the fixed
roster aggregate undefined; report absolute harm, not an epsilon-derived pass.

Keep the old past-normalized metrics and their original aggregation as secondary
diagnostics. No old loss, easy label, threshold or calibration is automatically
valid under the new scale. This amendment authorizes scoring, not an unregistered
new model or safety threshold. Register those before the next fit.

## First Readout, Fixed Before Running

Use only the four already explored auxiliary SDD sites: coupa, deathCircle, gates,
hyang. Keep bookstore, original validation/test and main/external readouts closed.
Verify the previous source audit's complete hash chain and private cost rows.
Score all seven fixed causal controls, the other-source-sites-selected baseline,
and the explicitly future-informed oracle diagnostic. Never deploy the oracle.

Re-score all three existing source-crossfit neural prediction caches, matched to
their original 15,430 static-history complete-label rows. Do not present this
subset as full-population neural evaluation. Reconstruct IDs from existing bound
metadata without opening excluded labels; verify cached costs against the matched
original labels. No model/seed is selected or refitted.

Report a 3,000-resample physical-scene bootstrap with fixed seed 38113 as an
exploratory, conditional interval over four previously explored sites. Preserve
all three seed scores; resampling overlapping rows is prohibited. Independent
calibration and confirmation remain separate unresolved evidence requirements.

Stage5C execution and SMC remain disabled. No deployment, foundation, true-3D,
metric/seconds or submission-candidate claim follows from this amendment.
