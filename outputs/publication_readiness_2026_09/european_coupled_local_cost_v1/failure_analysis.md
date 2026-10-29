# Coupled Local Costs Improve Some Fitting Utility, Not Complete Safety

## What Is New And What Is Not

All 288 source/mode/comparator-seed groups complete and replay exactly,
including fresh recomputation of every neighbor, prediction and action.
This is a nonparametric cost-estimation control, not new neural optimization
or trajectory dynamics. It reuses the original forecasters and compares
against the freshly trained paired heads from the preceding experiment.
The deterministic retrieval policy repeats across comparator seeds; it is
not three independently trained new models. These are fitting-only results.

## Main Results

| Mixed-bank estimator | Known ADE gain vs floor (%) | Full risk support /444 | Undefined easy views | Known all/easy violations |
|---|---:|---:|---:|---:|
| Original fresh known-only MLP | +0.580799 | 77 | 62 | 85 /234 |
| Local known-only retrieval | +1.334797 | 6 | 6 | 276 /321 |
| Local completion retrieval, primary | +0.830158 | 30 | 15 | 237 /273 |
| Global completion mean | +0.004133 | 291 | 123 | 6 /9 |
| Original within-recording shuffled control | +0.812390 | 0 | 12 | 369 /363 |

The primary's +0.830158% known utility has a fitting locality interval
[+0.234853,+1.429495], not independent evidence. Its full completion-lower
change versus the known-only MLP is only +0.006377pp, CI [-0.721776,+0.635341].
The matched lower-utility condition also fails. Coupling the outputs and
using a different estimator do not repair the complete gate.

## Failure Taxonomy

1. **Moment consistency is necessary but insufficient.** The construction
   preserves common reference estimates, EH<=H, ER<=R, B<=R and B+H<=E.
   Unlike the older reference-offset analogue, no independently corrected
   pair can violate EH<=H. Nevertheless, 273 repeated recording views have
   known easy-risk violations. Algebraically consistent predicted moments
   are not calibrated conditional risk estimates.

2. **Higher mean gain accompanies many more interventions.** The primary
   selects 303,363 repeated row views, including 3,474 unknown outcomes,
   versus 80,414 /932 for the known-only MLP. Unknown labels are not declared
   harmful; their finite-completion accounting remains. Completion training
   in this retrieval family reduces interventions and violations relative
   to local-known, but does not make either policy safe.

3. **A conservative global mean trades utility for abstention.** It supports
   more recording views but leaves 123 easy-risk estimates undefined and
   achieves almost no utility. It is not a replacement winner. Empty or
   unsupported selected risk is never treated as zero or a passed certificate.

4. **The fine-neighborhood contribution is not established by the means.**
   The original shuffled control is close to the primary in known gain.
   It preserves donor-recording target profiles, but not each causal action-
   eligibility stratum, so it is not a clean fine-motion-only intervention.
   The separately registered stratified mechanism control preserves those
   strata before comparing actual paired paths. Its result does not replace
   the original primary or reopen any independent data.

5. **There is a small, limited neural-candidate signal.** Primary mixed versus
   physical-only completion-lower gain is +0.040875pp, fitting CI
   [+0.001154,+0.086588]. The known-only contrast is +0.053131pp, CI
   [+0.013893,+0.098689]. This is retained rather than hidden, but the policy
   is unsafe and the intervals are exposed-development, nominal and not
   corrected for repeated research selection. They do not establish a
   deployable neural contribution or independent world-dynamics improvement.

## Interpretation And Next Action

The current failure is not solely the MLP parameterization or broken cost
ordering. The compressed Euclidean metric may match irrelevant aspects of
history, and a finite donor average can miss rare harmful outcomes. Those
are hypotheses, not identified causes or an irreducible-information theorem.
The normalizer and upstream forecasters saw fitting data, so excluding donor
recordings does not turn this into whole-pipeline OOF.

Complete the fixed stratified-label comparison before claiming that precise
past-motion matching explains the utility. If it cannot beat a control that
preserves recording/action-support marginals, do not launch another k or
threshold sweep. Audit an actually different causal input or forecast proposal,
including whether supplied camera calibration metadata can be correctly
aligned to raw midpoint coordinates. Calibration metadata alone is neither
matched RGB nor verification of ground-truth metric trajectories.

Independent roles and deployment stay unchanged. The primary task remains
oblique detector-silver pixels, obs8/pred12/stride12 raw frames, not metric,
effective seconds, human gold, true 3D or foundation capability. No Stage5C
or SMC. The A-conference research goal remains unachieved and active.

## Completed Stratified Mechanism And Input Audit

The additional24-group mechanism now completes and replays exactly. Real
matching minus the recording/action-eligibility-stratified shuffle is
+0.140110pp on known utility,CI[-0.226939,+0.496201], and-0.122182pp on
completion-lower utility,CI[-0.482149,+0.230146]. The matched lower interval
also crosses zero. Fine matching is not stably supported here; this does
not prove feature insufficiency or that recording identity is the sole cause.
See stratified_readout.md. Actual weights use the fixed full causal fitting
pool, not exact per-exclusion source rebalancing; see model_data_card.md.

A subsequent metadata-only projection audit covers all14 existing fitting
cameras, without trajectory labels or policy scoring. Point/polygon filename
resolution differs for cameras82/126; inverse polygon-match RMSE for82 is
423.622pixels under the explicit filename-rescaling convention. Other cameras
also have nontrivial point-fit/leave-one-point residuals. These are internal
resource checks, not independent surveyed calibration or proof that such
errors caused the selection failure. The raw model uses box midpoints while
the publisher mapping code uses box bottoms. Verify frame resolution and
observation convention before a new projective proposal; do not silently
replace old inputs or reinterpret results as metric. See
../source_projection_audit_v1/readout.md.
