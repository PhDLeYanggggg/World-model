# Fitting-Only Switch Diagnostic

## Question

The completed fixed-occurrence experiment lost more benefit than it avoided
positive harm. Determine whether useful interventions are lost through causal
eligibility, utility scoring, all-risk rejection or easy-risk rejection, and
whether observed false-safe errors remain on the sources used to fit the heads.

This is not another utility-ordering experiment: that control already exists.
It is a label-complete decomposition of the current frozen scores on their
fitting sources. It cannot establish the cause of held-development failure.

## Frozen Scope

- All 108 registered contexts; unchanged 4/4/2/2 producer/controller/fitting/held
  source roles within the already-opened 12 development localities.
- Use only the two fitting sources per context for label accounting. Do not read
  any new held outcome or independent selection/calibration/confirmation source.
- Frozen forecasters, floor, ridge utility, original raw risk, and both completed
  fixed-occurrence arms. No parameter updates, threshold selection or new actions.
- Recompute causal scores on fitting rows, not by reusing misaligned held scores.
  Verify checkpoint hashes, row IDs, source roles, input schema and label hashes.
- Preserve the original 2% risk target. No selected denominator replacement.

## Readout

1. Apply the existing independent sign screen using past-only scores. Partition
   rows, in this order: not moving; unsupported; nonpositive utility; all-risk
   rejected; easy-risk rejected; admitted. The partition is exclusive; its order
   is descriptive and does not assign unique causal blame for correlated failures.
2. Report source/query-balanced retained and missed benefit, harm, utility MSE and
   bias, observed false-safe mass, selected all/easy positive-harm ratios.
3. For every source/recording/frame query, record defined, violating, undefined and
   unknown-selected risk statuses. Unknown labels are never imputed as zero;
   abstention and zero reference denominators are undefined, not certified safe.
4. On complete-label queries only, use the sign screen's intervention count for
   a descriptive ranking comparison. Rank the same causal eligible pool by frozen
   utility and, separately, by realized signed gain. The latter is an oracle upper
   bound that ignores risk constraints, not a deployable model or training input.
   Report zero-count/no-choice/incomplete query exclusions, regret and whether
   utility-only ordering would worsen the sign screen. No joint solver is run.
5. Report each fitting locality, context and equal-context descriptive mean.
   Repeated windows and overlapping fitting contexts are not independent samples;
   no in-sample significance or generalization claim will be made.

## Execution and Verification

Register and commit this protocol and implementation before the first readout.
Run one measured local pilot, then all 108 groups and a full exact replay. Native
arm64 Python, CPU4, interop1, workers0; retain a 10GiB reserve. Write one resumable
group receipt and heartbeat per group. No new row cache is needed.
Verify partition sums, label support, oracle inequalities and query coverage;
record tests, replay scope and result provenance. Upstream data/checkpoints are
cached_verified; this diagnostic is fresh_run; retraining and independent testing
are not_run. A follow-on intervention requires its own registration based on the
fitting evidence. Do not tune against the previous development readout.

Image-local detector-silver, obs8/pred12 with stride12 raw frames. No metric,
seconds-level, true-3D, foundation or human-gold claim. Stage5C and SMC remain off.
