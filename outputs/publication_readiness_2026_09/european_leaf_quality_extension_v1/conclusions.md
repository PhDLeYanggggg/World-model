# TRAIN-Identity Leaf-Quality Extension

## Question and Controlled Change

The preceding support diagnosis associated large positive-link validation errors
with quality coordinates outside the TRAIN leaf range. We tested one extension:
clip those seven past-only coordinates to each leaf's known TRAIN min/max before
the frozen cost link. Routing, coefficients, mean constraints, B/R/ER, feasibility
projection, policy, thresholds and the 2% risk budget stay fixed. Clipping is not
assumed to reduce risk monotonically. No validation bounds or new fitting.

All 72 heads' known TRAIN predictions are bit-identical. The original forest,
additive, Poisson and squared-cost controls replay exactly. New extension
inference and policy evaluation are `fresh_run`; controls and checkpoint lineage
are `cached_verified`. Transfer and independent confirmation are `not_run`:
the registered advance screen failed, and reserved roles remain closed.

## What Improved

The extension reduces normalized signed-score MSE versus the squared-cost head
by 0.126005, nominal 95% locality interval [-0.246783, -0.030947]. It improves
63/72 heads and 11/12 locality means. Relative to the original forest, change is
-0.021852 [-0.057900, -0.000175], with 50/72 heads and 9/12 locality means improved.
These are dimensionless cost-score errors, not trajectory ADE/FDE improvement.

Raw, pre-projection MSE changes by -12171.302627 versus squared cost. The large
value reflects extreme unprojected cost tails, not executed physical motion.
Raw error versus original is -0.008068 [-0.055300, 0.036872], inconclusive.
No-clipping rows have exactly zero change. The one-factor intervention therefore
supports an extrapolation contribution to the fixed model's prediction error;
it does not establish a universal cause or a new architecture contribution.

## What Did Not Improve Enough

Almost all projected MSE repair is on the extension's unselected cohort:
-0.126005348 versus -0.000000011 on selected rows. The selected-row interval
overlaps zero. Every head changes predictions, but only 10/72 change any actions.
Selected occurrences fall from 96,720 to 96,718; unknown selected occurrences stay
926. These are repeated model-view occurrences, not unique agents or samples.

Full utility versus unextended cost changes +0.00001480% of full known reference
error mass, interval [-0.00000312, 0.00004134]. The matched-count change is
+0.00000997% [0, 0.00002446]. Neither lower bound is strictly positive. Utility
versus original is nominally positive, but lower than additive and Poisson.
MSE versus additive is +0.010294 [-0.001532, 0.026280], not supported improvement.
All four controls remain in the comparison; none is removed to declare success.

## Safety Failure Remains

The policy has four known-label and seven completion-upper violations, unchanged
from the cost control. Worst selected easy-risk upper is 5.7195%, versus the
original's 5.4058% and the frozen 2% budget. Complete support is 36/72; easy-risk
denominators are defined for 47/72, down from 48/72 for unextended cost.
Undefined support is not counted as a safe zero.

The four known failures in localities 112/124 retain exactly the original and
unextended cost actions. They cannot be excused by missing outcomes. The other
three failures in 067 involve 9-10 unknown selected outcomes per head, preserved
in completion bounds. These bounds are not observed easy degradation. The
maximum upper remains unchanged even though one 067 head changes its actions.

The registered relative advance screen and the absolute supported-risk screen
both fail. **No deployment promotion.** Repairing unselected prediction tails
alone does not repair the errors that determine harmful selection.

## Execution and Verification

- Registration commit: `c03c8de0`; configuration and protocol unchanged after run.
- Real pilot: 50.37 seconds, peak RSS 7.111 GB, 299 scalar checks.
- Full run: 72 heads, 426.82 seconds, peak RSS 10.297 GB, 586.975 MB read and
  hash-checked from owned CREATE checkpoints; no new Slurm job or numerical cache.
- Native arm64, four compute threads, zero DataLoader workers.
- Ten focused tests, 21,528 in-run scalar checks, 12,660 independent scalar and
  bootstrap checks pass. The legacy whole-repository test suite was not rerun.
- No new training, neural dynamics learning, threshold selection or forecast change.

Evidence: [complete receipt](complete.json), [independent verification](verification.json),
[all comparisons](findings.md), [machine-readable findings](findings.json),
[protocol](protocol.md), [reproduction guide](reproduction_zh.md).

## Scientific Limits and Next Priority

The 72 heads reuse 12 already exposed development localities; recording-held
validation within locality is not held-locality confirmation. All 3,000-resample
locality intervals are nominal and not adjusted for the development search.
Three cost-head seeds are not three new end-to-end predictor trainings.
Image-local detector-silver obs8/pred12 at rawstride12 is not metric, seconds,
human gold, physical safety, true 3D or foundation evidence. Stage5C/SMC stay off.

This closes the unbounded quality-extrapolation explanation as a sufficient
policy repair, not as a contributing error mechanism. The next candidate
question is temporal benefit/harm target identifiability using the existing
TRAIN/development trajectory labels. Prior audits found early/late error-order
reversals, while adding ordered history features alone failed. A temporal target
test would need a separate fixed protocol and matched controls; no such test is
claimed or started here. Do not repeat a cutoff sweep or drop unknown labels.

Independent calibration/confirmation, controlled positive scene-joint outcomes,
credible strong forecaster comparisons and a supported neural contribution are
still missing. These results are a bounded development control, not a successful
world model or a submission-ready claim. The long-term goal is currently paused;
this report closes the already-running experiment without launching its successor.
