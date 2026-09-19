# Fixed-checkpoint training-gradient diagnostic

The preceding 24 importance-corrected heads remain negative on the explored
source folds. All logged training gradient norms exceed the existing cap of 5,
but this does not establish that clipping caused poor prediction.

Before changing another training factor, inspect every final checkpoint, all
four training complements, both arms and all three seeds. There are no optimizer
steps, model selection, new held predictions or new deployment in this audit.

1. Accumulate the full uniform-training objective gradient in batches of 128.
   Partition static and nonzero targets, dividing BOTH contributions by the
   total number of training rows. Future targets appear only in this loss audit.
2. For each frozen model, draw 128 batches of 64 from each of the uniform and
   episode proposals. The latter uses the exact inverse-probability factors.
   Use an independent fixed probe RNG; compare means before/after cap-5 clipping.
3. Record component directions, norms and layer energy. Finite Monte Carlo
   discrepancy is not a proof of estimator bias. Clipping analysis is not an
   Adam update analysis, a training intervention or a held-out causal result.
4. Evaluate fixed training-output multipliers 0, .25, .5, .75 and 1. These are
   a shrink-to-CV diagnostic, not a deployment-policy search.
5. Check guarded training access, checkpoint hashes and zero mutation. Save
   per-model receipts for resumption and permit independent exact replay.

The approved primary protocol, folds, units and risk allowance are unchanged.
This is offline supplied-annotation SDD source development: pixel coordinates,
8 observed / 12 predicted steps, stride 12 raw frames. It does not establish
seconds, metric calibration, sensor-as-of validity, true 3D or foundation-model
ability. Main evaluation, outer folds and bookstore remain unscored. Existing
predictors and their raw data/checkpoints are not republished. No Stage5C or SMC.
