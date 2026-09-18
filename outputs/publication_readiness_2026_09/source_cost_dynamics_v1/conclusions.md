# Source Trajectory Cost Comparison: Negative Result

## What Ran

`fresh_run`: all 60 fixed neural fits, 120,000 optimizer updates, 44,864 parameters
per model. Two losses, two image arms, five physical-site folds and three seeds.
Summed fitting time was 5,878.56 seconds; the full-run log spans 106.62 minutes,
including one interruption and recovery. The 100-update pilot is included in the
fixed budget. No model, final checkpoint or threshold was selected from scores.

`cached_verified`: 22,374 complete stationary-history queries, 726 scoped agent
IDs and 36 videos from the five approved original SDD training sites. The same
6,460 incomplete-label stationary queries remain unscored, not negative. All 51
unsupported spatial contexts remain exact baseline predictions. Main training
and sealed development, calibration and confirmation roles were not opened.

## Result

The tested cost-alignment hypothesis is not supported. Direct ADE training does
not yield useful new trajectories compared with log-ADE, and RGB does not have a
stable incremental benefit. All 60 fits have worse held-site ADE than stationary
CV, both without control and with the fixed probability guard.

| Loss | Input | Uncontrolled ADE gain vs CV | Fixed-guard gain vs CV |
| --- | --- | ---: | ---: |
| ADE | Mask/geometry | -1.665340% | -0.004907% |
| ADE | Past RGB/geometry | -1.698347% | -0.017120% |
| log-ADE | Mask/geometry | -1.606832% | -0.004134% |
| log-ADE | Past RGB/geometry | -1.568712% | -0.015508% |

These are ratios of equally weighted physical-site mean parent-normalized ADE,
averaging seed losses rather than paths. Conditional 2,000-resample site
intervals for all four uncontrolled gains are wholly negative. Equal-agent
sensitivity is also negative. See [all scores and intervals](results.md).

RGB-minus-mask contrasts are -0.032466% under ADE and +0.037517% under log-ADE;
both conditional intervals cross zero. ADE-minus-log-ADE uncontrolled contrasts
are -0.057582% for mask and -0.127632% for RGB, also with intervals crossing zero.
These small differences do not justify selecting an image arm or loss.

## Failure Taxonomy

1. **Training improvement is absent, not only transfer improvement.** Every
   final model also has worse ADE on its complete training complement. Its gain
   ranges from -2.054110% to -1.074482%. This differs from the previous binary
   classifier, where better training fit failed to generalize. It would be wrong
   to diagnose this trajectory experiment only as overfitting or domain shift.
2. **Batch difficulty can create a misleading loss curve.** Replaying the exact
   training sampler recovers the same draw counts for all 15 site/seed streams.
   Initial predictions have exactly zero excess loss over same-batch CV in all
   60 fits. Final logged excess normalized ADE is positive in all 60, ranging
   from 0.00954 to 0.02576. Raw batch-loss fluctuations are not learning progress.
3. **Optimization remains unresolved.** All logged ADE batches and 98.41% of
   logged log-ADE batches trigger gradient clipping. This is a sampled-trace
   observation, not a measurement of every batch or proof that clipping caused
   failure. The fixed-budget run does not establish convergence. Nonsmooth loss
   around a large exact-zero target mass, decoder scale and insufficient useful
   conditional signal are distinct remaining hypotheses.
4. **Easy cases acquire unnecessary movement.** The uncontrolled equal-site
   easy absolute harm is about 21.73-23.92 parent-normalized units, or
   0.02173-0.02392 annotation pixels. The baseline error on these rows is exactly
   zero; percentage easy degradation is undefined, not a passing <=2% value.
5. **The guard reduces damage, not creates information.** Equal-site switch
   rates are 0.264% for the mask classifier and 0.953% for RGB. Gains remain
   negative. Its threshold0.9 is a frozen, uncalibrated diagnostic, not a safety
   certificate. Guarded RGB/mask differences also change the classifier gate;
   they cannot isolate image content.
6. **The context bound does not explain away the entire gap.** The
   future-informed containing-ball oracle has 99.67-100% error-reduction
   headroom by site. This is only a geometric ceiling, not a realizable causal
   model. The actual learned paths offer very little binary-oracle headroom.
   More permissive routing cannot manufacture a trajectory the head did not
   learn. Larger models or another threshold grid are not yet justified.
7. **Information and annotation support remain limited.** The inherited audit
   found mostly generated offline annotation histories and many very small
   future annotation changes. No result here establishes a human walking-start
   label, visible-intent prediction or an intrinsic impossibility of forecasting.

## Verification

- 60 exact checkpoint-to-prediction replays.
- 15 four-way checks of row draws, feature normalization, objective scale and
  same-arm probability guards; every held physical site is excluded from fitting.
- 181 immutable artifacts and the final report unchanged on completed resume;
  zero added fits or updates.
- Real recovery after the old process/handle disappeared: 43 completed fits
  preserved; the next fit resumed its optimizer/RNG state at step600.
- 58 focused tests pass. Full legacy suite has not been rerun.
- Both generated figures were visually inspected; caches, images, checkpoints
  and per-row predictions remain private and ignored by Git.

Report SHA256:
`6e9abdc5e3e8edaea588a1fa7f37f212578a238dea813f1a33b570bea45a6eb1`.
[Reproduction commands](reproducibility.md), [verification receipt](verification.json),
[learning trace](learning_trace.svg), [forecast comparison](trajectory_gain.svg).

## Next Action And Research Status

Before another held-site matrix, use only an admitted source training complement
for an exact-replay, small memorization/optimization diagnostic. Separate whether
the head can fit nonzero trajectory targets at all from its tendency to perturb
zero targets. Test one output-scale or optimization factor at a time while
keeping the scoring coordinate and primary metric unchanged. A successful
training-only check is an engineering prerequisite, not a forecast result.

The baseline-relative joint-intervention contribution is still unproved. A
useful candidate trajectory family must precede calibration and policy claims.
Independent scene calibration, public-method comparisons under the approved
protocol and sealed confirmation are not supplied by this experiment. The
long-term research goal remains active and unmet; this is not submission-ready.

There is no new deployment. The experiment is source-internal, stationary-query
and pixel/raw-frame only. Its +144 raw-frame source horizon is not time-equated
to the main native observe8/predict12 task. Five exposed sites and overlapping
training folds yield conditional sensitivity, not independent confirmation.
No metric, seconds, true3D, foundation, Stage5C execution or SMC claim is made.
