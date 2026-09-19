# Native Resolution Does Not Repair This Motion Readout

## Result

Increasing observed-image resolution and reducing the flow averaging window
do not provide a reliable probability advantage in this fixed source study.
The input pipeline is verified, but no new neural forecast, selector or
deployment benefit is established. The full M3W research goal remains open.

This is a fresh experiment on the already-explored stationary-history subset,
not the primary benchmark or independent external confirmation. Eight observed
and twelve predicted annotation steps use stride 12 raw frames. SDD coordinates
remain pixels. No seconds, metric, true-3D or foundation claim is supported.

## What Was Actually Run

- Recovered 25,300 native 96x96 past crops from 29 admitted source recordings.
  Every supported reduction exactly matches its previous 32x32 RGB and coverage.
  Decoding/cropping took 354.50 seconds; the private cache is 889.66 MiB.
- Extracted four prespecified resolution/window variants over 23,890 unique
  observed pairs each, 95,560 measurements in total. This took 136.66 seconds.
  The old lowpass control is exactly reproduced for every pair.
- Fitted all 64 fixed logistic probes: four held-source folds, four measurement
  variants, quality/motion inputs and two supervision labels. Fitting took
  111.87 seconds, with at most 1,462 of 4,000 allowed iterations. No convergence
  warning, threshold search, held-driven model selection or row removal.
- Used exact raw future labels only for supervision/evaluation. There are 728
  queries with excursion strictly above 10 pixels, from 115 scoped tracks in
  23 recordings. These are not 728 independent events. Site positive counts
  are coupa 137, deathCircle 204, gates 66 and hyang 321.
- Replayed all 95,560 measurements and 64 coefficient predictions exactly.
  Recomputed 128 train/held score sets and all 64 contrasts/intervals. Input
  poisoning on 128 queries and 48 prohibited training-role checks pass.
  Completed resume adds zero work and preserves 486 scientific artifacts.
- Passed 48 scoped tests. The broad legacy integration suite was not rerun:
  several integrations regenerate historical reports. This is not a claim that
  the entire legacy repository has been freshly tested.

Registration was committed/pushed as `7afef428` before decoding/fitting. Recovery
tools and the real pilot were recorded in `e65808d2`. All required processes
have exited normally. No CREATE job was submitted or freshly inspected.

## The Key Comparison

Positive Brier reduction would favor adding motion to the matched quality
control. Instead, all four mean reductions are negative for larger excursions:

| Measurement | Brier reduction | Conditional four-site 95% interval | Mean held AUROC |
| --- | ---: | --- | ---: |
| 32px, nominal 45px window | -0.001492 | [-0.003437, -0.000378] | 0.4923 |
| 32px, nominal 15px window | -0.001269 | [-0.002459, -0.000298] | 0.4847 |
| 96px, nominal 45px window | -0.001478 | [-0.002258, -0.000617] | 0.4946 |
| 96px, nominal 15px window | -0.001331 | [-0.002576, -0.000284] | 0.4826 |

All four motion variants lose to the training-prevalence constant predictor on
Brier at all four held sites for this label. The quality-only counterparts also
lose to that reference. Training AUROC is 0.7944-0.7994, while held-site mean
AUROC is 0.4826-0.4946. Fitting the source observations is not transferring.

The direct native-minus-lowpass contrast at the same nominal 45px window
worsens Brier by 0.001031, interval [0.000625, 0.001586]. Shrinking the native
window has a small favorable Brier point difference, but its interval includes
zero. This is not evidence that one of the new motion settings is a useful gate.

There are favorable ranking contrasts versus weak quality controls: mean AUROC
increases by 0.0060-0.0125. They coexist with worse Brier/log loss and low absolute
AUROC. For any annotation change, native_w15 beats the prevalence reference on
Brier at three sites, but adding motion still worsens its mean Brier by 0.003105
and mean AUROC. These partial observations are retained, not promoted to a
forecasting claim or silently omitted.

Intervals use 2,000 paired site resamples over only four explored sites with
shared training folds. They are conditional, unadjusted exploratory summaries,
not independent confirmation or a multiplicity-controlled discovery.

## Failure Taxonomy

1. **Corrupt or misaligned old inputs:** not supported by the new exact native
   reduction, source-key alignment and old-control replays. These checks cover
   this pipeline, not every historical dataset or an unverified sensor clock.
2. **Resolution/window size alone:** not a sufficient repair under this fixed
   regional representation and readout. Box support increases to about 99.6%,
   but prediction quality does not. Coverage is not accurate body-motion truth.
3. **Weak transfer or overfitting:** consistent with the large training/held
   ranking gap. The experiment cannot isolate missing intent, scene shift and
   finite-readout limitations as separate causes.
4. **Limited independent motion support:** 728 overlapping positive queries
   cover only 115 scoped tracks and four explored sites. Window count should
   not be confused with broad event or scene diversity.
5. **Representation limits:** regional medians may discard pose/local motion;
   boxes are not segmentations, surround flow is not verified camera motion,
   and resolution also changes the pixel lattice and polynomial neighborhood.
   This is not a proof that images contain no predictive information.
6. **Annotation/protocol limits:** offline supplied/interpolated annotations
   remain distinct from sensor-as-of observations and human-gold motion labels.
   Exact-label handling repairs the previous numerical threshold ambiguity,
   not the underlying scientific limits of the dataset.

## Consequence For The Research Route

Follow the registered decision: do not spend another full trajectory-fit budget
on these same four motion measurements or tune a deployment threshold from the
held results. No new neural trajectory model was trained in this comparison.
The current evidence does not establish an independently verified deployable
neural improvement; historical Stage26/37 scores remain exploratory.

The next useful step is to return to the broader approved source training
population, inventory distinct moving/turning/stopping episodes and the
existing full-source predictors, and test whether a candidate has transferable
directional headroom before adding another safety gate. Reuse existing fits
where their protocol matches; do not duplicate them or change the primary task
to make this stationary subset easier. Further source diagnostics cannot stand
in for the still-missing independent calibration and confirmation evidence.

The full goal still requires strong public baselines, useful predictors,
same-predictor independent/scene/joint decisions, matched intervention budgets,
three-seed primary results, independent support and a defensible paper package.
No Stage5C execution, SMC, deployment upgrade or submission-readiness claim.

[Complete metrics and per-site results](complete_results.md)
and [aggregate figure](motion_information.svg).
