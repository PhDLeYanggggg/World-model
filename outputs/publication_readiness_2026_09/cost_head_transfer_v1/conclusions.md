# Frozen Cost-Head Transfer to EqMotion

## Evidence Status

Completed 2026-09-22. `fresh_run`: transferred cost scores, fixed decisions,
outcome reductions, checkpoint replay, separate arithmetic verification and
post-readout feature/support forensics. `cached_verified`: twelve EqMotion
predictors, thirty-six Transformer-trained cost heads and causal source inputs.
`not_run`: EqMotion-specific cost-head fitting, independent calibration,
independent confirmation and deployment. Registration commit: `76cf3536`.

The task is eight observed and twelve predicted sampled annotation steps on four
design-exposed SDD sites. SDD stride is twelve raw annotation frames. Units are
annotation pixels, not verified meters or seconds. This is neither a true 3D nor
a foundation world model. No Stage5C or SMC execution.

## Fixed Question and Result

Can the existing bounded-fraction cost head transfer to a stronger predictor
without retraining, changed normalization or a threshold search? All arms,
three seeds and controls were fixed before this transferred outcome readout.
The choice of the fraction arm followed earlier development evidence; this is
not retrospective promotion of the previous study's secondary result.

**The combined primary criterion fails.** Fraction-strict improves equal-site
ADE over causal CV by 3.32610%, scene bootstrap CI [2.25060%, 4.66805%], but its
positive-easy error increases 3.78188%. The paired advantage over direct-strict
is just 0.14597 percentage points, CI [-0.55555, 0.98260]. The interval does not
establish an advantage. Two of three seeds exceed the 2% easy limit.

| Fixed policy | ADE gain vs CV (%) | Positive-easy degradation (%) | Complete exact-CV harms, query/seed instances |
|---|---:|---:|---:|
| CV floor | 0.00000 | 0.00000 | 0 |
| Uncontrolled EqMotion | 11.04350 | 33.36170 | 34,584 |
| EqMotion with past-stop veto | 12.18575 | 27.63510 | 21 |
| Direct-native strict + stop | 3.18013 | 4.38976 | 0 |
| Bounded-native strict + stop | 6.42425 | 13.78037 | 6 |
| Bounded-fraction strict + stop | 3.32610 | 3.78188 | 0 |
| Direct-native matched count | 4.59005 | 5.78483 | 13 |
| Bounded-native matched count | 4.80337 | 8.03300 | 0 |
| Bounded-fraction matched count | 4.93853 | 7.65561 | 0 |

Matched counts are fixed offline batch controls, not prospective deployment.
Fraction-minus-direct at matched counts is 0.34848 pp, CI [-0.22404, 0.92101];
fraction-minus-bounded-native is 0.13516 pp, CI [-0.08698, 0.30571]. Neither
demonstrates stable improved ranking. The full registered policy family,
FDE, scene, seed, tail and support reductions is retained in `analysis.json`.

| Seed | Fraction-strict ADE gain (%) | Positive-easy degradation (%) | Switches |
|---|---:|---:|---:|
| 17 | 2.97653 | 2.77909 | 13,012 |
| 29 | 1.42282 | 0.90287 | 7,543 |
| 43 | 5.57896 | 7.66368 | 21,867 |

Seed29 cannot be selected after observing these results. Seed43's worst site,
gates, has 17.75202% easy degradation. No policy or threshold is changed.

## Safety, Missing Outcomes and Comparability

Fraction-strict has no observed harm among complete futures exactly predicted
by CV. That is not population safety: its selections include 818 unknown-ADE
and 7,479 incomplete-future query/seed instances. Unknown outcomes are not zero
error, and repeated query/seed instances are not independent agents. The
confidence intervals resample four physical sites 3,000 times after averaging
seed errors, not coordinates or overlapping windows. They describe conditional
development uncertainty on only four already-explored sites.

The current easy/hard cutoffs come from complete cost-training outcomes using
frozen training-only quartiles. The earlier native predictor comparison used
its all-supported fitting population. This explains why uncontrolled EqMotion
has 33.36170% easy degradation here versus 35.24968% in that comparison; neither
its predictions nor the earlier score has been altered. Subset percentages
must not be silently compared across these two definitions.

## Failure Forensics

1. **Conditional harm is underestimated.** Every one of twelve fraction-strict
   site/seed views underpredicts realized switching harm. Realized/predicted
   mean harm ratios are approximately 3.38 to 9.81. These are continuous cost
   estimates, not calibrated event probabilities.
2. **Predictor-dependent features move out of fitting support.** Identical
   causal rows have identical past, neighbor and shared features. Changes are
   confined to candidate-rollout/disagreement columns. The largest shift is
   the candidate's first predicted lateral coordinate (feature331).
3. **The global extreme is not the whole explanation.** Very large standardized
   values often occur on stopped rows already vetoed by the rule. A separate,
   future-free audit of actual fraction-strict selections still finds 2.23% to
   72.33% of rows beyond ten fitting standard deviations on feature331,
   depending on site/seed. This establishes extrapolation in decision support,
   not that extrapolation alone caused the observed harm.
4. **No constant-column or row-alignment repair was identified.** No changed
   feature uses a standard-deviation floor in any view. Blindly clipping those
   extremes or refitting normalization on held-source rows would change the
   experiment and could hide the problem; neither was done.
5. **More accurate prediction is not sufficient.** A head trained on Transformer
   predictions cannot be assumed calibrated for EqMotion's different forecast
   distribution. The negative transfer does not prove that a properly retrained
   EqMotion-specific or mixed-predictor cost head cannot work.

## Next Necessary Experiment

Construct EqMotion pair-excluded producers before fitting a new cost head.
For an outer held source A and a cost-training row from B, its predictor must
exclude both A and B from fitting and normalization. The existing outer-held
predictor is suitable for A's evaluation, not for manufacturing clean training
predictions on every other site. Reuse the established source population,
three seeds, 4,000-update budget and source-uniform draws. Preserve independent
calibration/final-test roles and do not tune this policy on the transferred
readout. A later fixed comparison can test predictor-specific or mixed cost
training and robust candidate-feature representations. Its outcome is not_run.

## Reproduction

The analysis SHA256 is
`15959b1c2d0583d45e58a6187be4393db155862921860a5b9b750dd5f9d0af1b`.
All36 heads replay; a second arithmetic implementation checks144 policy/view
choices,1,152 scene reductions and1,581,804 scored rows. This is same-agent
verification, not independent research replication. See `execution_notes.md`.
