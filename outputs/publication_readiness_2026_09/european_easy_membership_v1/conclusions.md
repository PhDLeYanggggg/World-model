# Direct Easy-Membership Transport: Conclusions

## Material Passport

Fresh_run: 288 Torch classifiers, 576,000 updates, frozen predictions and
36 group readouts. Cached_verified: source-development data, causal producer
forecasts and parent lineage. Not_run: a new expected-harm model, a deployment
policy, independent calibration or confirmation. Registration 4bdbc117 and
prediction freeze 269d7ec0 preceded training and outcome readout, respectively.
Sensitivity c493cf1f was declared after training began but before readout.

## What Worked

The registered full-input MLP membership diagnostic passes. Across all six
source assignments, the 95% locality-bootstrap intervals are positive for
Brier skill against the training-prevalence constant, AUROC minus 0.5, and
log-loss gain. These are probability-prediction results, not ADE/FDE gains.

| Producer -> controller | MLP Brier skill % [95% CI] | Stronger conditional-constant skill % [95% CI] |
|---|---:|---:|
| 0 -> 1 | 27.85 [25.31, 29.93] | 25.47 [22.23, 28.46] |
| 0 -> 2 | 37.13 [28.82, 46.49] | 34.06 [26.40, 41.71] |
| 1 -> 0 | 35.70 [32.64, 38.77] | 31.86 [22.39, 41.33] |
| 1 -> 2 | 40.27 [33.34, 47.21] | 35.63 [31.11, 39.11] |
| 2 -> 0 | 46.31 [42.67, 49.14] | 40.29 [30.13, 50.45] |
| 2 -> 1 | 52.60 [47.76, 58.44] | 50.53 [45.91, 55.08] |

On positive-disagreement rows, median full-input MLP AUROC is 0.86203,
AUPRC 0.59012, Brier 0.10332 and ECE 0.04539 across 72 dependent views.
The training-prevalence constant has median Brier 0.16678. The stronger
training-conditional-prevalence sensitivity retains six positive Brier and
six positive log-loss intervals. Thus the full result is not explained only
by using the wrong unconditional prevalence control.

Both full-input arms improve fitting and held Brier over the registered
constant in all 72 views. The MLP also beats the original reference-cost ratio
in AUROC in five of six paired intervals; the sixth includes zero. That ratio
was never a calibrated easy-event probability.

## What Did Not Pass

- The linear arm misses the registered log-loss requirement: five positive
  intervals, one overlapping zero. Its Brier and AUROC contrasts are positive
  in all six assignments. A sensitivity does not retrospectively rescue it.
- Motion-only MLP looks favorable against the unconditional constant, but
  against the stronger conditional constant it has one positive, one negative
  and four overlapping Brier intervals. All six log-loss intervals include
  zero. Eleven of 72 motion-only conditional views have fewer than 20 positive
  examples; none is dropped. Full conditional has no such weak support flags.
  This negative control remains visible in the sensitivity figure.
- ECE is not uniformly small: the full MLP range is 0.00726 to 0.22652.
  Passing the diagnostic is not a finite-sample calibration or safety claim.
- No conditional harm magnitude or selected-policy risk was evaluated here.
  No trajectory prediction or deployment improvement can be inferred.

## Interpretation and Next Test

The prior harm-readout failure cannot be attributed simply to a total absence
of easy-membership signal in the full causal inputs. Membership is learnable
under this source-locality diagnostic; converting it into accurate expected
harm is still unresolved. This supports testing a factorized expected-cost
model, not deploying the classifier as a new intervention rule.

The next registered source-only experiment should compare a direct expected-
easy-harm head with P(E|x) times E[H|E,x], including zero-harm easy examples.
The latter conditional factor must learn severity, not merely event ranking.
Use equally trained controls, fitting-only preprocessing and nested/out-of-fold
membership predictions where they become another head's input. Keep reference
costs fixed; require improvement in expected-harm error and tail behavior
before evaluating a policy. This follow-on is not_run in the current round.

## Limits

Three seeds are averaged within each locality before 3,000 resamples of four
localities per assignment. Roles and overlapping windows are dependent;
these are exploratory source-development intervals, not independent-domain
confirmation or multiplicity-adjusted evidence. Full and motion-only models
have different positive-disagreement populations; their conditional scores
are not a matched-population scene/interaction ablation.

The inherited easy event excludes exact-zero baseline error. Detector-derived
image pixels and eight observed/twelve predicted annotation steps do not
establish human gold, metric units, seconds, physical safety, true 3D or a
foundation model. Reserved independent data remain closed, deployment is
unchanged, Stage5C and SMC are off, and the project is not submission-ready.
