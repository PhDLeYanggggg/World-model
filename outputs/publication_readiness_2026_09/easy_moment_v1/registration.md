# Conditional Easy-Moment Development Experiment

Registered 2026-09-24 before the new moment fits and their prediction readout.
This is delegated development work, not an independent test or changed scientific
data-role assignment. All four source sites have already influenced design.

## Hypothesis and single changed factor

Fixed generic harm protection discards much forecast benefit. Test whether
estimating easy-weighted harm directly improves the empirical gain/easy tradeoff
relative to multiplying separately estimated easy probability and harm. This
tests the conditional-moment mechanism in the existing algebra note, not a new
theorem. The original neural benefit/harm scores and predictors are frozen.

Let b be complete-path CV ADE, h=max(candidate ADE-b,0), d the past-only mean
forecast disagreement, C the existing training-only positive-easy cutoff, and
e=1{b<=C}, including zero-b cases. One four-output forest fits, on supported
source rows only:

```
e*h/d, e*b/C, e, h/d
```

For d=0, harm fractions are zero and the policy never switches. These zero-loss
rows carry no fitting mass, matching the historical bounded-cost sampler. All
four target fractions are in [0,1]. C is specifically the existing shared
reporting cutoff from the excluded-site parent, not a new complete-case
quantile recomputed by the bounded learner's preprocessing. Restore predicted q=d*E[e*h/d|X],
r=C*E[e*b/C|X], p=E[e|X], and h_hat=d*E[h/d|X]. Compare:

```
joint_easy_moment:       existing net-positive support AND r>0 AND q <= .02*r
product_easy_marginals:  existing net-positive support AND r>0 AND p*h_hat <= .02*r
```

The same fitted model supplies both policies; no model-capacity or sampler
difference explains their contrast. Predicting the joint moment is generally
not equivalent to multiplying marginals. If true conditional moments were known,
the pointwise inequality is sufficient for a corresponding population easy-risk
ratio. Estimated moments, missing labels, finite support and a site shift do not
inherit that statement as a certificate. The .02 multiplier expresses the
existing easy tolerance in this surrogate, not a new calibrated risk threshold.
No confidence, probability calibration or formal safety is claimed.

## Frozen matrix and controls

- Actions: damping005, full Transformer, full EqMotion. No forecast retraining.
- Four excluded source sites x three seeds x three actions = 36 forests.
- Each forest: 128 trees, depth16, leaf64, 1/3 features, CPU4, checkpoints every16.
- Exactly the corresponding cached neural head's 768,000 training draws; no
  outer-site rows, no unknown future labels sampled. Neural scoring producers
  also exclude their supervised row's site. Complete source path labels only.
- Common existing input features, training-only standardization and easy/hard
  cutoffs. Raw endpoint/future masks are never features or decision eligibility.
- Existing net-positive and strict-harm policies are retained as fixed controls.
- No new hyperparameter trial, threshold search, winner selection, external
  prediction, calibration, confirmation or deployment. HT21 is not admitted;
  DUT is not reopened; DroneCrowd remains closed.

## Evaluation and rejection criteria

All 36 fits and all past-only decision archives must complete before a new outer
label readout. Retain all 175,756 past-eligible windows, including partial and
missing futures. Primary descriptive metric remains equal mean of physical-site
relative available-point ADE gain over CV. Three-seed mean errors are not an
ensemble forecast. Use paired 3,000-site bootstrap; four exposed sites cannot
provide independent confirmation or broad distribution-shift evidence.

Report each fixed policy, every site/seed, complete/hard/positive-easy/zero-CV
subsets, endpoint FDE, tails, intervention counts, missing-label counts and full
grid gain bounds. Primary mechanism contrast: joint minus product, separately
for each action. Secondary: joint minus the historical strict policy. Also
report matched intervention counts ranked by the existing predicted net gain;
this is outcome-blind offline allocation, not a deployable query-local rule.

Observed easy degradation must be <=2% at every site/seed; zero-CV damage is
separately disclosed. Joint-moment prediction must not be presented as a success
merely because it passes more interventions or reduces fitting loss. If utility
does not improve, easy fails, or the contrast is uncertain, retain the negative
result with no threshold repair in this version. No new best deployment follows
from this development study even if the empirical contrast is positive.

## Runtime and provenance

Local arm64 `.venv-pytorch`, CPU4/interop1, workers0. A short first-tree pilot
resumes into the same 128-tree budget. Atomic checkpoints, fit traces, heartbeat,
locking, input/label/sampler hashes and exact inference replay are required.
Slow fitting is not a reason to reduce the registered matrix. Remote queue
status remains unknown after the latest recorded CREATE authentication failure;
this bounded local forest experiment requires no new HPC login.

Annotation pixels, obs8/pred12 native steps. Not verified online observations,
metric/seconds, true3D, foundation or physical safety. Stage5C and SMC stay off.
