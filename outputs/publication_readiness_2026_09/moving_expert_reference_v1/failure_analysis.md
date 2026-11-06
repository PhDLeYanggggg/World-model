# Shared Expert Context Does Not Repair The Complete Risk Contract

## Completed Experiment

Both new arms completed 24 source-held reference fits at seeds17/29/43,
4,096 updates each, followed by all48 complete exact refits. Their shapes,
initial reference states, preprocessing, sampling and budgets match. The sole
feature contrast admits or masks70 shared frozen-expert descriptors. H/EH and
label-support outputs are bitwise unchanged per candidate. All18 new choice
caches froze before outcome readout. Original2%/support/NA rules did not change.

This is real Torch reference training, not a newly trained trajectory dynamics
model. The underlying forecaster remains frozen, with one teacher seed. Data
are already-exposed MovingDroneCrowd++ own-B, not independent confirmation.

## The Primary Does Not Establish Lift

Expert-context minus history-refit joint-free paired completion-lower gain is
+0.133479pp, exploratory source interval[-0.140981,+0.556238]. The known-label
contrast is+0.175721pp[-0.090314,+0.594128]. Both intervals cross zero.
Relative to the frozen prior model, the context lower contrast is+0.072390pp
[-0.244467,+0.535662], also inconclusive. Do not substitute the larger known
gain point, a secondary comparison or separate lower-bound subtraction.

Only3/8source lower contrasts are positive. Source059 contributes+1.487083pp;
sources053,060,067,073 and076 are negative. Seed17 lower contrast is-0.021279pp,
versus+0.087349/+0.334368 for29/43. These are three risk seeds, not three
forecaster seeds. Bootstrap uses3,000resamples of8sources after averaging seeds,
not overlapping windows or72 independent scenes. It does not account for the
entire adaptive project search and is exploratory exposed-development evidence.

## The Safety Failure Remains

| Model, joint-free | Known ADE gain vs CV % | Full original supported views | Known easy harm violations | Undefined easy views |
|---|---:|---:|---:|---:|
| Frozen prior | 0.667496 | 23/72 | 27 | 7 |
| History-only refit | 0.614917 | 25/72 | 23 | 6 |
| Shared expert context | 0.790637 | 24/72 | 26 | 6 |

Context also has3known-all violations,8signed easy-net violations,
9completion-all and42completion-easy violations. Its mutually exclusive joint
categories are26observed violations,18completion-only violations,4support-only
failures and24passes. Some failures overlap, so support-only counts and six
undefined easy views need not match. No required recording/seed view is removed.

Worst defined known-easy gross risk is40.2986%; worst signed easy-net degradation
is9.0168%, still above2%. The frozen maxima are69.3306% and22.1189%. Those tail
reductions are descriptive; they do not remove undefined views or the remaining
failures. Gross harm and signed net degradation are different quantities. Full
uncertainty and tails remain in policy_uncertainty_and_tails.json.

Context independent choice passes28/72, while its joint policy passes24/72.
Joint optimization still improves mean utility while exposing risk prediction
errors. Every model fails the full required roster. No policy is promoted.

## What The Model Change Actually Did

### Total Reference Improves, Easy Reference Worsens

On the identical known candidate cohort with frozen source-specific fitting RMS:

| Reference model | R MSE | ER MSE | R bias | ER bias |
|---|---:|---:|---:|---:|
| Frozen | 3.418436 | 0.214651 | -0.140591 | -0.004474 |
| History refit | 3.360713 | 0.230431 | -0.167548 | -0.004916 |
| Expert context | 3.154554 | 0.280262 | -0.151116 | +0.006367 |

Context reduces R MSE about6.1% versus its matched refit, but increases ER MSE
about21.6%. All24 final matched training minibatches improve, yet the held-source
easy-reference task worsens. Fitting loss, global R quality and selected ER risk
are different targets. This is not a blanket conclusion that context is useless.

### Selected Easy Credit Is Still Optimistic

The context joint policy overpredicts selected ER in62/72views, versus60/72 for
history-refit and61/72 frozen. Of its predicted selected ER mass,87.89% falls on
known rows that do not supply ER under the original realized easy definition;
the frozen value is85.78%. On identical frozen choices the fractions are86.26%
and85.78%, respectively. Unknown-label contributions are counted separately.

These are pooled repeated-view attribution diagnostics, not a misclassification
rate, calibration theorem, independent estimate or a requirement that soft
probabilities be zero on every realized negative. A conditional expectation can
assign positive ER to a realized non-easy row. The diagnostic measures how much
the final selected budget relies on unrealized easy reference, and the observed
gross-risk failures show why average prediction alone is insufficient.

The realized non-easy category also includes zero-reference cases excluded by
the original easy definition; it is not synonymous with the HardBench subset.
Future easy labels appear only in supervision/diagnosis, never policy inputs.

### Changing Reference Also Changes The Benefit Cap

The new models additionally cap5,941history-refit and4,503context candidate/seed
views relative to frozen conditional benefit. These are repeated candidate
views, not distinct agents. The primary shares the same cap and composition
code, but a comparison with frozen outputs is not a pure denominator-only
intervention. Floating-point recomposition is not claimed bitwise unchanged for
benefit; the exact invariants are H/EH and support. All actual replay arrays
must nevertheless match exactly.

### Frozen Harm Is Not Already Solved

A posthoc illustration uses the worst signed-easy recording, then its largest
absolute selected easy harm. Source053/recording11/seed17 has a selected alpha0.5
case whose available-label ADE rises6.175879 to10.959726pixels. Actual positive
harm is4.783846 versus predicted H0.765837; predicted R38.814562 versus observed
R6.175879. Predicted EH is0.285495, also much below observed easy harm. The case
has8future labels out of12requested, not a complete12future or a representative
sample. The private trajectory figure shows the baseline closer to those labels.
The scalar summary is in failure_case_summary.json. No case was used for tuning.

Thus a reference-only input intervention cannot be treated as a complete harm
repair. Reference and harm errors coexist. The action can be admitted through
query pooling despite an unfavorable own easy-budget excess. This motivates
learning their shared easy-event support, not declaring the frozen harm valid.

## Diagnosis And Next Decisive Repair

First restore simple fixed-amplitude controls on this exact exposed own-B
cohort. The earlier six-fraction uniform-policy audit failed on its different
195,286-row/348-view cohort; it is not a matched score for the current62,466rows.
Report all six original fractions and a causal stationary-guard control without
choosing a favorable coefficient from readout. This is a strong-control gap,
not a reason to relabel a tiny blend or fallback as learned success.

The evidence narrows the next target to easy-conditional costs and selected use,
rather than lack of optimization or total-R input alone. The shared R/gate
decoder, ER-only regression and frozen harm errors are plausible contributors;
this experiment does not isolate their gradients or prove a unique explanation.

The next controlled model should explicitly separate easy-support prediction
from conditional reference **and harm** magnitude, instead of another global
threshold or network-width increase. Training labels may identify the original
easy event; inference must use predicted probabilities and past-only context.
Use the same learned event support for ER and EH, with coherent, conservatively
linked global moments. Keep the forecaster, causal actions and original contract
fixed; register exactly which cost outputs change before fitting. Compare the
old product supervision against the new source-excluded factorization. Measure
easy-event reliability, conditional R/H errors and selected ER/EH, not merely
global MSE. No independent held test is a development tuning source.

This is not proof that factorization will succeed or a new novelty claim for a
hurdle model. Source-held calibration, finite-record variability and independent
confirmation remain necessary. No new C/H is opened to rescue this failed model.

## Evidence Boundary

fresh_run:context derivation,48fits,48full refits,new choices/readout,bootstrap
and attribution. cached_verified:parent features,labels,forecasts and frozen
controls. not_run:new independent C/H,three forecaster seeds,new dynamics or
deployment. Native obs8/pred12 available-label image-coordinate ADE is not
official complete12ADE, SDD raw t50, metric/seconds, true3D, foundation or human
gold. Original24-source-group main stays NA. Annotation-generation causality
is unverified. Stage5C and SMC remain off; the paper is not submission-ready.
