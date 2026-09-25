# Stopping Defect Repaired; Support Filtering Does Not Improve Overall Selection

**Completed exploratory experiment. Deployment unchanged. Not submission-ready.**

Registration `3ff17937` preceded all new decisions. Both decision banks froze before
new readout. All 720 views ran: 72 original-controller views and 648 guard/control
views, across two target references, two normalization modes, three source folds,
three seeds and two event objectives. Forecasts, fitted heads and population are
cached_verified; support fitting, decisions and evaluation are fresh_run. No new
neural model was trained in this experiment.

## What Worked

Requiring movement in the latest observed step, instead of anywhere in the observed
history, eliminates the zero-CV harm in all 24 supported views per target family.
The other twelve views have no zero-CV examples. These are four unique underlying
rows in one locality with only two future labels and no endpoint, not 24 independent
cases. This repairs a concrete decision defect on the observed evidence, not
full-horizon stationarity prediction or physical safety.

The stop-guarded CV-target controller retains 0.1249% to 1.7634% all-ADE gain over
the protected floor; the floor-target version retains 0.1225% to 1.6410%. Each has
36 positive conditional all-ADE intervals, also 36 on complete-label ADE. The gains
already exist in the unchanged controllers. Stop-versus-original changes are only
-0.00000384% to +0.00011856%, with **no positive or negative paired intervals**.

All policies satisfy the descriptive 2% positive-CV easy-error limit on this opened
population; the worst locality degradation is 0.4369%. Zero-CV rows are accounted
for separately and are not hidden inside that positive-easy definition.

## What Failed

The rectangular fitting-support rule reduces all-ADE performance relative to its
unchanged parent in every view:

| Parent | All-ADE gain over unchanged parent | Negative / positive conditional intervals |
|---|---:|---:|
| CV targets | -0.1278% to -0.0181% | 36 / 0 |
| Both-floor targets | -0.1224% to -0.0149% | 33 / 0 |

The common-denominator accounting shows more removed benefit than avoided harm
in all 36 views per family. Rejection is not automatically useful. Support and
combined guards produce identical decisions: no fitting fold has the two required
sources with sufficient terminally stopped support, so support already rejects
that state. These are duplicate policies, not independent successful mechanisms.

The stop policy has **zero flexible same-frame quotas in every view**. Its risk and
random count controls therefore produce identical decisions. This is a coverage
safeguard, not evidence of better multi-agent allocation.

Support has nontrivial quotas, but its overall ranking advantage is inconsistent.
Versus same-count random selection, CV-target support has five positive and five
negative all-ADE intervals; floor-target support has four and four. Positive-easy
accuracy does improve in some matched comparisons: 27/36 and 23/36 positive easy
intervals versus random, respectively. This tradeoff is retained, not hidden; it
does not overcome all-ADE loss or establish robust hard-case ordering. Hard
performance versus the unchanged parent worsens in 35/36 points in each family.
The floor-target controller still has a hard locality with 2.8055% degradation
relative to the floor after stopping protection.

## Verification and Scope

720 saved policies replay with separate scalar decisions; 144 coordinate arrays,
8,208 metric reductions and 432 original metrics verify. Same-recording/current-frame
counts match. Source fitting excludes all eight held development localities per
fold. 314 scoped tests in 50 files pass; the full unrelated legacy suite is not_run.
The two evaluation phases took 525 and 531 seconds locally; no crash or downscaling.

The twelve EuropeanSquares localities and overlapping windows are opened development
data. Three seeds and 3,000 locality-bootstrap draws per metric do not turn these
repeated views into independent confirmation. Image pixels, obs8/pred12 rawstride12,
released detector tracks: no verified metric/seconds, human gold, true 3D, foundation
or physical-safety claim. No Stage37/SDD raw-t50 recertification. Independent
selection/calibration/confirmation remain closed. Stage5C and SMC remain off.

## Next Decision

Retain the last-step stop check as a frozen engineering candidate, not a new model
claim. Do not search wider percentile boxes on these outcomes. First separate
history-support exclusion from model-generated disagreement and the two-source to
four-source producer shift, with removed-benefit accounting and the same-frame
controls fixed. The eventual safety claim also needs genuinely independent,
adequately labelled stop/start scenes; the four partial-label cases cannot supply it.

[Full results](results.md), [failure analysis](failure_analysis.md), [gates](gates.md),
[decision context](decision_context.json), [reproduction](execution_notes.md),
[Chinese guide](operation_zh.md), [literature position](literature_position.md).
