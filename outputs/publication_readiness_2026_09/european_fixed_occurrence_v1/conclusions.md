# Decision: Do Not Promote Fixed Occurrence

## Material Passport

Fresh native-arm64 Torch paired training and development evaluation, with
cached_verified inputs, warm starts, forecasters, floor and utility. All108
action groups and the complete numerical readout replay exactly. Independent
arithmetic verification is documented separately in verification_report.md.
Independent confirmation was not run. No deployment changes.

## What Was Tested

Both arms start from the same warm head, duplicate it into identical occurrence
and cost branches, reset AdamW and use matched query sequences for2,000 updates.
Only occurrence trainability differs. This is a controlled freezing comparison
within the split architecture, not proof that splitting versus the old shared
trunk is beneficial. The forecaster itself was never changed.

## The Registered Result

| Comparison at matched query counts | ADE gain % | Nominal95% locality interval |
|---|---:|---:|
| Fixed versus trainable occurrence | -0.0163878 | [-0.0290309, -0.0059681] |
| Fixed versus raw-risk control | -0.0550033 | [-0.1073037, -0.0044347] |
| Trainable versus raw-risk control | -0.0385291 | [-0.0818672, +0.0097693] |

The specified fixed-occurrence remedy is not supported. Its primary contrast
is negative at9/12 localities; the three positive localities have small gains.
The raw control remains stronger in this matched-count comparison. These are
small effects on previously opened development sources, not independent proof
of a universally inferior architecture. Comparisons across earlier experiments
do not share this experiment's common matching anchor.

Selected-risk violations fall from94 to89/216 dependent views when freezing,
but both new arms are worse than the raw matched control's75. All three matched
policies retain22 undefined-risk views. Every-view easy preservation passes its
developmental check, but the accuracy and selected-risk checks fail. Fewer
violating views is not a risk certificate, and undefined risk is not zero.

## Why Lower Prediction Loss Did Not Suffice

On fitting data, fixed occurrence has worse marginal loss:0.002507829 versus
0.002378599; only2/108 paired groups improve. On held development data, fixed
occurrence has slightly lower Brier (0.158131 versus0.159688) and signed-risk MSE
(0.00376135 versus0.00382361). Those are descriptive point comparisons, not a
paired confidence claim about calibration. They coexist with worse downstream ADE.

The loss and the policy optimize different quantities. Accurate average scores
need not preserve useful interventions near the joint allocation boundary. The
benefit/harm decomposition in failure_analysis.md measures that tradeoff using
a common full-floor denominator. It does not replace selected risk, identify
which covariate caused the errors, or prove a domain-shift mechanism.

For fixed conditional costs and p>0, q=p*(h-0.02r) has a sign independent of p.
Occurrence can affect joint relative weighting and cost training, not act as a
separate individual safety veto. This prevents interpreting a better Brier
score as evidence that unsafe individual switches must disappear.

## Engineering Recovery

After95 groups the original solver-result handler encountered a null dual
certificate. A registered compatibility adapter treated it as missing, invoking
the existing checked-anchor fallback without editing sealed algorithms. All95
old groups remained byte-identical; full replay also reproduces the single
missing-certificate event and all108 actions. The solver reported infeasibility;
the underlying numerical cause was not established or claimed fixed.

## Next Research Action

Do not launch another occurrence-freezing or gradient-cap sweep on these
outcomes. Use the now-complete fitting-only signed-benefit/harm labels to locate
utility-ranking and false-safe risk errors within supported queries. Separate
lost beneficial switches from harmful admissions, then preregister one targeted
remedy if the fitting evidence supports it. Keep the current forecaster/floor,
source roles, risk estimand and deployment unchanged. No independent source
should be opened merely to search for a favorable result.

The raw anchor already forces at least16 unsupported matched views. Their risk
cannot become defined by this branch intervention. Independent source-level risk
calibration, support and the original selected-risk primary remain unresolved;
full-floor accounting cannot resolve those gaps by renaming the denominator.

## Claim Boundaries

Only12 already-opened European development localities. Repeated groups and
three forecast seeds are not independent samples. Intervals use3,000 paired
locality draws and are nominal, not multiplicity- or selection-corrected.
Observation8/prediction12, raw-frame stride12, image-local detector silver.
No metric, seconds-level, human gold, physical-safety, true3D, foundation or
submission-readiness claim. Historical exposed scores remain exploratory.
Stage5C and SMC remain disabled. This is risk-controller research, not a new
world-dynamics result.
