# Temporal Error Structure: A Positive Probe, Not a Policy Repair

Completed 5 October 2026. The registered diagnostic screen passes. The research
goal, selected-policy safety requirement and submission-readiness gates do not.
The deployed policy is unchanged.

## What Was Actually Run

- `fresh_run`: 72 TRAIN-only analytic temporal probes, exact refits, inference
  replays, recording-held development readouts and 3,000 locality bootstrap draws.
- `cached_verified`: original forests, source forecasts, source labels, causal
  input lineage, whole-recording partitions and original action hashes.
- `not_run`: neural auxiliary training, changed routing, changed thresholds,
  independent calibration/confirmation, new cross-domain evaluation or deployment.

The 72 views cover 12 already exposed European development localities. They are
not 72 independent scenes or three new end-to-end forecaster trainings. Every
head retains its whole-recording TRAIN/validation separation. Frozen causal leaf
assignments receive three matched TRAIN-mean probes: a per-step leaf mean, a
whole-trajectory leaf mean repeated over time, and a global per-step mean.
Targets are neural-minus-reference error and reference error at each of 12 steps.

## What Worked

The temporal-leaf probe improves signed-error prediction against the leaf-local
whole-trajectory mean by **-0.094342 normalized MSE**, nominal 95% locality interval
**[-0.158824, -0.038614]**. It improves 69/72 head views and 11/12 locality averages.
Against the global temporal mean, the signed-error change is **-0.137388**
**[-0.232556, -0.047422]**. This distinguishes temporal variation from a purely
global error-by-horizon curve and from causal conditioning without temporal detail.

The same signed contrasts on complete-label validation are -0.098473
[-0.159674, -0.047870] and -0.171893 [-0.270002, -0.085365]. The complete-label
analysis is an offline sensitivity check, not an inference-time completeness
filter or a cure for missing labels. Signed error and the mean of the two error
channels satisfy all eight preregistered diagnostic conditions.

All 5,986,968 observed validation step occurrences have score support. Only 126
step occurrences use any tree's global TRAIN fallback, across 10 head views.
These counts repeat source rows across heads. They do not establish independent
sample size, label correctness or policy-support guarantees.

## What Did Not Work or Remains Unproved

On the **frozen selected cohort**, temporal prediction does not reliably improve
the matched leaf-local mean: signed MSE changes **+0.000043**
**[-0.000140, +0.000256]**. The two-channel change is also inconclusive:
+0.006656 [-0.004446, +0.022262]. This stratum has 11 supported localities; an
empty selected locality is not assigned a zero error. Beating the global mean
on this stratum does not compensate for failing the stronger leaf-local control.

Locality 110 worsens against both controls in the all-row signed readout.
Locality 082 improves against the leaf-local mean but not the global temporal
mean. Thus the average diagnostic result is not universal transfer.

The substantial reference-error-channel improvement cannot alone establish
better neural gain/harm decisions. The separate signed-error gate passes on
all/complete rows, but the selected-cohort result is the important limitation.
Original predictions and selections are unchanged, so **no trajectory accuracy,
utility or selected easy-risk improvement was demonstrated by this experiment**.
The original/additive/Poisson/squared-cost controls remain required policy
comparators; these diagnostic means do not replace them with weaker baselines.

## Why the Target Must Not Be Replaced

For observed steps let d(t) be neural error minus reference error. Write
H = max(mean(d), 0), G_H = mean(max(d, 0)), G_B = mean(max(-d, 0)), and
C = min(G_H, G_B). Then H = G_H - C exactly. Predicting per-step errors may
provide auxiliary information, but substituting G_H for H changes the task.
Nor does taking the positive part of a predicted mean identify expected H;
nonlinearity and conditional uncertainty remain relevant.

Across the frozen selected head-view occurrences, gross step harm is 2967.188394,
cancellation is 1182.067048 and original harm is 1785.121347. Cancellation is
**39.84% of gross step harm**. These pooled image-local masses are descriptive,
not risk ratios or model gains, and cannot be compared as common metric units
across cameras. There are 52,873 selected occurrences with some positive step
harm but no positive whole-trajectory harm. There are also 918 selected
occurrences with entirely unknown future labels; they stay unknown.

The initial reporting mask could omit ER=0, EH>0 easy-harm cases. A failing
regression test identified it before the full run. The original registration
and pilot are preserved; amendment `d046c265` fixed reporting and required exact
agreement with original selected EH totals. The training target, action rule and
2% selected positive easy-harm budget did not change.

## Next Experiment and Stop Rules

This result supports a **separately registered matched auxiliary-training
comparison**, not immediate rollout. Retain the primary B/H/R/ER/EH objective and
compare the same causal cost-head architecture with no auxiliary, whole-trajectory
mean auxiliary, and temporal auxiliary. Match producer-chain exclusions, training
rows, draws, seeds, initialization and fitting budget. Any auxiliary weight is
fixed before development readout or chosen entirely inside TRAIN, not on test.

Primary evaluation must include original signed-cost prediction, selected harm,
full and matched-count utility, unknown-outcome bounds and all strong policy
controls. A global temporal MSE reduction alone is insufficient. Do not switch to
gross step harm, change the risk denominator, relax 2%, or tune cutoffs to pass.
If the temporal arm only helps unselected/reference-error rows, retain that
negative decision result and do not promote the model.

Storage feasibility is checked before new checkpoint allocation: the full-run
receipt records 9.33 GB free, below the inherited 10 GiB numerical-cache reserve.
Two authorized read-only CREATE connection attempts timed out before any directory
or scheduler observation. No remote job was submitted, restarted or cancelled;
remote state is unverified. The completed cache-free local experiment did not
depend on remote access. This is not a reason to declare the research complete.

## Reproduction and Claims

Full fitting/replay: **279.13 seconds**, **9.719 GB peak RSS**, native arm64,
4 compute threads, 0 DataLoader workers. Every probe was exactly refit and its
inference replayed. Twelve scoped tests, 1,656 run-time checks and 3,696 independent
scalar/bootstrap checks pass. The full legacy test suite was not run this turn.
These engineering checks do not establish scientific generalization.

Intervals describe already exposed development data and are nominal, not
adaptive-search-adjusted. Partial-label fitting uses per-step TRAIN means,
whereas evaluation averages observed steps within each row; those weightings
can differ with label availability. The complete-label sensitivity analysis
does not remove this limitation or make missingness ignorable.

Keep image-local detector-silver, obs8/pred12, rawstride12 language. No seconds,
metric, human-gold, physical-safety, true-3D, foundation or submission-ready claim.
Stage5C and SMC remain off; the long-term goal remains active.

Evidence: [tables](results.md), [summary](summary.json),
[completion](complete.json), [independent verification](verification.json),
[statistical interpretation](statistical_interpretation.md),
[Chinese reproduction guide](reproduction_zh.md).
