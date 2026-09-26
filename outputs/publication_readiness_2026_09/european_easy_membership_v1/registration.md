# Direct Easy-Membership Transport Diagnostic

## Material Passport

Registered source-development experiment. Previous turn was progress:
288 frozen readouts completed and verified; the expected-harm gate failed.
Cached_verified parents, causal input schema, source forecasts and labels are
retained. New membership classifiers and their evaluation will be fresh_run.
No future error, future endpoint or realized easy label is an inference input.
No independent confirmation, new trajectory model or new policy is run here.

## Hypothesis and Distinction

Outside-easy rows dominate excess MSE in 36/42 full views where the last
fractional-feature readout worsened against original_mean. Test whether the
existing causal inputs support prediction of easy membership in held
localities. This is a necessary diagnostic for a potential later conditional
cost factorization; it is not sufficient to justify that factorization.

Define E = 1(0 < CV_error <= fitting_easy_cut). Keep the existing positive-CV-
error 25th-percentile definition; CV_error=0 is excluded by that inherited
definition and must not be silently reclassified. Easy cases with no harm
are positive E labels. This differs from the prior hurdle label H_easy > 0.
The target remains supervision/evaluation only. Unknown labels stay unknown.

## Models and Fitting

Two arms use identical causal features: logistic linear and a width-64
GELU MLP with one logit. Preprocessing is inherited/recomputed on the three
fitting localities, with equal-locality weights over known labels. Inputs
contain past context and causal producer rollouts, not ground-truth futures.
No hidden-feature cache from the held locality is fitted. Original mean
D_easy/D_all is a cached causal ranking proxy, not an event probability.

Use unweighted-in-class binary cross-entropy and the existing equal-locality
sampler; no class oversampling, temperature, threshold or coefficient search.
Both final output layers start at zero weights and a training-prevalence
logit bias. Their initial probabilities/losses match. AdamW lr0.0003,
weight_decay0.0001, batch256, gradient clip5, 2,000 updates, checkpoints and
heartbeats every200. Sampler draws/RNG match each other and the original
mean control. Architecture capacity differs and is reported, not hidden.

Proper probabilistic scoring motivates separating probability error from
ranking. [Gneiting and Raftery (2007), Section3 Examples1 and3](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)
describe quadratic/Brier and logarithmic scores. Reading scope: these
examples and the surrounding definition, not the whole paper. The citation
does not provide a finite-sample calibration, domain-transfer or safety
guarantee for this experiment, nor make membership prediction novel.

## Source Roles and Budget

Six ordered source assignments x seeds17/29/43 x full/motion-only forecast
pairs x four leave-one-B-locality-out folds x two arms =288 new fits,
576,000 updates. Source-A producers exclude B; three B localities determine
normalization, prevalence and easy cut. The fourth is excluded throughout
all fitting. Existing context loaders may load cached held labels, but those
labels do not enter fitting or selection. Previously opened source development
is not relabeled independent test. Six selection localities are unused;
12 reserved calibration and6 confirmation localities remain closed.

First run a real100-update linear pilot, included in its fixed budget, then
resume the whole run. The pilot estimates only linear fitting cost; the first
MLP provides actual MLP timing. CPU4/interop1/workers0, native arm64, one
process, no NumPy fallback. Preserve checkpoints on interruption and keep
10GiB disk reserve. No time-based downscaling or favorable stopping.

## Readout and Diagnostic Criterion

Commit/push all prediction hashes before current held-label readout. Compare
linear, MLP, a training-only prevalence constant, and original reference
ratio (proxy only). Report all and causal-envelope-positive subsets, Brier,
log loss, AUROC, AUPRC, easy prevalence, ten fixed-width probability bins,
ECE, and fitting/held differences. Log loss clips only evaluation probabilities
to [1e-7,1-1e-7] for numerical stability. No post-hoc probability calibration.

Primary diagnostic: MLP Brier skill against training prevalence on full,
envelope-positive rows. Require six positive assignment-level intervals,
plus six positive intervals for AUROC-0.5 and log-loss improvement against
the same training constant, to say this probe established transportable
membership signal. Report linear separately; do not select it to rescue a
failed MLP criterion. Retain all adverse role/seed/motion-only results.

Within each assignment average three seeds within locality, then bootstrap
four localities 3,000 times (seed47131). No favorable locality removal;
missing support is not_estimable and cannot pass. Repeated windows/source
roles are not independent. Intervals are exploratory and not adjusted for
multiple comparisons. The main forecasting endpoint and2% risk tolerance
are unchanged. Even this diagnostic passing permits no deployment, policy
advance or opening of independent data: conditional harm must still be tested.

## Research Boundaries

Eight observed / twelve predicted annotation steps, raw stride12,
detector-derived image pixels. No human-gold, metric/seconds, physical safety,
true3D, foundation or submission-ready claim. Stage5C and SMC remain off.
Keep raw data, per-row probabilities, features and weights private. Preserve
unrelated staged files. Local resources fit the proposed run; the recently
verified CREATE queue observation does not establish a remote M3W path/job,
and other projects' jobs are left untouched.
