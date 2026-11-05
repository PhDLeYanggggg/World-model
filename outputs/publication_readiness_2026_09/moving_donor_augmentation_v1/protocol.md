# Fixed Donor-Augmentation Information Experiment

Six within-producer EqMotion crossfits produced125,540source-held rows in the
previous experiment. The present hypothesis is that their additional source
coverage improves original120 signed-risk learning on already-exposed own-B.
No outer C/H, reserved source, main split or evaluation rule is changed.

## Matched Experiment

Eight existing query-source folds, three fixed arms each: own-only, A-only and
50/50-weight own+A. Use the original120 causal motion/rollout/neighbor features,
64/32 SiLU MLP, AdamW lr.001, wd.0001, gradient clip5, batch512, seed17.
Each model receives4,096updates. This is a matched update budget, not32epochs;
sample with replacement using source/query-balanced fitting weights. Fit all24
models and repeat all24 before the single final development readout.

Pool weights are fixed before seeing query outcomes. Query-source own-B rows
are excluded from fitting, preprocessing, target normalization and sampling.
Parent-A donors come only from that parent's crossfits. Their teachers never
fit the donor source or the parent context's outer C/H. No warm critic or router
is used to produce A candidates, avoiding query-label exposure through routing.

For A donors, use all six predetermined interpolation fractions from1/32 to1
between the parent's training-defined floor and OOF prediction. Every fraction
is retained when causally nonidentical to the floor. Compute causal features
before filtering supervision to known outcomes; count unknown candidates but
never call them safe. Labels, easy cuts and float32 risk subtraction match the
original task. Original own-B proposals remain fixed, not regenerated.

This donor treatment changes source coverage, teacher training size and candidate
mixture. It is not a pure causal estimate of source count. A-only is a diagnostic
control. No donor source or amplitude is chosen by its observed predictive gain.
The original same-proposal own-B readout is used for every comparison.

## Readout And Decision

Compare fresh own-only, A-only, augmented plain risk heads; cached shared64,
short neural-plain and continued neural-plain are verified controls. Primary
information screen: augmented source-macro decision cost at least1% lower than
matched own-only, false-safe cost no higher, and augmented normalized MSE lower.
Require2of3contexts for support. Also report the stricter forest comparison,
all source folds, errors, admissions and harmful admissions. Use each original
own-B control's fitting-only scale for evaluation of all arms, not arm-specific
MSE denominators. The screen is not the original full-policy safety estimator.

No threshold search, final-epoch choice or new test. Main24-group estimate is
still NA. Original2% all/easy harm, signed easy-net and support remain; empty
support/control is NA, not pass. No identity-fallback certification. Old
failures remain. No deployment or submission upgrade follows this screen alone.

## Runtime And Records

Native arm64 .venv-pytorch, CPU4/inter-op1/workers0, no probing. Atomic checkpoint
every128updates, heartbeat every256. Real128+128vs256pilot checks resume exactly.
Keep5GiBfree, cap new private artifacts at1GiB, allow healthy execution up to12h.
Resume the same identity after interruption; do not duplicate running jobs.
Freeze24primary fits and24complete refits before evaluating. Final replay forbids
fitting, verifies predictions and preserves public scientific files.

fresh_run: donor feature/label generation, risk fitting and development readout.
cached_verified: crossfit bank, own-B inputs and old controls with hashes/lineage.
not_run: new outer test, three-seed independent confirmation, full-policy upgrade.
Native image-coordinate8/12silver labels, no verified seconds or metric scale;
annotation-generation causality remains unverified. No true3D/foundation claim,
Stage5C execution or SMC. No new CREATE job is required by the local pilot.
