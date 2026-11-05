# Producer-A Crossfitting For Risk Donor Coverage

The fixed-budget risk continuation reduced fitting loss in all eight heads but
failed all three decision screens. Their donor banks contain only one or two
sources. This experiment repairs donor availability, not the evaluation rule.

## Hypothesis And Scope

Forecasts held out by source inside each original producer-training A set can
provide more diverse risk-learning donors. Blind pooling of existing critic
caches is disallowed: shared teachers do not override outer roles, and other
teachers may have trained on the intended critic sources.

Train six EqMotion controls, seed17, 4,000 updates each, with the existing
architecture, source-balanced loss, optimizer and native obs8/pred12 inputs.
For producer folds0 and1, reuse the existing three-way source partition as
internal held-A folds. Each model starts randomly, fits only A-minus-inner-fold,
and predicts that inner fold only. Baseline selection and loss normalization
use its fitting sources only. No early stopping or checkpoint selection.

The new models never train on or predict their parent context's outer C/H.
The same physical source may have different roles in another context; permission
is checked separately for each parent. The current exposed own-B populations,
outer splits and protected sources are not changed. No reserved input readout.

## Outcomes

Primary deliverable: complete row-aligned OOF donor bank with teacher ancestry,
source coverage, per-source label support, finite forecasts, exact resume and
replay checks. Fit all six models before reporting donor outcome summaries.
Prediction ADE and FDE against causal CV are training-donor diagnostics only,
not main held performance or a deployment comparison. Masked labels remain
unknown. The original 24-group primary remains NA, including empty group068.

A later risk-learning experiment must compare fixed own-B-only versus augmented
donors, retaining source exclusions and a fixed information screen. Training on
more donor sources does not itself prove risk transfer. New teachers fit fewer
sources than the original teachers, so teacher-size/distribution shift remains.
No blind replacement of the original risk targets, thresholds or teacher IDs.

## Runtime And Evidence

Native arm64 .venv-pytorch, CPU4/inter-op1/workers0. Atomic checkpoint every200
updates; heartbeat every50. Real 20+20 versus40-update recovery check before
the main run. The pilot prefix may continue only as the identical first job.
Keep a5GiB disk reserve; artifacts capped at1GiB; healthy runtime up to12h.
On interruption resume unchanged identity; never restart a healthy job.
Private arrays/checkpoints are not committed. Public config, code, source-role
audit, loss traces, result hashes and commands are committed.

fresh_run: new fitting, packing and forecasts. cached_verified: pinned metadata,
source admission, code and partitions. not_run: downstream risk-head test,
new C/H evaluation, three-seed confirmation, deployment and submission.

Original2% all/easy harm, signed easy-net and support requirements stay intact.
Empty controls remain NA, not pass; no identity-fallback certificate. No new
CREATE job. Image-coordinate native steps, not verified meters/seconds, true3D,
foundation, human-gold labels, Stage5C execution or SMC.
