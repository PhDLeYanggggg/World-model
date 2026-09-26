# Support-Conditional Fractional-Harm Intervention

## Material Passport

Registered before this intervention's training/readout. Parent 42554148 is
cached_verified: 55 public artifacts, 88 source bindings and 144 matched mean
heads are checked. Prior held-source outcomes are already development-exposed.
This is a new loss experiment on that development design, not independent
confirmation or a new trajectory/policy result.

## Hypothesis and Distinction from Failed Repairs

The parent full risk head has median harm-event AUROC 0.792 on all rows but
0.486 on positive forecast disagreement. It has some costly-tail information,
yet harm coverage remains unstable. Test whether a bounded fractional-harm
auxiliary on causal positive-disagreement support improves held-locality harm
estimation without sacrificing tail retrieval or reference-cost fitting.

Existing `m3w_hurdle_risk.py` adds binary occurrence BCE on all known rows and
severity MSE only on positive harm. Existing `m3w_ranked_hurdle.py` adds pairwise
ordering of a realized H/(B+H) label. Neither is this auxiliary: it supervises
continuous H/e and H_easy/e on every known row with e>0, including zero harm,
without splitting occurrence from positive severity or constructing pairs.
It does not introduce a new forecasting architecture or claim a novel scoring rule.

For each sampled locality, average fractional Bernoulli cross-entropy across
the all/easy harm fractions and its positive-disagreement rows. Average these
locality terms, then add them with fixed coefficient 1 to the unchanged
four-moment, train-RMS-normalized MSE. Empty-support locality batches contribute
no auxiliary. Structural-zero rows remain in the original moment objective.
No oversampling, coefficient search, target clipping of true high errors,
threshold changes or refitting to C outcomes.

The sample target is bounded by the causal forecast-disagreement envelope e.
For a soft target q in [0,1], expected cross-entropy in a prediction p has its
minimum at E[q]. This elementary mean-target identity is not a calibrated
failure probability, finite-sample guarantee or domain-shift theorem. Gradient
arithmetic clamps predicted fractions at 1e-7 and 1-1e-7; the delivered moment
prediction is not clamped or rescaled. Target overshoot above one is allowed
only within 1e-5 numerical tolerance and is counted; larger violations stop.

[Gneiting and Raftery (JASA 2007)](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)
provide the general proper-scoring-rule context. Reading scope here is the
abstract and introduction, not full theorem verification. The use of a
fractional target is our loss choice; the cited work does not establish its
performance, novelty, calibration or transfer in this experiment.

## Controlled Fit

All six ordered source assignments, seeds17/29/43, full/motion-only pairs and
four leave-one-B-locality-out folds: 144 new heads, 288,000 updates. Match the
original mean head's width64, 24,836 parameters, initialization, batches,
AdamW lr0.0003/wd0.0001, clip5 and 2,000 updates. A zero-coefficient regression
must reproduce original weights exactly; direct/resume runs must match exactly.
Every real head must match its cached control's sampler counts/state and fixed
diagnostic batch. Extra objective computation means equal updates, not equal
wall time. The original control is not retrained unnecessarily.

The same three fitting localities determine normalization, cost scale, easy
cut and score bins. Only causal features and predicted rollouts enter inference.
No future input, central velocity, test goals or held-label preprocessing. Use
the parent's envelope-positive causal subset, not a B-fitted eligibility mask.
Inner easy definitions vary; do not pool their rows as one event population.

## Frozen Readout and Decision Rule

Freeze and push all heads, held scores and training bins before current held
readout. The primary comparison is positive-disagreement easy-harm MSE versus
the matched mean control. Calculate relative MSE improvement per locality,
average three seeds within locality, then 3,000 resamples of four localities
per source assignment, seed47131. Keep every role and both forecast pairs.

Report all/conditional AUROC, AP/prevalence, top10 harm capture, harm coverage,
four component MSEs, weak support and training loss. Reference error changes
are retained as potential damage, not hidden behind improved fractional loss.
Secondary contrasts: top10 harm capture (pp), AUROC, and reduction in absolute
log harm-coverage error. Unsupported comparisons are not_estimable, not zero.

Advancement to a new policy experiment requires six positive primary full-pair
MSE intervals and no negative full conditional tail-capture or coverage-error
interval. Motion-only is a required robustness report, not a substitute for a
failed full result. This engineering/development gate cannot authorize a
deployment or independent-risk claim. No C policy is evaluated this round.
Do not pick a favorable seed, role or coefficient after readout.

## Runtime, Data Roles and Publication Boundary

Local native-arm64 CPU4, interop1, workers0, checkpoint/optimizer/RNG resume,
PID/heartbeat and 10GiB free-disk reserve. Run a real 100-update pilot, then
resume inside the fixed budget. Parent local training was inexpensive and
memory fits; no new CREATE job is needed. The last read-only queue receipt is
dated 2026-09-26 00:19 UTC, not a fresh availability claim. Remote M3W asset
inventory is still not_run because an authorized project path is unverified.

Six opened selection localities are not used this round; twelve reserved
calibration and six confirmation localities remain closed. Primary obs8/pred12
annotation steps, raw stride12 and the 2% risk tolerance remain unchanged.
Image pixels and detector-derived labels only, not metric/seconds, human gold,
physical safety, true3D, foundation or submission-ready. Stage5C and SMC stay off.
