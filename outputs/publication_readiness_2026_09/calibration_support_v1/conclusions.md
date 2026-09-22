# Calibration Support: Exclude the Entire Producer Chain

## Material Passport

Date: 2026-09-22. Fresh metadata-lineage audit, with hash-verified existing
cost heads, forecasts and reports. No future-target array was deserialized, no
model fitted, no role assigned, and no calibration or final test executed.
This turn adds a tested refusal mechanism and a real-artifact audit, not an
accuracy result. The last fixed experiment retains its 4.10% developmental ADE
improvement and failed per-scene easy protection.

## Fresh Findings

The audit binds 1,104 dependencies, including 12 completed heads and their
producer manifest. All 12 current outer views retain valid fitting exclusion.
The failures below concern **proposed reuse for an additional calibration role**,
not a new finding that the completed outer-held development scores leaked their
targets.

| Proposed shortcut | Rejected views, of 36 | Reason |
|---|---:|---|
| Reuse the head on one of its three fitting sites | 36 | Direct fitting/preprocessing exposure |
| Remove that site's rows and refit only the head | 36 | Scoring predictor trained on the site |
| Also replace the scoring predictor with the pair-excluded model | 36 | Retained cost-target producers still trained on the site |

The last case contains 72 exposed target-producer incidences. A held site's
outcomes influence forecasts used as another site's cost-training inputs and
labels. Removing the held rows alone does not remove that indirect path.

Example: outer=coupa, proposed calibration=deathCircle. After removing
deathCircle from head fitting, the gates target producer still trains on
deathCircle+hyang, and the hyang producer on deathCircle+gates. The current
outer-scoring predictor also trains on deathCircle. All these ancestors must
exclude the calibration site, not just the final head's row mask.

`require_calibration_exclusion` now rejects head, preprocessing, scoring-producer
and target-producer exposure before calibration reads. It refuses unknown
parented/pretrained producers in this specific random-initialized source design.
The general experiment contract still governs approved roles and recursive
artifacts. This helper is not an independence certificate.

## Strict Nested Development Requirements

Four sites and three seeds give 36 proposed two-fit/one-calibration/one-outer
views. Existing 18 pair-excluded predictors can forecast both the calibration
and outer sites using the same two-site-trained model. Cost training on the two
fitting sites requires **12 additional triple-excluded predictors**, each trained
on only one site, plus **36 new cost heads**. Head preprocessing must exclude
both calibration and outer sites. No current head can simply be reused.

The hypothetical corrected graph passes 36 fitting-exclusion checks. Those
models are not trained and their performance is unknown. With the inherited
4,000-update/batch64 predictor settings, the additional predictor budget would
be 48,000 updates and 3,072,000 draws. Heads at 12,000 updates/batch256 would add
432,000 updates and 110,592,000 draws. These are feasibility estimates, not an
approved new training matrix or completed compute.

The previous 18-predictor run took 7.75 hours locally. A new real pilot should
estimate single-site training and cache cost before choosing local versus HPC.
This is not a reason to shorten a subsequently registered training budget.

Even that repaired experiment remains development: all four sites have
influenced design, and each policy has only one calibration site. Rotating sites,
changing seeds or collecting overlapping windows does not create independent
calibration scenes or an untouched final test.

## Statistical Support Is Separate

The existing screening function was exercised on synthetic best-case zero
losses, one fixed policy, one unit-range risk and illustrative delta=0.05.
Assuming independent clusters, its Hoeffding/union-bound upper limit is 1.0 for
one cluster, 0.86541 for two and 0.61194 for four. These are arithmetic examples,
**not real calibration, M3W risk estimates or the project's 2% easy-ADE rule**.
They expose the limitations of this particular generic bound with few scenes,
not an impossibility theorem for every method or assumption. The analytically
unchanged baseline is a separate identity, not a learned zero-risk claim.

The calibration backend requires an approved past-normalized bounded risk; the
current main readout is native ADE with equal-site relative gains. A 2% relative
easy-ADE diagnostic is not a 2% harm-event probability or clipped-excess mean.
Do not silently change the risk functional, tolerance, cluster definition or
missing-label policy to obtain a passing bound.

Learn then Test requires valid calibration tests for a stated risk and handles
selection among a fixed policy family. Our training weights do not implement
that procedure. Reading scope here: sections 1.1 and 2.1-2.2, not all appendices.
[Primary paper](https://arxiv.org/html/2110.01052v5).
Subgroup calibration is also established prior work; the abstract alone was
rechecked here. It is not novelty supplied by renaming a selector.
[Multicalibration](https://proceedings.mlr.press/v80/hebert-johnson18a.html).

## Independent-Asset Check

DUT screening was freshly rebuilt from existing source, conversion and quality
receipts. All 28 metadata records validate; 112 proposed fit/development/
calibration/confirmation admissions are refused. Actual new admissions: zero.
Label API calls, training and accuracy evaluations: zero. One recording retains
its duplicate-track quarantine; the remainder need source-use, exposure and role
decisions. This is not a failed forecasting experiment.

The current author repository describes two campus locations and 28 clips, not
28 independent scenes. It requests citation; this check does not resolve the
project's outstanding source-use approval. The locally identified raw/filtered
coordinate discrepancy remains unresolved, so no metric or seconds claim is
added. [Author repository](https://github.com/dongfang-steven-yang/vci-dataset-dut).

## Priority and Decision Boundary

1. Establish independently usable scene support and freeze calibration versus
   confirmation purposes before prediction readout. Resolve source eligibility
   and historical exposure instead of presuming them from file availability.
2. If another source-internal development run is chosen, exclude the complete
   producer chain with the additional fits above. Do not call it independent
   confirmation or calibrated deployment.
3. Define a risk functional compatible with the claim while retaining the 2%
   empirical easy criterion and exact-zero-reference checks.

The author has been asked to choose independent-scene-first (recommended) or
strict nested development. No new assignment has been executed while that
decision is pending. The audit runs locally with 54 GiB free. No new CREATE job
or current remote-queue claim is made.

## Verification and Limits

63 scoped tests pass in 17.91 seconds, including 10 new audit tests. Fresh audit
and exact verification agree. Analysis SHA256:
`7905e86a27e61d42b9b024b1868c97b4569dcbe452c6c9164951daeff7a20fd0`.
Tests cover direct/indirect exposure, preprocessing/selection ancestry, unknown
parents, incomplete views, the hypothetical corrected graph and the bound
illustration. This is same-agent engineering verification, not independent
research replication.

The evidence remains annotation-pixel 8-to-12 development. Missing futures,
overlapping windows, repeated design exposure and four sites limit inference.
Original closed roles remain closed. No metric/seconds, true3D, foundation or
deployment claim. Stage5C and SMC remain off; submission is not ready.
