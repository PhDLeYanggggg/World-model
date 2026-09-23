# Whole-Source Reservations and Executable Admission Guard

## Decision

I reserve the complete DroneCrowd collection for candidate external confirmation,
not training, model selection or calibration. I reserve the two-location DUT
collection for candidate calibration, except `dut_intersection_04`, which remains
excluded because two source IDs describe the same moving trajectory. This is a
pre-outcome allocation under the author's delegated research authority, not a
claim that the sources are already eligible for predictive experiments.

The choice avoids dividing unresolved DroneCrowd physical sites across data roles.
Its official train/test folders contain overlapping backgrounds. The 45 conservative
exclusion groups are useful restrictions, not 45 verified independent locations.
Keeping the whole collection in one role preserves all those restrictions, including
possible co-locations that matching did not detect. DUT's 28 recordings similarly
do not provide 28 independent calibration sites; the author describes two locations.

The main task remains eight observed and twelve future native annotation steps.
Raw-frame t+50 is supplemental. Native strides are not assumed to represent equal
time across datasets. Pixel/dataset-local coordinates remain unverified for metric
claims, and the 2% empirical easy-error criterion is unchanged.

## Implementation and Real-Asset Verification

`configs/m3w_external_source_reservations_v1.json` binds all 140 recording identities,
their metadata and source/geometry fingerprints to the existing source receipts,
exclusion constraints, observation disposition and delegated authorization.

- 112 DroneCrowd recordings: confirmation reservation.
- 27 DUT recordings: calibration reservation.
- One DUT recording: excluded in every predictive role.
- 560 cached payloads / 209,752,578 bytes independently hash-checked.
- All 421 attempted wrong-role assignments refused.
- All 139 matching reservations still require separate predictive admission.

The initial freeze is `fresh_run`; exact replay is `cached_verified`. Analysis SHA256:
`4b69baf31c813cfee56c05717c2b3653d51c19d6b4654e81285ea9a88b10f098`.
Registry SHA256:
`cff9232f8cf50537f44afe047678c0d465d6459b3041741fe063a422a3cfed9b`.
Payload bytes were read for checksums, not deserialized into model rows or scored.

The production experiment admission path now checks source reservation before its
legacy declaration-only branch. Renaming a source or clearing a summary flag does
not authorize its reuse. Source-file and coordinate-array hashes detect known copied
geometry. Generic ID vectors and all-true masks are deliberately not duplicate-source
identities. A matching role still cannot bypass the existing source-use review,
quality quarantine, frozen predictor lineage or one-shot evaluation claims.

This is an accidental-use guard in the supported experiment pipeline, not a security
sandbox: arbitrary direct access to raw files remains technically possible. Low-level
source-audit readers are not scientific-admission interfaces. It does not establish
all possible transformed duplicates, historical exposure or geographic independence.

The unrelated portable protocol path remains usable without downloading protected
raw/cache trees. Tests cover missing registries, incomplete inventories, changed
evidence, changed metadata, schema-only identities, copied geometry, quality quarantine
and a production training command that refuses before Torch/checkpoint creation.
Final scoped integration tests: **347 passed**, not the full historical test suite.

## What Has Not Run

No new forecasting, training, real calibration or confirmation was performed.
No accuracy, easy-preservation, neural-contribution or publication-readiness gate is
passed by these engineering checks. Stage5C and SMC remain disabled. Historical
source-use flags in the caches were preserved rather than rewritten to manufacture
eligibility. Existing development-exposed SDD results stay development evidence.

## Remaining Scientific Work

1. Resolve source-use and historical-exposure evidence for the reserved sources.
   This is delegated work, not a new request for routine author approval.
2. Freeze the predictor, matched independent/joint controls, learned preprocessing,
   risk-head producers and all selection rules using development data only.
3. Bind the appropriate new source adapters and separate one-shot claims before
   predictive label access. The current generic contract does not yet support the
   dense DroneCrowd cache or replace unknown sites with invented physical IDs.
4. Treat independent calibration support as genuinely small. Two DUT locations and
   an unresolved DroneCrowd site count do not justify a distribution-free 2% safety
   guarantee. Reserve the confirmation collection rather than consuming it to tune
   this deficiency away. Report empirical transfer separately from formal risk claims.

The whole-source reservation settles where these recordings must not be used. It
does not settle calibration-to-confirmation exchangeability: different collections
and sampling clocks may violate it. Any later empirical evaluation must retain that
limitation rather than treating cross-source separation as proof of IID sampling.

## Reproduction

Use the local arm64 environment. Raw inputs and the two audited caches are required
for full hash replay; none are redistributed in Git.

```bash
.venv-pytorch/bin/python scripts/freeze_m3w_external_reservations.py --verify --verification-receipt verification_local_replay.json
.venv-pytorch/bin/python -m pytest tests/test_m3w_source_reservations.py tests/test_m3w_intake_admission.py tests/test_m3w_experiment_contract.py -q
```

The script refuses changed evidence or existing, conflicting outputs. Use a new
verification receipt name for another replay; previous execution receipts are kept.
It does not launch model training or mutate the source caches.
