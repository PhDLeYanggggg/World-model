# DroneCrowd Multi-view Exclusion Groups

## What Changed

`fresh_run`: all 6,216 recording pairs were screened using first/middle/last
images. The run attempted 55,482 frame pairs, with early stopping after strong
support in two distinct frames on each side. It found **66 strong multi-view
pairs and 46 ambiguous pairs**. Of these, two strong and nineteen ambiguous
pairs cross the supplied train/test folders. Ambiguous evidence is retained as
a no-cross-role constraint, not silently treated as independence.

The original 64 positive first-frame relations are preserved even when the new
screen fails to rematch them. Combined strong/prior evidence gives 66 components;
including ambiguous relations gives **47 conservative components**. These are
exclusion groups, **not 47 verified independent physical sites**.

A separate source-image review adds four explicit constraints, two corroborating
existing associations and two conservatively linking possibly adjacent views.
The resulting machine-readable manifest has **45 exclusion groups**, including
five groups that mix the supplied release roles. The original release split is
therefore rejected for our independent-scene protocol.

## Source Evidence

| Relation | Observation | Treatment |
|---|---|---|
| 00002 / 00097 | Corresponding market awnings, facades and roof layout under changed illumination/view | Corroborates image matching |
| 00007 / 00073 | Same circular monument, paths and canopy under rotation and low light | Retain despite weak automated support |
| 00013 / 00015 | Possible shared high-rise corner and junction from opposite views | Uncertain shared-site constraint, not exact-match proof |
| 00019 / 00028 | Possible shared curved building, lawn and paths in adjacent campus views | Uncertain shared-site constraint, not geographic certification |

Seven first-frame contact sheets and the listed full-resolution source pairs
were inspected. This is self-audited visual evidence, not human-gold annotation.
Image identities are bound by SHA256 in `exclusion_constraints.json`. Raw images
and contact sheets remain local and are not committed.

The five mixed-release groups are:

- 00009, 00010, 00065.
- 00013, 00014, 00015, 00016, 00054.
- 00031, 00032, 00033, 00036, 00037, 00038, 00042, 00043, 00044.
- 00075, 00076, 00077.
- 00094, 00095, 00096.

This fixes a concrete split hazard; it does not prove that all remaining groups
are unrelated. Adjacent streets, different sides of buildings, low-texture views
and different campus areas can still share a physical site without enough planar
correspondences. No matching method used here has a certified false-negative
rate. Negative matching results must not be used as an independence certificate.

## Method and Execution

Fixed configuration: 960x540 grayscale, CLAHE, RootSIFT, visible-head enlarged
masks, mutual nearest ratio matches, deterministic RANSAC and two-sided inlier
coverage. Strong support requires at least 25 inliers, 0.40 inlier fraction and
0.04 hull coverage in each image; ambiguous support requires 12, 0.30 and 0.01.
The median symmetric reprojection error must not exceed 2.5 resized pixels.
All thresholds were source-audit choices, not selected using forecast outcomes.

The timing pilot extracted 336 feature sets and evaluated 50 pairs in 54.07
seconds. The continuation reused those verified caches and computed the remaining
6,166 pairs in 1,277.29 seconds. Together, the original source work took about
22.2 minutes locally. No source run was downgraded or stopped for slowness.

An initial completed-run replay exposed tuple/list JSON equality in the controller.
The scientific extraction/matching code, source/config hashes and all computed
pairs were unchanged. The producer controller is preserved in Git `ca2af46b`
(SHA256 `4dd2ba84538dc1175d42c4e56fcf807cb6d44e5cfab6554387672411f8f6353d`).
The fix permits a narrowly pinned **completed-run-only** compatibility replay;
changed scientific code, sources, config, runtime or incomplete caches are refused.
The new controller checksum is recorded separately in `verification_execution.json`.

`cached_verified`: all 336 image/feature checks and 6,216 cached pair receipts
reconstructed the exact analysis in 0.555 seconds, with zero new features/pairs.
This is cache/hash/aggregation verification, **not a second feature-matching run**.
The constraint manifest also verifies the graph and rejects cross-role relations.
The final scoped source/reader/contract suite passes 230 tests in 3.81 seconds,
including eight completed-run identity and JSON replay checks. This is not the
entire legacy repository suite.

Analysis SHA256:
`896c9f64937c3bab5ee59af5fdbd78fca2b0b1f8a688c7754bf66d68bcc50cd6`.

Constraint manifest SHA256:
`12c8479daf35215d47bde4111aaf74c56849947b1c0d8c43476d9c5778353976`.

## Research Decision and Remaining Gate

Do not admit the official folders as independent-scene fit/test roles. All clips
remain unassigned/quarantined; future role manifests must obey every automatic,
prior-positive and visual-review constraint. The lazy recording cache is ready
for integration but does not grant experimental admission.

Next, resolve or conservatively exclude remaining possible shared-site views,
bind the exclusion manifest into the producer-chain contract, and freeze fit,
selection, calibration and confirmation roles **before any predictive readout**.
Where independent units remain unverified, report that limitation and do not
substitute overlapping windows as calibration sample size. This technical work
is delegated; no additional routine user approval is pending.

No forecasting, fitting, loss readout, independent calibration, confirmation or
deployment was performed here. Image-to-image transforms are not metric ground
homographies. Effective seconds, sensor-time annotation provenance and scale
remain unverified. This is image-pixel/raw-frame offline annotation research,
not true 3D, foundation-model success or submission readiness. Stage5C/SMC stay off.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/audit_m3w_dronecrowd_grouping.py --resume
.venv-pytorch/bin/python scripts/build_m3w_dronecrowd_group_constraints.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_dronecrowd_grouping.py tests/test_m3w_dronecrowd_grouping_resume.py
```

Omit `--resume` for a new source audit. Keep the ignored descriptor/pair caches
to resume or replay the local completed run. The original producer can be
reconstructed from the pinned Git revision; no downloaded third-party code is
executed by this audit.
