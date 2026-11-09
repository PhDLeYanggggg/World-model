# HT21: how many detection tracks can be scored together?

I used the released detection boxes to check whether denser labels solve a
problem I found on AerialMPT: a clip can have plenty of labeled pedestrians
but still lack a complete set of futures for the tracks a detector emits.
HT21 improves individual coverage. It does not solve the whole-roster problem.

## Results

| Raw frame stride | Online 8-step histories | Complete 12-step futures | Reference histories with complete futures | Multi-agent windows with all futures | Windows with at least two complete futures |
|---:|---:|---:|---:|---:|---:|
| 1 | 20,702 | 20,064 (96.9%) | 51,600 | 36 / 285 | 285 |
| 5 | 2,839 | 2,575 (90.7%) | 8,030 | 2 / 57 | 57 |
| 10 | 859 | 727 (84.6%) | 3,047 | 0 / 22 | 22 |

At stride 1, the 638 unscored or incompletely scored histories include 236
without a stable label match, 386 with partial futures, and 16 with no future
label. The 36 fully supported windows come from HT21-01, -02 and -03 (10, 22
and 4); HT21-04 contributes none. At stride 5, the two come from -02 and -04.
The recordings are not established independent sites. An individual track
can recur across disjoint windows, and the three cadence arms overlap.

High coverage of emitted tracks does not mean high reference recall:
20,064 of 51,600 complete stride-1 reference histories were recovered (38.9%).
Neither the complete cases nor a handpicked pair should stand in for a full
crowd. The drop at wider stride combines tracking and label continuity;
this check cannot assign it to one cause or select an evaluation cadence.

## Protocol and boundaries

- Four [HT21/CroHD](https://motchallenge.net/data/Head_Tracking_21/)
  publisher-train recordings only. For each stride, take the first 8 of every
  disjoint 20 sampled positions as observations and use the next 12 solely to
  count label availability. No new forecast, calibration or confirmation role
  was assigned. Publisher-test members were not opened by this audit.
- Start a fresh causal tracker per window using all released boxes and no GT
  identities. Hash its observed positions and assigned identities per clip
  and cadence before reading that clip's GT for matching. One-to-one box
  matches require IoU of at least 0.10 and center distance no more than half
  the label diagonal (with a one-pixel floor). The same pair must match at all
  eight observations. A complete future requires all 12 positive, visible
  human labels. Unmatched and partial tracks stay in the denominator.
- These are frozen publisher detections, not new image inference. The exact
  detector checkpoint, its training exposure and the provenance of the box
  stream are not verified. GT interpolation and moving-camera geometry remain
  concerns. Head-center image pixels are not ground-plane measurements, and
  raw strides here are not an aligned prediction clock across datasets.
- `fresh_run`: prefix association and support counts. `cached_verified`:
  input archive hash and byte-identical prefix/count replay. A separate CSV
  and set-intersection count verified all 12 reference denominators. 30 scoped
  tests passed. The first parse exposed a frame-column indexing bug; it was
  rejected, and the corrected ingestion now has an end-to-end regression test.
- `not_run`: prediction error, learned or joint selector fitting, independent
  risk certification. The existing 2% rule and deployment policy did not
  change. No reserved confirmation results, Stage5C or SMC were used.

## Replay

The [source receipt](../ht21_annotations_v1/source_manifest.json) identifies
the local labels-only archive. Only hashes and aggregate counts are published
here; no source boxes or track rows are included.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/audit_m3w_ht21_detection_support.py --verify
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_ht21_detection_support.py tests/test_m3w_ht21_admission.py tests/test_m3w_ht21_cadence.py
```

The next source needs both complete online rosters at the intended cadence
and enough independent locations. Reusing these clips at stride 1 may inform
tracking development, but it does not establish joint forecasting utility.
