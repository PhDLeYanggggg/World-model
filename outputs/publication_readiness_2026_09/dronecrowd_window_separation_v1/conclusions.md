# DroneCrowd Multi-Agent Window Support and Input/Label Separation

## What Changed

2026-09-23. The preceding turn verified read-only CREATE access. This turn
implemented a local annotation reader with separate past-input and future-label
paths, then ran it over all 112 authorized recordings. No HPC was required:
the complete structural audit took 39.966 seconds locally.

`fresh_run`: new reader, shared-query counts, sensitivity to target-filtered
neighbor inventories, and 336 real-recording prefix checks with two perturbations
each. `cached_verified`: archive identity, all XML member identities, release
metadata and the previous audit's implementation hashes; all 672 per-recording
window totals also match the prior independently implemented run-length count.
`not_run`: model training, forecasts, error readout, calibration, confirmation,
image download, physical-site grouping or camera-motion estimation.

The author has authorized annotation acquisition/audit and the independent-scene
route. Scientific data roles, final source format and the external observation
protocol are not assigned by this engineering work. XML is an explicit audit
reference here, not a silently approved replacement for MAT/TXT scientific labels.

## New Structural Evidence

The totals below pool the release's 82 train and 30 test recordings for **data
quality inspection only**, not for fitting. Every query in each row has at least
10 complete target agents. The release partition is not an M3W role assignment.

| Annotation-index probe | Shared query times | Complete agent windows | Non-overlapping raw intervals within clips | Queries losing complete-history neighbors under target filtering |
|---|---:|---:|---:|---:|
| Observe 8, predict 12, stride 1 | 31,472 | 4,458,777 | 1,680 | 22,658 (71.99%) |
| Observe 8, predict 12, stride 12 | 8,064 | 930,714 | 112 | 7,874 (97.64%) |
| Observe 8, raw t+10 | 31,696 | 4,500,642 | 1,792 | 21,445 (67.66%) |
| Observe 8, raw t+25 | 30,016 | 4,191,882 | 1,008 | 26,238 (87.41%) |
| Observe 8, raw t+50 | 27,216 | 3,699,886 | 560 | 25,612 (94.11%) |
| Observe 8, raw t+100 | 21,616 | 2,791,746 | 224 | 21,093 (97.58%) |

At stride 1 the 8/12 probe has 22..418 complete target agents per query.
These are overlapping windows. Even the temporally disjoint intervals can share
people, a camera and a physical site; **none of these counts is an independent
sample size**. There are 112 clip IDs, but the number of independent physical
sites is still unverified. The stride-12 probe consumes 229 raw frames including
history and future, so a 300-frame clip supports only one disjoint interval.
The probe does not choose a new frame rate, confirm seconds, or change the
approved eight-observation/twelve-prediction main task.

Requiring every box to stay inside the nominal image bounds would remove 7,247
of the stride-1 8/12 target windows and 6,279 raw-t+50 target windows. This is
a reported sensitivity, not an adopted exclusion rule; no box was clipped and
the raw source was not modified. Labels remain released head-box centers.

## A Future-Availability Trap Prevented

A convenient but invalid construction would first find agents with a complete
future trajectory and then use only those agents as the model's observed crowd.
That makes the input inventory depend on later disappearance or occlusion.

On these annotations, this hypothetical construction would remove **62,135
agent-query instances with complete past histories** in the stride-1 8/12
probe, affecting 22,658 queries. At raw t+50 it would remove 214,585 such
instances. These are repeated instances, not distinct people, and not a measured
effect on prediction accuracy. This audit does not establish that an earlier
M3W result actually used this faulty construction.

The new reader avoids it explicitly:

- Current visibility, not future lifetime, defines the input agent inventory.
- Partial past histories remain present with masks; gaps are never interpolated.
- Backward velocity is valid only when its entire raw interval is visible. Its
  first history entry is masked; no central difference or future frame is read.
- Future positions, future visibility and complete-target eligibility live in
  a separate label dictionary. They never remove neighbors from past inputs.
- No total clip track count, future goal or future lifetime enters the input.
- Archive/member hashes are provenance outside the input payload, not features.

The [machine-readable schema](window_schema.json) describes this diagnostic
reader. It is deliberately not registered as an approved training dataset;
the existing intake-admission refusal remains unchanged. No feature store,
train-only prototype or model rows were exported.

## Verification and Limits

Eighteen new tests cover future disappearance, future-only births, future
coordinate/inventory changes, partial histories, raw gaps hidden by subsampling,
backward-velocity units, bounds flags, invalid requests and counting distinctions.
Together with prior archive, metadata, provenance, admission and experiment
contract tests: **162 scoped tests passed**. This is not the full legacy suite.

Across every recording, queries at raw frames 7, 149 and 298 were checked.
Removing all later frames/future-only tracks and separately perturbing later
coordinates/visibility produced **zero input mismatches over 672 checks**.
When future labels were available they changed as expected. This verifies the
implemented dependency boundary on exported rows, not the provenance of those
rows: unrecorded VATIC interpolation may itself depend on later controls.
No sensor-time causality guarantee is inferred.

The first test run had the expected missing-module error before implementation;
the next exposed an inconsistent malformed-XML exception type, now normalized
to an explicit refusal. There was no training or numerical experiment failure.
All six counting profiles reproduce each prior per-agent structural total for
each of the 112 recordings. A complete offline replay reproduces the new audit.

```bash
.venv-pytorch/bin/python scripts/audit_m3w_dronecrowd_windows.py --verify
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_dronecrowd_windows.py tests/test_m3w_dronecrowd_archive.py tests/test_m3w_dronecrowd_intake.py tests/test_m3w_dronecrowd_verification.py tests/test_m3w_annotation_export_provenance.py tests/test_m3w_intake_admission.py tests/test_m3w_experiment_contract.py
```

Analysis SHA256:
`3a42fc7557e9c2ad5e1b6e4aabc1d77803a9f0f1b57f5b1a93814857193295dc`.
The archive and metadata remain the hash-bound official snapshots described
in [the acquisition audit](../dronecrowd_annotations_v1/conclusions.md).
No third-party code, image acquisition, remote job or authentication change ran.

## Priorities After This Audit

1. Establish physical-site groups and camera movement from source evidence,
   ideally sparse official imagery. The image-acquisition question remains
   unanswered; the CREATE handoff does not authorize that download.
2. Dispose of the XML/MAT/TXT differences and explicitly define offline-annotated
   versus sensor-time observations. More compute cannot resolve this semantic
   choice. Pin external stride/time conventions without inventing seconds.
3. Freeze source/site-disjoint fit, selection, calibration and confirmation roles
   before predictive readout. Previously explored SDD sites remain development.
4. Then run the fixed-predictor intervention controls and three-seed experiment;
   retain matched-budget, easy, tail, worst-site and interaction metrics.

There is substantial multi-agent annotation support, so raw row volume is not
the immediate blocker. Independent scenes, camera geometry, observation
provenance and approved experimental roles remain the decisive missing evidence.
The current research remains a pixel/raw-frame 2.5D trajectory/world-state
system, not a verified 3D, metric, foundation, deployable new model or
submission-ready result. Stage5C and SMC remain off.
