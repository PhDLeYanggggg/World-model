# Original XML Does Not Necessarily Establish Causal Observation Provenance

## Material Passport

2026-09-22. `fresh_run`: primary-source review, three pinned VATIC source files
(76,122 bytes), a constructive synthetic dependency check and five regression
tests. `cached_verified`: exact offline source/check replay and the previous
DroneCrowd metadata snapshot. `not_run`: DroneCrowd annotation download/read,
actual interpolation-rate measurement, camera-motion estimation, new training,
risk calibration, independent confirmation or forecasting evaluation.

The previous turn established the official metadata and conversion convention.
This turn changes the next acquisition criterion: obtaining the release's
original XML may still be insufficient for a strict observation-time claim.
It does not conclude that actual DroneCrowd histories leak future information.

## Primary Evidence and Exact Scope

DroneCrowd's paper section 3.1 states that VATIC was used for annotation and
double checking. Section 3.2 evaluates counting, localization and tracking;
those protocols do not establish the source-time eligibility of inputs to a
future forecasting experiment. The reviewed sections do not provide a per-clip
physical-site/camera-motion mapping for our proposed calibration roles.
[Author paper, sections 3.1-3.2](https://arxiv.org/pdf/2105.02440).

The official VATIC repository is inspected at commit
`7de990ac0f7882dc0420b0f529b08951ae0f1230`. Git-blob and SHA256 checks pin every
read file. `cli.py` lines 442-448 apply `LinearFill` to the export track boxes.
The XML writer at lines 649-668 records frames, boxes and visibility but does
not serialize the generated flag or keyframe provenance. Other export paths
preserve a generated flag. This is static source reading, not execution of the
third-party program. The exact VATIC version/export route used for DroneCrowd
is unknown; this code cannot establish that all or any of its released rows
were generated in this way.
[Pinned author exporter](https://github.com/cvondrick/vatic/blob/7de990ac0f7882dc0420b0f529b08951ae0f1230/cli.py#L442).

The release's `xml2mat.m`, already hash-bound in
[the preceding metadata audit](../dronecrowd_metadata_v1/conclusions.md), cannot
restore omitted provenance. A raw XML is upstream of that converter, but is
not necessarily upstream of all annotation interpolation.

## Tested Boundary, Not a Dataset Result

The constructed example has a query at frame 8 with eight nominal past inputs
at frames 1..8. Their positions can be interpolated from controls at 0 and 10.
Changing only the control at frame 10 changes all eight inputs and changes the
backward finite-difference velocity from [1,0] to [2,0]. Thus a backward velocity
formula does not repair future dependence already present in supplied positions.

Direct dense positions and sparse-control interpolation produce byte-identical
unflagged XML in the witness. The existing structural reader correctly keeps
interpolation provenance unresolved and exports no model rows. Missing flags
must not be interpreted as all-original annotations. This is an identifiability
counterexample, not measured DroneCrowd contamination, not a performance result,
and not a proof about how human annotators obtained individual positions.

65 scoped tests pass in 0.22 seconds, including five new source-boundary tests
and the previous metadata/XML and UCY control-lineage tests. No full legacy
suite claim. Exact replay matches analysis SHA256:
`7a4a9b36fcbe79980a01ab00b56ddb7ada8d549ba189f4a1674367b7c472dd07`.

```bash
.venv-pytorch/bin/python scripts/audit_m3w_annotation_export_provenance.py --verify
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_annotation_export_provenance.py tests/test_m3w_dronecrowd_intake.py tests/test_m3w_dronecrowd_verification.py tests/test_m3w_zara_media_lineage.py
```

First source acquisition uses `--download-source`. The allowlist is three pinned
official code/documentation files. None is imported or executed. It never
downloads the annotation archive or accepts the outstanding warning. All
third-party files remain ignored by Git.

## Consequence for M3W and the Next Data Check

The current SDD adapter explicitly reports `offline_annotated` observations;
this distinction remains unchanged. Past-index-only access and sensor-time
causality are separate claims. This new source review neither silently changes
that approved task nor retroactively establishes or invalidates its performance.
Existing SDD generated/control audits and source-lineage reports must be read
with each experiment's stated observation mode.

After the pending archive permission, inspect whether actual XML retains useful
provenance before converting anything. If flags/control sources are absent, do
not infer them from linear-looking motion. A strict construction-time audit
would need the underlying control points/export provenance or another verified
past-only observation route. A mere generated flag may still be insufficient
without its producer semantics and latest source-frame dependency.

An offline-annotated external benchmark may remain scientifically useful with
appropriate disclosure and approved independent roles, but it is not a silent
substitute for an online sensor-causal experiment. No new choice between these
tasks is made here. Camera motion, head-versus-ground-point semantics, physical
site grouping and scale/time mapping also remain unresolved. The 112 clip IDs
and 70 source-reported scenarios must not be counted as verified independent
calibration sites.

The two existing author questions remain pending; no extra confirmation or role
assignment was invented. No new CREATE job or live remote queue inspection.
The 4.10% development ADE result and failed per-scene easy protection remain
unchanged. No metric/seconds, true-3D, foundation, deployment, risk-guarantee or
submission-ready claim. Stage5C and SMC remain off.
