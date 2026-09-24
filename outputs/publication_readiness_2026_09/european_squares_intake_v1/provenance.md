# Source Provenance and Admission Limits

2026-09-24. Status: fresh metadata/code audit; complete trajectory audit pending.
No prediction errors, model selection or calibration outcomes are opened.

## Why This Source

The moving-zero support diagnosis found very few relevant source events and an
annotation-time provenance concern. A different threshold search on the same SDD
tracks would not repair either problem. The European Squares release offers
many nominal locations and a pre-smoothing tracker export, making it a concrete
candidate for additional support and independent site allocation.

The official [v1.1 Zenodo record](https://zenodo.org/records/18267205) identifies
the dataset as CC-BY-4.0. Its overview lists 39 square IDs; the two statistics
tables contain 147 comparative and 244 seasonal recording records. These are
metadata counts, not a local count of valid trajectories or independent sites.
The 9,451,499,010-byte trajectory archive is being acquired with resume and a
15 GB free-space reserve. Dataset identity, MD5 and SHA256 are retained.

## Metadata Problems Found Before Admission

The season table assigns both Varberg, Sweden and Biberach, Germany to ID 9;
seven records have the Biberach name under this ID. It also contains coordinate
strings `57,106,027` and `12,252,087`, which cannot be accepted as numeric degrees
without an explicit repair. The adapter preserves these conflicts and does not
infer a median location across inconsistent identities. The official overview
assigns Biberach to ID 8 and Varberg to ID 9. Raw archive paths must be checked
before a recording-level reconciliation is proposed.

There are distinct nominal squares within Trencin (67/68) and Brzesko (104/105).
These are not automatically independent calibration units. Physical locality,
recording aliases and shared cameras require review before any allocation.
No square is admitted to independent calibration or confirmation by this intake.

## Processing Code Versus Description

Code is inspected as text at publisher commit
`d7225dd37d6b2bf962ee8533f9ca603593f5c1fc` in the
[official repository](https://github.com/kaktusracing/pedestrian_trajectories).
All retained source files match their pinned Git blobs. No publisher code,
weights, videos or live webcams are executed or accessed.

- `Object_detection.py:29` defines a per-frame callback: current-frame detection,
  person/confidence filtering, ByteTrack update and immediate CSV append. A
  smoother is constructed elsewhere but not applied by this callback. This is
  evidence about the published code, not an execution trace for the archive or
  a verification of the exact installed tracking library version.
- `calc_center.py:23` and `:27` compute and round the bounding-box center in both
  axes. The repository's prose describes the bottom of the box instead. Our
  input explicitly names its representation `box_center`, retains all four box
  edges, and does not substitute processed positions.
- `calc_center.py:41` applies a trailing rolling median. Prose describes a moving
  average. The median is not central velocity, but neither smoother is needed
  by the raw adapter. It uses no interpolation or smoothing.
- `modify_csv_detections.py` removes short whole tracks before and after spatial
  masking. Its reduced output also changes the time grid. These operations can
  change query eligibility using future track support. The raw adapter retains
  short tracks and incomplete/absent future labels separately.
- The publisher code specifies a 15 FPS output video and frame-index clock;
  exact video timing and geometry are not independently verified here. Strides
  1 and 12 are structural support probes, not a new shared seconds-level task.

Raw tracker IDs are scoped by recording. Detected tracks are automated labels,
not human gold. The resource archive contains georeferencing material, but its
presence does not validate transforms, point convention, projection or scale.
This intake makes no metric, physical-safety, foundation or true-3D claim.

## Checks Already Run

The pinned publisher demo contains 5,241 raw rows and 58 scoped tracks; 15 tracks
shorter than 30 observations remain. Six actual-data prefix checks pass: changing
or removing future rows does not change current agents, their history or causal
finite-difference velocity. The demo has 4,527 K8/stride1 past-supported queries,
of which 3,694 have all twelve future labels. At stride12 the ten-second demo has
no complete twelve-point future. This is not a finding about the full release.

45 scoped tests pass, including 20 tests for this intake and 25 existing IMPTC
tests. Synthetic regressions cover frame gaps, label availability, duplicate IDs,
unsafe ZIP members, future mutation and ambiguous metadata. They do not prove
full-archive quality or the online provenance of publisher tracking.

The existing SDD protocol, zero-reference harm rule, reserved DroneCrowd role
and deployment stay unchanged. New raw support must first finish checksum and
recording-level audits. No model improvement is claimed.
