# Independent Source Acquisition Record

## Material Passport

Date: 2026-09-22. Primary pages and the official DroneCrowd download folder were
read live. Local TRAF annotations were freshly screened. No new source roles,
forecasting results or independent calibration have been created. This record
distinguishes source-reported inventory from locally verified raw data.

## DroneCrowd: A Concrete, Unadmitted Candidate

The [official repository](https://github.com/VisDrone/DroneCrowd) separates the
sparse ECCV2020 counting challenge from the full release. The full release is
described as 112 clips, 33,600 frames and 70 scenarios. Scenarios have not been
verified as 70 independent physical sites. Do not mix these versions.

The original paper reports head trajectories and an 82/30 sequence division at
different locations. It reports 25 FPS at acquisition, but no annotation-to-video
mapping or camera-motion calibration has been verified locally. Source-reported
FPS does not establish our prediction step duration or a metric coordinate frame.
[Paper, section 3.1](https://arxiv.org/pdf/2105.02440).

The [official Drive folder](https://drive.google.com/drive/folders/1EUKLJ1WmrhWTNGt4wFLyHRfspJAt56WN)
was opened in the browser, not inferred from a search snippet. The visible files
include:

| File | Displayed size | Public file ID |
|---|---:|---|
| annotations.zip | 41.3 MB | 1NeUK0AqgACG1iPiu4rjz3J68Pnsaj5LN |
| README.md | 4 KB | 1H0BpOCa7qqZ6E-SPyQENaR7HMtuw1R5c |
| trainlist.txt | 572 bytes | 1ZnKYzsCZ16RdNnyc2AkHKpmAQtO4AxPq |
| testlist.txt | 208 bytes | 1ZfT8RIcOnz_fiReREA0A89xNB_BO2sZ2 |
| train_data.zip | 7.72 GB | 1piwSmJ5ySlQJsKHXSSG2gicbE86Ib4HV |
| test_data.zip | 2.63 GB | 1C9PGUZ9GPB_NnUBVGbkJQFXhRxD3bfT9 |
| val_data.zip | 107.9 MB | 1hh43TsvHirhvcwQMDxWTtToOjtPYL2yL |

These are visible remote metadata, not downloaded content hashes. No image/video
archive has been downloaded. The annotation preview shows an `annotations`
folder; its member list and actual trajectory support are not yet checked.

The [release README](https://drive.google.com/file/d/1H0BpOCa7qqZ6E-SPyQENaR7HMtuw1R5c/view)
was read in the official viewer. It explicitly states that validation is sampled
from test. It also describes original XML annotation, frame/ID/bounding-box rows,
derived point annotations, and academic/non-commercial usage. This resolves the
previously missing dataset-specific usage description, not all source admission
requirements. The public repository's citation request alone was insufficient.

An earlier [overlap issue](https://github.com/VisDrone/DroneCrowd/issues/11) was a
useful lead, but the release README is the direct support for the overlap policy.
It must not become a claim that the authors' stated train/test locations overlap.
Nor does an annotation-format description establish point-level causal provenance:
XML keyframe/interpolation semantics and the point-conversion rule still need
inspection before strict real-time or past-only-input claims.

An annotation-only download was attempted. Google presented its cannot-virus-scan
warning before download, and the action was not confirmed. A user question asks
whether to proceed with local archive inspection only. Acquisition is pending,
not complete; no files from the archive have been executed. The ~10 GB image
archives are unnecessary for this first schema check.

## Other Existing Candidates

| Source | Established support | Limitation for the next experiment |
|---|---|---|
| DUT | Existing hash-bound local conversion and limited quality audit | Only two stated physical sites; source-use/exposure/role decisions and one duplicate-ID quarantine unresolved |
| TRAF | Fresh audit of all 30 local files | Undocumented types/duplicate IDs, box convention and camera/site mapping unresolved; not automatically top-down pedestrian support |
| inD | Local README/reference only in the inspected dataset folder | Raw trajectories not present there; obtain author-authorized academic access rather than redistributing a toolkit pointer |
| Edinburgh | Local documentation/download helper | One foyer is not many independent sites; variable sampling and raw-versus-spline provenance require care |

The inD access route is the [author dataset site](https://www.ind-dataset.com/).
This is an acquisition prerequisite, not a claim that an account/application was
created or approved. The local file inventory is scoped; it is not proof of
absence from all user drives or CREATE.

## Priority and Boundaries

1. Resolve the concrete DroneCrowd annotation download prompt, inspect archive
   members without executing included code, record hashes and raw schema.
2. Establish identity continuity, interpolation/visibility semantics, physical
   scene grouping, camera motion and possible overlap with historical sources.
3. Propose a role design on eligible scene groups for author approval. Do not use
   the supplied validation/test folders as independent roles; do not pick a split
   or stride by looking at forecast quality.

All new-source forecast/calibration/confirmation runs remain `not_run`. TRAF
availability is `fresh_run`; its exact recount is `cached_verified`. This source
work does not alter the 4.10% development ADE result or its failed safety gate.
No metric, seconds-level, true-3D, foundation or deployment claim is added.
