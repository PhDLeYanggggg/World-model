# IMPTC Sample Intake: Real Conversion, No Forecast Readout

## Result

`fresh_run`: acquired the publisher's complete **sample package**, not the full
IMPTC dataset. Parsed all four included sequences into private, source-aligned
row arrays: **142,361 observations, 371 tracks, one physical intersection**.
The archive is 347,692,691 bytes and matches the publisher MD5. Our SHA256 is
`b815d1a6652e1ba496034746d52dab17e4f80ed3b5dd6055579eb4a9f76dc44e`.
Dataset rights are CC-BY-NC-4.0 in the
[publisher record](https://zenodo.org/records/14811016); its separate code license
is not substituted. Raw data and the 6,897,122-byte derived cache stay local.

| Sequence | Observations | Tracks | Person tracks | Person observations |
|---|---:|---:|---:|---:|
| 0000_20230322_081506 | 20,573 | 63 | 3 | 3,646 |
| 0001_20230523_105516 | 43,104 | 102 | 22 | 16,860 |
| 0002_20230523_111106 | 41,843 | 116 | 22 | 16,230 |
| 0003_20230523_111514 | 36,841 | 90 | 14 | 19,638 |
| Total | 142,361 | 371 | 61 | 56,374 |

The remaining tracks are 282 cars, eight trucks, 12 bicycles, three scooters
and five motorcycles. These are publisher whole-track audit classes, not
inference inputs or an eligibility filter. No prediction accuracy was computed.

## Obs8/Pred12 Support

| Population / source-index stride | Past-eligible windows | Full future12 | Partial future12 | No future12 |
|---|---:|---:|---:|---:|
| All agents / 1 | 139,764 | 135,312 | 4,081 | 371 |
| Person-labelled audit slice / 1 | 55,947 | 55,215 | 671 | 61 |
| All agents / 10 | 116,538 | 85,648 | 27,327 | 3,563 |
| Person-labelled audit slice / 10 | 52,104 | 44,784 | 6,710 | 610 |

Queries are retained based on past support only. The table is structural
availability, **not independent sample size**, forecast evaluation or a new
official protocol. The full audit also covers K16/32/64. Stride10 is an intake
diagnostic, not a replacement for SDD native8/12 stride12 and not evidence of
matched physical time. The entire archive has 1,538 entries; only 375 trajectory
and master-clock JSON members were parsed. Other context and bundled media were
not used for model inputs, viewed or extracted.

## Repairs and Verification

The released track dictionaries use ordinal keys, although the documentation
illustrates timestamp keys. The reader joins each record's `ts` to the master
clock instead of treating dictionary order as time. It also handles the released
`lenght` spelling solely in overview validation. Every observation joins exactly;
all 371 tracks have continuous master-index support. The four master-clock
intervals do not overlap. None of that establishes independent physical sites.

One sequence has timestamp increments of 40,000; the other three contain small
increments of 39,965/39,978/39,979/40,009. Consequently the cache preserves exact
integer timestamps and source-master indices; it does not silently replace
them with a uniform physical clock. Publisher meter/UTC declarations remain
distinct from our unperformed geometric and sensor-clock verification.

`cached_verified`: reacquisition check, complete second source parse, exact
array/alignment equality and exact aggregate replay all pass. A separate
set-based support calculation also checks all 142,361 raw-to-cache rows and
agrees with every K/stride count. It is a second implementation checked by the
same agent, **not external review**. Actual-data future mutation/truncation
checks pass in 136 sequence/query comparisons. **59 scoped tests pass**, including
25 new IMPTC tests; the unchanged full legacy suite was not rerun.

## Why Admission Is Still Closed

The [source paper, sections III-A to III-C](https://arxiv.org/html/2307.06165v1)
describes stereo/LiDAR coordinates, calibration and IMM tracking. Local prefix
invariance does not prove the temporal provenance of exported coordinates,
occlusion bridging or overview classes. The input reader therefore excludes
publisher velocity, overview class, total track length, end time and future
availability. It derives backward differences in coordinate units per master
index. Neither filtered coordinates nor silver/automatic tracking are called
human gold or raw online observations.

The four sample recordings add at most **one** physical-site group. They do not
repair the independent multi-site calibration deficit. The paper also relates
this installation to earlier VRU work. OpenTraj VRU files are already local;
although the checked Stage31/35 converters do not select them and no local
result match was found, shared-site lineage still needs an explicit exclusion
across the full producer chain. Current remote exposure remains unknown because
the latest saved CREATE refresh could not authenticate. No new SSH attempt or
remote action was needed for this local audit.

`not_run`: model/baseline errors, training, calibration, confirmation, full-IMPTC
acquisition, geometric recalibration and proof of online source causality.
There is no model improvement, policy promotion or submission-readiness claim.
DroneCrowd stays closed. Stage5C and SMC remain off. M3W is still a 2.5D trajectory
world-state research system, not true3D or a foundation model.

## Next Admission Test

Retain IMPTC as an **unassigned source-audit candidate**. Before predictive use,
close the source/related-VRU exposure audit and freeze a source-specific
observation and role contract. More recordings of this intersection cannot
replace additional independent sites. Do not tune the current controller on this
sample while calling it future calibration data. The delegation of routine
audits removes a user-approval bottleneck, not these scientific conditions.

The earlier tentative TGSIM route was rejected after locating prior conversion
and benchmark records. AerialMPT also has prior training/readout exposure.
Re-downloading either would not create independent evidence.

Evidence: [scope](scope.md), [source manifest](source_manifest.json),
[analysis](analysis.json), [exact replay](verification.json),
[separate arithmetic and class support](separate_checks.json),
[local operation guide](operation_zh.md).
