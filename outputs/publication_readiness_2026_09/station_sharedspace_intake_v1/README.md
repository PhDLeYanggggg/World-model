# New Data for an 8-Observed, 12-Predicted Study

**TUMDOT-ING is the strongest new structural candidate in this intake.**
This was a fresh release audit, not a model evaluation. No outcome-based
source choice, forecasting, formal test assignment or DroneCrowd
confirmation access. Counts use the published identities and exact sample
indices: a target enters with eight observed samples; its twelve future
labels are counted separately. Query windows can overlap. Even the
non-overlapping counts are not independent sites.

| Publisher source | Audited scope | History-eligible targets | Complete 12-step targets | Non-overlapping complete multi-agent intervals |
|---|---|---:|---:|---:|
| TUMDOT-ING | First day-one file at each of three intersections | 116,829 | 100,736 (86.23%) | 128 |
| Hannover shared space | All three published trajectory CSVs | 23,541 | 19,047 (80.91%) | 149 |
| Czech station trajectories | Three dates at each of eleven cameras | 439,294 | 11,469 (2.61%) | 172 |
| Aschaffenburg VRU | All pedestrian segments in author archive | Not established on a shared clock | Not established | Not established |

These are structural label-presence counts, not reference risk denominators,
human-gold annotations, prediction scores, or independent statistical samples.
Coordinates, images, demographic attributes and forecast errors were not
parsed for evaluation. No scale or timing result changes SDD's existing
pixel/raw-frame contract.

## TUMDOT-ING

The [TUM release](https://www.mos.ed.tum.de/en/vt/forschung/datensaetze/tumdot-ing/)
describes three intersections on the same urban triangle, three consecutive
collection days, and deliberate drone overlaps. The public LRZ share lists
115 individual recording CSVs across the three intersection folders
(44/38/33, 9.64 GB uncompressed file bytes, not counting continuous timelines).
I acquired the earliest listed day-one CSV in each folder via its individual
download endpoint, 166 MB total, and verified the listed byte sizes and
local SHA-256 hashes. The bulk-folder server ignored a range probe; the
individual file route avoids downloading overlapping derived timelines.

All three sampled release clocks match a recording-relative **29.97 Hz**
grid, within 0.000002 s, with no missing timeline indices. This is a check
of these exports, not a claim that every publisher recording shares that
grid. Stride 12 corresponds to about 0.4004 s per sample in these three
files only. Complete 12-step pedestrian windows by intersection are
41,224 / 37,572 / 21,940, and complete non-overlapping multi-agent counts
are 45 / 50 / 33. No exact UUID occurs in more than one sampled file; that
does not rule out shared people with different IDs. Adjacent flights and
continuous derivatives must not be split into independent units. This is
one acquisition campaign and one city, not three external domains.

## Czech Railway Stations

The [paper](https://www.nature.com/articles/s41597-024-04071-9) describes
anonymous stored trajectories but no retained videos and an unknown raw
camera sampling rate. The [Figshare deposit](https://doi.org/10.6084/m9.figshare.27058384)
contains a 3.74 GB ZIP. I inspected the central directory (3,184 camera-day
CSV members) and read only the first, middle and last dates shared by all
11 cameras: 2022-09-05, 2023-01-22 and 2023-05-21. The range reader fetched
66.15 MB in total and verified stable storage version, byte ranges, ZIP CRCs
and each sampled member SHA-256. This is not a full-archive MD5 verification.

The 33 sampled files contain 2,008,347 rows and 220,865 camera-day-local
track IDs. Whole-second release times agree with within-track relative
times. But there are 179,894 gaps above one released second (largest 4,408 s).
No gaps were filled. On the exact one-second export grid, only 2.61% of
eligible histories have complete future labels. Pardubice supplies 164
of 172 non-overlapping complete multi-agent intervals. Numeric IDs recur
between cameras, and eleven viewpoints represent only two stations.
This weak complete-tail support is a reason not to prioritize the source
for the primary forecasting experiment.

## Smaller Sources

The [Hannover release](https://data.uni-hannover.de/en/dataset/trajectories-in-a-shared-space)
contains three trajectory CSVs on its documented half-second grid. It has
382 clip-local pedestrian IDs and good support, but the package alone does
not establish three physical sites; treat it as one group. Separate
demographics and imagery were not opened. No scale conversion was used.

The [Aschaffenburg publisher](https://www.th-ab.de/hochschule/organisation/organisationseinheiten/labor-fuer-kooperative-automatisierte-verkehrssysteme/trajectory-dataset/)
describes four-column CSVs with timestamps; the directly downloaded
pedestrian archive instead contains 1,068 action-organized text segments,
each with a sample-count/50-fps header and five payload columns. A shared
timestamp schema was not verified. Filename prefixes recur; they are not
counted as independent people. Row-ordinal coverage (321 files long enough
for stride 20) does not establish simultaneous multi-agent supervision.

## Next Decision and Reproduction

Prioritize TUMDOT-ING for a prospectively fixed cross-source diagnostic,
with recording-lineage and overlap checks across all selected source files
before scoring. Hannover is a useful separate single-group check. Neither
intake authorizes assigning a new official confirmation split or changing
the original 2% rule. DroneCrowd's 30 confirmation groups remain sealed;
Stage5C and SMC remain disabled.

The output JSONs bind the actual member/file hashes. Local publisher
archives/CSVs stay under ignored `external_data` directories. The station
reader goes directly to the publisher and requests only the required ranges.

```sh
out=outputs/publication_readiness_2026_09/station_sharedspace_intake_v1
.venv-pytorch/bin/python scripts/audit_m3w_station_and_sharedspace.py \
  --cafile /etc/ssl/cert.pem \
  --hannover-archive external_data/HannoverSharedSpace_source_audit/trajectory_data.zip \
  --output "$out/analysis.json"
.venv-pytorch/bin/python scripts/audit_m3w_station_and_sharedspace.py \
  --aschaffenburg-only external_data/AschaffenburgVRU_source_audit/pedestrians.tar.gz \
  --output "$out/aschaffenburg.json"
.venv-pytorch/bin/python scripts/audit_m3w_tumdot_ing_support.py \
  --directory external_data/TUMDOT_ING_source_audit --output "$out/tumdot_ing.json"
```

[Station and Hannover counts](analysis.json) | [Aschaffenburg counts](aschaffenburg.json) |
[TUMDOT-ING counts](tumdot_ing.json)
