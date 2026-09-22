# TRAF Raw Intake Audit

## Material Passport

Fresh raw-file structural screening. No center/footpoint conversion, model fitting,
forecast scoring, split assignment, goal construction, or metric/seconds claim.
These overlapping availability counts are not independent samples or an approved
8-to-12 evaluation protocol. Stride 12 covers all phases for inventory only.

Files: 30; filename families: 22.
Independent physical sites: unknown. File families must not become scene IDs.
Quarantined recordings: 27; admissions: 0.

## Geometry Conflict

Of 592,266 parsed boxes, 587,251 violate
the local README xyxy ordering. Positive third/fourth fields are compatible with
width/height but do not establish that convention. Conversion remains refused.
The local dataset README states 20 FPS; the toolkit table states 10 FPS.
Neither is validated against this release/video metadata.

## Availability Before Recording Quarantine

| Class | Tracks | Points | 8-to-12 stride 1 | 8-to-12 stride 12 |
|---|---:|---:|---:|---:|
| bike | 109 | 22326 | 20185 | 6018 |
| bus | 6 | 1696 | 1563 | 547 |
| car | 143 | 54046 | 51208 | 31334 |
| chair | 25 | 2666 | 2230 | 769 |
| cycle | 3 | 601 | 544 | 199 |
| desk | 63 | 3900 | 2872 | 92 |
| human | 46 | 3099 | 2254 | 327 |
| man | 24 | 3564 | 3127 | 1340 |
| object | 67 | 7877 | 6683 | 1934 |
| ped | 159 | 65858 | 62646 | 39004 |
| rick | 99 | 37399 | 35528 | 17831 |
| rickshaw | 31 | 7901 | 7187 | 2377 |
| sccoter | 5 | 1406 | 1311 | 635 |
| scoote | 1 | 197 | 178 | 0 |
| scooter | 290 | 61941 | 56455 | 14192 |
| truck | 4 | 1632 | 1556 | 720 |
| untyped_id | 1638 | 316157 | 283840 | 94086 |

## Per Recording

| Recording | Parsed frames | Tracks | Pedestrian tracks | Quality |
|---|---:|---:|---:|---|
| TRAF11 | 1024 | 89 | 11 | quarantined |
| TRAF12 | 929 | 153 | 48 | quarantined |
| TRAF13 | 872 | 42 | 0 | quarantined |
| TRAF15 | 883 | 51 | 0 | quarantined |
| TRAF17 | 1828 | 173 | 0 | quarantined |
| TRAF18 | 1004 | 46 | 0 | quarantined |
| TRAF21 | 943 | 28 | 0 | quarantined |
| TRAF22 | 1376 | 62 | 0 | quarantined |
| TRAF23 | 599 | 11 | 0 | quarantined |
| TRAF25 | 1237 | 61 | 0 | quarantined |
| TRAF26 | 805 | 80 | 0 | quarantined |
| TRAF27 | 3055 | 105 | 0 | quarantined |
| TRAF28 | 3100 | 93 | 0 | quarantined |
| TRAF29 | 3098 | 92 | 0 | quarantined |
| TRAF30 | 3100 | 113 | 0 | quarantined |
| TRAF31 | 3091 | 96 | 0 | quarantined |
| TRAF32 | 3019 | 104 | 0 | quarantined |
| TRAF37 | 3100 | 95 | 7 | structure_screen_only |
| TRAF38 | 3100 | 170 | 40 | structure_screen_only |
| TRAF46 | 3100 | 146 | 20 | structure_screen_only |
| TRAF47 | 3100 | 200 | 33 | quarantined |
| TRAF53_1 | 1449 | 70 | 0 | quarantined |
| TRAF53_2 | 1450 | 78 | 0 | quarantined |
| TRAF53_3 | 1450 | 51 | 0 | quarantined |
| TRAF53_4 | 1410 | 40 | 0 | quarantined |
| TRAF53_5 | 1450 | 60 | 0 | quarantined |
| TRAF53_6 | 1450 | 66 | 0 | quarantined |
| TRAF53_7 | 1191 | 72 | 0 | quarantined |
| TRAF53_8 | 1395 | 119 | 0 | quarantined |
| TRAF53_9 | 1450 | 147 | 0 | quarantined |

## Remaining Admission Requirements

Resolve the annotation convention, source-specific terms, physical-site grouping,
camera motion/perspective and historical exposure before any role assignment.
Resolve quarantined IDs/rows without silent deduplication. Exact-box matches are
a limited duplicate screen, not a complete recording-overlap audit.
Source field parsing does not establish causal availability of annotation geometry.
No traffic-domain result is substituted for pedestrian top-down generalization.
