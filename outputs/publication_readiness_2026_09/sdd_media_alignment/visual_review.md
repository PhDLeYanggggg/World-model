# Limited Visual Review

Reviewer: assistant self-audit, 2026-09-17. This is not human-gold labeling,
independent inter-rater review, or certification of every box/frame. The JSON
generators correctly record `not_reviewed` at generation time; this separate
note records the later displayed-image inspection without rewriting those receipts.

Six private contact sheets, each with five fixed frame indices, were displayed:

| Source annotation | Displayed media | Observation |
| --- | --- | --- |
| bookstore/video0 | bookstore/video0 | Small resize; boxes broadly coincide with visible actors, but early black/warped border is present. |
| deathCircle/video2 | deathCircle/video2 | Unscaled boxes land on background; mapped boxes broadly follow visible people/vehicles at all five displayed indices. |
| hyang/video7 | hyang/video7 | Large resize correction places boxes near the depicted moving agents rather than unrelated path/grass regions. |
| nexus/video10 | nexus/video2 | Repaired source link and resize show correspondence to actors along the path; unscaled coordinates fail. |
| nexus/video3 | nexus/video5 | Later mapped boxes correspond to the group on the path; frame0has extensive black padding, so image extent is not visibility. |
| nexus/video6 | nexus/video8 | Five-frame repaired-link sample shows boxes near depicted agents instead of background; all requested indices are available. |

The observations support the diagnostic resize/link hypothesis on these samples.
They do not prove the exact original compression process, precise bounding-box
quality, all-source temporal alignment, pose identity, physical time or metric
geometry. Some objects are tiny or occluded, limiting inspection at contact-sheet
resolution. Do not use this note as a forecast validation result or training
admission decision.

Evidence filenames and hashes are preserved in `resize_mapping.json` and
`nexus_identity.json`. Native images and contact sheets stay in the ignored local
`data/stage_cvpr2027_experiments/` directories. No third-party image is published.
