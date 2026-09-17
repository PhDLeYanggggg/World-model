# Local SDD Media Inventory, Not Training Admission

Follow-up: the [completed full decoder/correspondence audit](../sdd_media_alignment/conclusions.md)
supersedes the header-only readiness assessment below. It finds video resizing
and misnumbered Nexus media; explicit diagnostic links are now available. The
original header inventory is preserved unchanged and is not training admission.

`fresh_run`: read local video headers and SHA-256 hashes for 60 files under
`external_data/StanfordDroneDataset/video`, covering eight folder-level scenes.
All 60 headers are readable; total size is 1,195,953,035 bytes. The inventory
opens neither annotation labels nor any sealed evaluation role.

Every stream header reports H.264 and average rate 2997/100. This is container
metadata, not proof of annotation timing, physical elapsed time, complete video
integrity or original dataset identity. Header frame counts range from431 to
14,558; nine files have fewer than1,000 header frames:

| Scene | Videos | Header frame counts |
| --- | --- | --- |
| deathCircle | video2, video4 | 431, 452 |
| hyang | video7, video8, video9 | 574, 574, 574 |
| quad | video0, video1, video2, video3 | 509, 509, 509, 509 |

These short files require a source-completeness/frame-alignment audit before
being used. The inventory does not label them corrupt or complete. No full-frame
decode, new feature/latent extraction, sampling rule or training has been run.

## Admission Boundary

The active fit roster is ETH, Hotel and grouped Zara, not SDD. Adding SDD as
auxiliary training/pretraining is a separate prospective source-role decision.
The existing eight-to-twelve task, approved primary metric, development,
calibration and confirmation boundaries must remain unchanged unless explicitly
revised. Existing SDD use and past teacher exposure must remain in the lineage
record; old scores cannot become independent confirmation by renaming them.

If admitted, first verify source annotations against decoded frame indices,
physical-scene grouping, duplication, observation availability and stable
sampling. Source images/videos and all large caches stay local. More files
alone do not demonstrate that stationary-start support increases or forecasts
improve. Current status: **media available; training admission pending; not_run**.

Reproduce with `scripts/inventory_m3w_sdd_video_headers.py`.
[Per-file metadata and hashes](headers.json). No metric/seconds, Stage5C/SMC,
world-model performance or submission-readiness claim is made.
