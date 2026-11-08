# ACME GMU: Causal Association Repair and First Motion Comparison

**2026-11-08.** A fresh one-camera development diagnostic. This clip is
from a different institution than the Michigan probe, but it is not
an independently held-out confirmation result.

## Camera and Clock

I downloaded the 1,059,265,728-byte [GMU `engr_elev_1` AVI](https://huggingface.co/datasets/acmesocnavdataset/ACME-dataset/blob/a0f117c11a1aee25413da95cf8612f5f8209c171/BEV/BEV_videos/gmu/engr_elev_1.avi)
and verified its publisher SHA-256. The video has 29,451 frames at 30 FPS.
Its previously audited trajectory file is on a 10 Hz annotation grid;
the last usable released tick is 9,816. The [publisher tracking code](https://github.com/social-navigation-group/sbpd-dataset-pipeline/blob/8ec39f0adce6bc24ea8ac805915b2b966cd93fcf/anonymization/anonymize_and_track.py)
records a box's bottom centre, using every third camera frame starting at
the first. I used the resulting mapping `camera_index = 3 * annotation_tick`
without searching offsets. The 96 sampled decoder timestamps matched the
30 Hz camera grid. This is a release-specific observation, not a metric
calibration or a timing result for SDD.

The publisher script also contains interpolation and a smoothing pass,
and the release contains manual track edits. The exact upstream settings
for this clip are not assumed. I used the released trajectories as
evaluation references only, not as sensor inputs or source of velocity.

## Original and Repaired Association

Before decoding, I registered 12 disjoint 8-observation / 12-prediction
windows, sampled at fixed quantiles of the previous explicit-clock
support schedule. Past reference presence informs which time intervals
are probed; future completeness does not select them. This is a
support-enriched diagnostic, not random all-camera coverage. The
detector produces all its rows from images only. A motion history is
scorable only if its causal track corresponds to one reference identity
in all eight past frames. The predeclared diagnostic floor was eight
stable histories from at least three windows.

| Association | Stable histories | Windows with a stable history | Engineering floor | Future scoring |
|---|---:|---:|---|---|
| Original last-box tracker | 7 | 6 | Fail | Not run |
| Causal two-point transport, same detections | 8 | 6 | Pass | 5 complete targets in 4 windows |

The original failure remains on record. The follow-up forecasts a
track's next anchor from its two most recent matched boxes and compares
that transported box to the current detections; a last-box match is
still available. It changes only association. Detections, sampled
windows, one-to-one reference match, support floor and forecast bank
remain fixed. This follow-up was motivated by the observed support
failure, so it is explicitly **development**, not fresh independent
validation. Ignoring online IDs, 17 reference histories had a matched
point at every sampled frame. That diagnostic measures how often points
were available but not linked; it is not a realizable tracker result.

## Motion Baseline Readout

Both predictors use the same eight detector-derived ground anchors.
CV uses the last difference; OLS4 fits the last four positions with
the prediction anchored at the last observation. No learned neural
forecast was evaluated. Future reference positions are fetched only
after the stable-history check.

| Same 5 complete targets, equal weight over 4 windows | ADE (image px) | FDE (image px) |
|---|---:|---:|
| Causal CV | 79.44 | 125.05 |
| Anchored OLS4 | 63.19 | 124.56 |

Three of eight stable histories lack a full future. The roughly 20%
ADE difference on the five complete cases does not establish better
population performance; endpoint error is nearly unchanged. This
sample cannot support a confidence interval across cameras or the
original 2% risk decision. No new site was assigned to confirmation,
DroneCrowd's 30 groups remain sealed, and no model was promoted.

**Next:** Retain causal association as a camera-input candidate. Test
the frozen pipeline on further nonoverlapping cameras before scaling
the downstream experiment. The highest-priority scientific gap is
still independently verified support and risk, not adding another
forecast architecture.

## Replay

Place the pinned AVI in ignored `external_data/ACME_source_audit/` as
`gmu_engr_elev_1.avi`, with `gmu.toml` from the previous source audit.
The pipeline uses the frozen Waterloo COCO detector, native CPU with
four threads, window checkpoints, and heartbeat. To resume, use the same
registration with `produce --resume`. No raw inputs or caches are in Git.

```sh
.venv-pytorch/bin/python scripts/run_m3w_acme_gmu_camera.py register
.venv-pytorch/bin/python scripts/run_m3w_acme_gmu_camera.py produce --resume
.venv-pytorch/bin/python scripts/run_m3w_acme_gmu_camera.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_acme_gmu_reassociate.py
```

[Original registration](registration.json) | [Original support](analysis.json) |
[Reassociation registration](reassociation_registration.json) |
[Reassociation and baseline scores](reassociation_analysis.json).
Two runs of reassociation and scoring agreed exactly on the fixed detections.
48 focused tests passed, including synthetic future-mutation, frame-clock
and association-prefix checks.
