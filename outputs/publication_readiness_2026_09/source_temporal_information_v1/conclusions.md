# Source Temporal Information Audit

## Material Passport

`fresh_run`: post-hoc past-only measurements on the same15,430queries and123,440
historical query-frame keys. `cached_verified`: existing pixel/box/embedding
arrays and their provenance. No training, new forecasts, future-label grouping
or main/outer scoring. The inherited identity loader reads broader arrays,
including labels for provenance; labels are not used in these measurements.

## Findings

- All history keys match the query's agent and expected past frames, stride12rawframes.
- No window repeats one image across all eight steps. Only two of108,010adjacent query-frame pairs have exactly identical RGB arrays.
- All windows have eight supported images; mean retained pixel coverage97.78%.
- Median agent annotation box spans9.34by11.86output pixels, occupying10.94%of the32x32crop. These boxes are not segmentation or human visibility labels.
- Mean adjacent RGB absolute change is2.365byte levels overall,3.224inside the common box region and2.248outside. These are descriptive pixel changes, not optical flow, intent or future predictability.
- Within-window temporal energy is2.5715%of frozen embedding energy on average; median2.0464%,10th/90thpercentiles0.8514%/5.0490%. No window has identical embeddings for all steps.

The cache has not silently replaced history with repeated/current frames or
empty image tokens. Substantial shared appearance remains in the frozen features.
That shared component is not proven to be background, and small temporal variance
does not prove that useful motion is absent. This audit cannot establish the
reason for poor forecast transfer by itself.

## Next Controlled Repair

The next fixed comparison removes per-query shared appearance, with and without
past-only variance normalization. Keep population, target, loss, architecture,
seeds and sampling fixed; compare actual trajectories to the existing controls.
Do not infer success from feature variability, an oracle or lower distribution
distance. Current native-detail/flow negatives on the separate main-fit cohort
remain in the record; no claim that resolution alone repairs source prediction.

Five focused helper tests pass. Machine-readable per-site summaries and private
row-archive hash are in[audit.json](audit.json). Offline supplied annotation
history, not sensor-as-of. No new deployment, Stage5C or SMC.
