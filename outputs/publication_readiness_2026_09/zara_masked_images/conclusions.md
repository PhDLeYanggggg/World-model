# Partial-Image Inputs Now Retain Entering Agents

## Material Passport

- Scope: local fit-source input construction and deterministic replay.
- Sources: hash-bound Zara01/02 and the preceding fixed past-media controls.
- Evidence: fresh conversion, completed-cache verification, independent rebuild.
- Excluded claims: new forecasting training, prediction improvement, independent
  scenes, scientific cohort admission, strict online observations or physical safety.

## What Changed

The previous diagnostic required a whole 96x96 crop to lie in the video frame.
That discarded the usable part of images near the boundary. The new reader
keeps the point-centered location, averages only available pixels in each 3x3
block, and returns 32x32 RGB plus a separate coverage count. It does not shift
the crop inward, synthesize outside pixels or treat zero padding as observed
black. Fractional coverage is available to downstream models, not a hidden
preprocessing choice.

| Recording | Source rows | Past8 windows retained | Full / partial crop rows | Windows containing partial crops |
| --- | ---: | ---: | ---: | ---: |
| Zara01 | 5,024 | 3,988 | 4,403 / 621 | 621 |
| Zara02 | 9,537 | 8,110 | 8,196 / 1,341 | 1,314 |

All source rows decoded; every retained window has at least some observed pixels
in each of its eight frames. No future-label availability is used to retain a
history. This gives 12,098 windows, including 1,935 with a partial crop, rather
than dropping boundary observations. It does not supply 12,098 independent
samples or additional sites: these remain two recordings at one physical scene.

Each source crop is stored once. Windows contain row indices, not eight repeated
images. Arrays total60,977,761 bytes (about61MB decimal), use read-only memory
mapping, and have individual SHA256 hashes. Source-control provenance is stored
separately from the whitelisted input fields. No future target columns, central
velocity or test goals enter this store.

## Same-Query Check

The prior 24 fixed inspection histories and their exact past frame indices were
reused without new outcome selection. All159 originally complete crops retain
full support. All33 formerly rejected whole crops now retain actual partial
observations: seven Zara01 crops have at least56.25% coverage;26 Zara02 crops
have at least52.08%. The mean recovered coverage is75.60%/77.16% respectively.
These are visibility numbers, not prediction gains.

Six private comparison sheets were generated; two pages containing boundary
examples (Zara01 page2 and Zara02 page0) were inspected. Checkerboard regions
denote unobserved pixels only in the inspection renderer, not generated content
in the data cache. The sheets do not establish identity, body pose or gold labels.
The new area-based reduction is not claimed pixel-identical to the old image
resizing operation; a future forecast comparison must explicitly fix preprocessing
and give its trajectory-only control the same cohort.

## Causality and Admission

The loader requires explicit `offline_annotation_diagnostic` or
`control_as_of_query_diagnostic` mode. It rejects `supervised_training` and
`official_eval` roles: the scientific observation-definition decision remains
pending. The source annotations still contain retrospective interpolation.

Actual traversal of the stricter diagnostic accepts111/186 Zara01/02 histories
and rejects3,877/7,924 with later controls. Rejected rows remain in the store;
there is no silent population filter. The accepted subset is not evidence of
online human annotation or identity availability. Image masks repair border
support, **not upstream temporal provenance**.

The existing8/12 parent split, primary normalized ADE, labels and performance
results were not changed. The earlier prospective primary-metric question also
remains open. No development, calibration or confirmation data was opened.

## Execution and Verification

The fresh full fit-source conversion took9.74s locally. A separate full rebuild
took9.68s and produced byte-identical18 array files and identical reported
statistics. Completed-cache `--resume` took0.80s, replayed every public input
window and performed zero new conversions. All processes exited successfully.
These timings describe input construction, not neural training or an epoch.

18 focused tests passed: partial/full/missing pixel support, black-versus-missing
distinction, unchanged past inputs after unrelated future-row corruption,
future/mixed-agent index rejection, hash corruption rejection, source-origin
mapping, role enforcement and explicit later-control failure. The unrelated
full legacy test suite was not rerun.

## Next Research Step

The implemented input repair is ready for a registered fit-only predictive
comparison after the observation definition is confirmed. That comparison needs
identical moving-history support, explicit masks, trajectory-only/image controls
and complete negative-result reporting. Do not infer benefit from recovered
pixel area, select a new threshold on old held outcomes, or claim that adding
Zara01/02 increases the independent-scene count by two.

This turn is concrete engineering progress, not a new deployable method or
submission-ready result. No new neural training, Stage5C or SMC. Original videos,
image/row caches and rendered sheets stay outside Git.
