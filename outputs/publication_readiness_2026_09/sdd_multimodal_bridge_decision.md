# Registered SDD Past-Image / Geometry Join

This prepares an actual multimodal data path while the separate auxiliary
source-role and sampling decision is pending. It does not admit training, change
the ETH/UCY primary or open sealed labels. The preceding geometry bridge is
complete; the older image reader only covers each video's first 64 frames.

Use exactly the geometry bridge's original 40 training recordings and its fixed
64 evenly spaced past-index queries per raw-frame stride (1 and 12), or all queries
if fewer. Query selection uses past support only. Do not read future labels here.
Do not select scenes, examples or image processing by forecasting error.

Collect the eight past frame/agent requests for each query and deduplicate them.
Sequential decode preserves frame-index identity; only requested RGB frames are
converted and cropped. Use the already verified annotation-to-video resize map,
96-to-32 pixel pooling and the existing inferred border-connected dark mask.
Keep original observed and retained RGB, geometric and retained support counts,
source boxes, generated/occluded flags and explicit query identities. No missing
cache entry may silently become a zero-image observation. Zero geometric/retained
support is a different, explicitly recorded condition.

Join geometry and images on recording, agent, frame and stride. Verify original
annotation boxes/flags, the exact past grid, masks, finite CNN features and a
zero-initialized model's exact CV forward. This is an interface check with zero
optimizer updates, not non-collapse, predictive lift or a training smoke test.
Retain the offline annotated-history restriction and retrospective interpolation
limitation. Pixels are not metric coordinates or evidence of physical duration.

Pilot bookstore/video0, then all 40 if the observed resource use is feasible.
Native arm64, one process, four decoder/Torch threads, Torch interop1. Save each
record atomically and preserve completed-record receipts on resume. The cache
contains sampled image requests, not all source frames or all medium episodes.
No images, per-row caches or source data enter Git.

Independently re-decode first/middle/last requested frames per recording and
compare exact image/crop bytes. Inspect input-selected contact sheets from
bookstore/video0, deathCircle/video2 and hyang/video7; this limited self-review
does not certify all person annotations or create human-gold labels. Report
missing support rather than discard it. Then verify a completed no-decode resume
by hashing cache arrays, metadata, query lists, receipts and main report.

No fit, threshold tuning, new primary, source-role admission, Stage5C or SMC.
After the pending scientific choice, register auxiliary-versus-control training
with fixed budgets and seeds. Do not call a successful data join a model result.
