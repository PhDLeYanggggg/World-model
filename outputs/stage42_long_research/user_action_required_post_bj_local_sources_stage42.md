# Stage42-BK User Action Required

- source: `fresh_post_bj_local_source_verification`

## ETH_UCY

- action: confirm license/terms for local ETH-Person XML files before using them as ETH_UCY t100 source-CV repair candidates.
- local_candidates: `['ETH-Person/data/bahnhof_assc_gt.xml', 'ETH-Person/data/jelmoli_assc_gt.xml', 'ETH-Person/data/seq0_assc_gt.xml', 'ETH-Person/data/seq0_assc_gt-interp.xml', 'ETH-Person/data/sunnyday_assc_gt.xml']`
- next_step_after_confirmation: convert XML to Stage42 external source rows, run no-leakage, then train-only source-CV.

## TrajNet

- action: provide official longer raw TrajNet++ / original trajectory sources with tracks longer than 100 raw frames, or confirm that only 8/20-step snippets are legally available.
- reason: current local TrajNet files parse as fixed short snippets and cannot repair raw-frame t100.

## Non-Claims

- Do not count ETH-Person XML candidates as converted/evaluated until conversion and no-leakage/source-CV run.
- Do not count TrajNet snippet files as t100 support.
- Do not report dataset-local/raw-frame horizons as metric or seconds-level.
