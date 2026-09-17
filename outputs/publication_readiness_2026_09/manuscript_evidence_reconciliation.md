# Manuscript Evidence Reconciliation

2026-09-18. Scope: revise the existing English working manuscript against stored
aggregate evidence. This is writing and consistency repair, not a fresh training
run, independent review, new bootstrap or successful research hypothesis.

## Changes

- The abstract now includes the one-direction motion-to-start result. That
  result was already in the body; it was not a newly discovered experiment.
- A source-role paragraph separates approved completed development, later
  fit-only repairs, diagnostic SDD inputs and unresolved independent confirmation.
- The SDD geometry/image bridge is described with its actual coverage and limits.
  The sparse image cache is not presented as covering the whole geometry index.
- A claim table distinguishes oracle ceilings, consistency, probability ranking,
  image access and forecasting improvement. None substitutes for the others.
- The independent-protocol draft no longer says the old development run is still
  running. Its historical status is retained without blocking already-approved
  work or implicitly admitting a new auxiliary experiment.

## Evidence Reused

The following aggregate files were read and hashed in this pass. The original
experiments were not rerun. Hash agreement establishes artifact identity, not
the correctness of every upstream raw-data or scientific claim.

| Artifact | SHA256 |
| --- | --- |
| `motion_start_information/report.json` | `c1d377b37aa85931078e8439ddf6ccfcc0875a52a0beb0d27ff80dbb74d335b6` |
| `sdd_step_bridge/report.json` | `1970c2a7feacf3f6bfa41ebed507374e05edf145ab9c47dc4cb4c51c1f22faa8` |
| `sdd_multimodal_bridge/report.json` | `0baf077c4c1b67ffa7d645fa8cdb4757c44d86a797457473625e310216c7f7d8` |
| `sdd_multimodal_bridge/input_quality.json` | `27fadfc215cd37cccfc8604b7e8cf5947d75511b22d75952754fdd7a85eba95c` |

Recomputed sums from per-record receipts: 40 original train recordings,
8,005,367 annotation rows, 3,045,974 stride-1 and 229,333 stride-12 past queries;
5,074 image/geometry joins, 39,144 crop requests, 38,449 beyond frame 63,
3,721 geometrically partial crops, 84 inferred-border intersections. Projected
short boxes below eight pixels are 15,145/39,144 = 38.69048%. These are diagnostic
requests and overlapping windows, not independent samples or quality exclusions.

ExtraTrees magnitude/directed Hotel-to-ETH AUROC = 0.81048/0.82152, absolute
Brier lift = +0.08030/+0.07870. Reverse AUROC = 0.50038/0.51181, absolute Brier
lift = -0.00484/-0.00101. These probability metrics are not trajectory gains.
All 54 local link occurrences in the revised manuscript and protocol draft resolve.
No new external-paper review or full test-suite run is claimed for prose edits.

## Remaining Priorities

1. A useful transferable predictor is still missing under the repaired protocol.
   Another threshold sweep over the same weak candidate pool is not justified.
2. The proposed next comparison uses a separate SDD auxiliary arm versus a
   matched no-auxiliary control. Original train-40, raw stride 12, eight observed
   and twelve predicted points is the recommendation, not an approved change.
   At this stride the endpoint is +144 raw frames, not twelve raw frames or a
   verified number of seconds. Main ETH/UCY rules remain unchanged.
3. Independent confirmation and informative scene-level risk calibration remain
   unresolved. Additional overlapping windows do not supply independent sites.
4. The manuscript is an evidence-bearing draft, not a finished submission.
   Main-method support, independent results and the final publication package
   remain required by the full goal.

No new model, input admission, metric, threshold, deployment or sealed-role access.
No CREATE query/job this pass; the previously documented access condition was
not reverified as live remote state. Stage5C and SMC remain disabled.
