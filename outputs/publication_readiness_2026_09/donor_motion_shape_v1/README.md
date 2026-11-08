# Can Motion From Another Site Improve the Same Edit?

I kept the original intervention decisions and tested a different trajectory
shape on the same recipient, query, and maximum displacement from constant
velocity (CV). I added a trajectory based on the eight closest past-motion
windows from the other six OpenDD sites. I also tested a shuffled control
with the same histories and neighbor indices but permuted future offsets
within each donor site. This tests the link between a past motion pattern
and its continuation, rather than giving the new method more edits.

## What changed

I aligned each history with its causal heading and scaled by past RMS step
length. The 8/12 library contains only complete donor futures. Each proposal
is the mean donor deviation from CV, mapped back to the target motion frame
and matched to the original peak deviation. The prefix variants add one
alternative at a time to the original six shapes, using forecast errors on
the observed suffixes after steps 4, 5 and 6. Their donor targets are the
corresponding donor history suffixes. No held-site target enters a library,
choice, fit, or threshold.

## Matched comparison

All arms make exactly 1,457 edits across 606 recording-seed views
(202 recordings, three frozen controller seeds). Each new candidate is
deterministic; the seeds vary the frozen recipient and count decisions.
Joint pass means the original recording risk and support rules pass and
conservative utility is positive against both CV and the four-observation
linear fit (OLS4). Undefined risk stays undefined.

| Shape at the same edit | Risk passes | Joint passes | Conservative gain vs original, pp [95% CI] |
| --- | ---: | ---: | ---: |
| Original prefix choice | 112 | 77 | reference |
| Forced matched donor | 129 | 89 | -0.059 [-0.119, -0.003] |
| Forced shuffled donor | 118 | 78 | -0.140 [-0.299, -0.038] |
| Prefix choice with matched donor | 124 | 90 | -0.023 [-0.068, +0.006] |
| Prefix choice with shuffled donor | 108 | 73 | -0.086 [-0.264, +0.011] |

The direct paired lower bound charges missing steps the distance between
the two predictions. Gains are divided by observed CV error mass within
site. The 3,000 bootstrap draws resample the seven sites after averaging
the seeds. The interval is descriptive: these are exposed development sites
with very uneven sample sizes, not independent confirmation data.

Matching beats the fixed shuffled control when forced, +0.074 pp
[+0.001, +0.192], but loses to the original. With prefix choice, matching
beats shuffling by +0.062 pp [-0.023, +0.200], not reliably. The prefix
choice takes the matched shape 764 times and the shuffled shape 473 times.
Eight of the 13 extra joint passes for the matched prefix arm come from
one site (rdb1); the direct site gain there is -0.0004 pp. The worst site
is rdb2 at -0.151 pp, with 50 rows. All arms have 334 undefined-risk views.

## Is there a better candidate in hindsight?

In the same 105 short-recording views (35 recordings), the one-edit oracle
qualifies 30 with the original menu, 42 after adding the matched donor,
and 39 after adding the shuffled donor. The
[prior geometric upper bound](../sustained_turn_v2/README.md) is 57. The
extra capacity is real for these observed evaluation bounds, but hindsight
cannot choose an edit at inference. The small gap between matched and
shuffled additions also means extra menu diversity explains part of the gain.

**Decision:** keep the donor trajectories as a development candidate bank;
do not promote the prefix chooser or reopen confirmation. More recordings
cross the joint threshold, but the matched arm has no reliable overall
trajectory advantage and does not resolve undefined risk. A next controlled
test should learn which added shapes help without increasing harm, validating
that choice on donor-only inner folds before any fresh calibration. Eight
nearest windows can include overlapping observations of the same donor
track; the retrieval library does not guarantee eight independent examples.

## Reproduction and scope

- `fresh_run`: seven cross-site donor libraries per prefix, matched and shuffled
  forecasts, recording-risk evaluation, one-edit hindsight enumeration, and
  site bootstrap. No new neural fit.
- `cached_verified`: frozen OpenDD cohort, recipient/count checkpoints and
  original anchor; all original metrics and short-view counts reproduced.
  The full prediction and score replay matched the immutable receipts.
- `not_run`: DroneCrowd calibration of these candidates and independent
  confirmation. The existing 15/30 split, 2% rule and sealed confirmation
  are unchanged.
- Native observation 8 / prediction 12; OpenDD dataset-local coordinates.
  No physical-unit, seconds, true 3D, foundation, Stage5C or SMC claim.

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_donor_motion_shape.py
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_donor_motion_shape.py --replay
```

The runner is single-process with four Torch threads, writes a heartbeat per
site and resumes from completed site receipts. The replay took about three
seconds locally. Sixteen focused tests passed, including donor exclusion,
held-input rotation, exact radius, and the existing calibration/confirmation
guards. The replay needs the existing local OpenDD data and frozen checkpoint
receipts; those assets are not part of this report.

[Registered protocol](registration.json) | [Per-site and seed results](analysis.json)
