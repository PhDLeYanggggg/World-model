# Moving Controls Narrow the Video-Input Gap

## Material Passport

Two fresh local image-correspondence runs and a hash-verified paired analysis.
No forecast model was trained, no future target API was called, and the approved
eight-observed/twelve-predicted native-step protocol is unchanged. The result is
input-diagnostic evidence, not a new ADE/FDE gain or independent generalization.

## What Changed

The preceding audit could decode stationary-agent images, but stationary crops
cannot meaningfully test temporal correspondence. This follow-up selects the first
contiguous eight-observation history with at least 12 projected pixels net motion
for each fit agent, then takes a fixed SHA256 ordering of up to 12 agents per site.
It uses no future completeness or future-change label in selection. ETH has 334
eligible agents and Hotel 278; 12 from each are inspected. These moving controls
are deliberately not representative of stationary starts or all agents.

All eight declared past/current images were retrieved for each control: 192 image
requests and 187 distinct frames. Four Hotel centered inspection crops lack full
image support, which remains explicit. All six contact sheets were inspected.
The markers commonly lie near the visible head/upper body in these examples,
making the previous upward-biased footpoint-style crop inappropriate as a body
assumption. This is qualitative inspection, not verified head/pose ground truth;
group overlap, occlusion and low resolution remain visible.

## Past-Frame Correspondence Results

A fixed zero-mean normalized cross-correlation matcher sees an earlier observed
image patch and the next observed image. Its search is centered at the earlier
annotation, not the later location. The later annotation scores correspondence
only; both images and both annotations precede or equal the query time. It is
not prediction beyond the observation window.

With radius 24, ETH median errors are 1.41 and 1.00 pixels for templates 15 and
31, against a mean zero-shift displacement of 12.87 pixels. Hotel initially has
mean errors 22.32 and 18.07 pixels. But 56/84 Hotel annotated displacements lie
outside that fixed search square. Its 28 in-range pairs have much smaller mean
errors of 2.52 and 1.78 pixels. All 84 ETH displacements are in range.

An explicitly registered adaptive follow-up changes only the search radius to
64. It keeps both template sizes, all selected controls, the direct frame-index
mapping and missing-support behavior. It fits no temporal offset.

| Source | Template | Jointly supported pairs / 84 | Radius24 mean error px | Radius64 mean error px | Radius64 missing pairs / 84 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETH | 15 | 80 | 3.065 | 6.493 | 4 |
| ETH | 31 | 77 | 3.603 | 7.513 | 7 |
| Hotel | 15 | 76 | 22.122 | 7.665 | 8 |
| Hotel | 31 | 69 | 16.707 | 4.223 | 15 |

Hotel improves even on identical support, so its initial large mismatch cannot
simply be read as temporal misregistration. ETH gets worse when distractors are
admitted. The wider search still has 11/15 ETH and 10/4 Hotel errors above ten
pixels among jointly supported pairs for the two templates. Thus neither search
is a reliable universal identity or orientation estimator. No best radius is
selected for forecasting. Boundary failures are not imputed as zero error.

The 168 adjacent pairs and two templates are repeated observations of 24 agents
at two fit sites, not independent trials. No bootstrap claim is made from this
small source diagnostic. Full per-setting metrics and paired support counts are
in [results.md](results.md) and [comparison.json](comparison.json); original run
results remain unchanged in their own directories.

## Source and Claim Boundaries

The [ETH Computer Vision Lab dataset page](https://vision.ee.ethz.ch/datsets.html),
checked 2026-09-17, states research-use scope and requests attribution. Its BIWI
section links annotation/video data. That narrows the local mirror's missing
license information; it does not authorize redistribution or establish every
local byte's original provenance. Images and row-level records remain local.
The indexed original paper PDF returned 404 when opened; its detailed point
semantics were not verified from that document.

The sampled correspondence supports the plausibility of the documented direct
frame-index mapping. It does not certify every agent, exact capture-time causality,
annotation construction, physical scale or the ETH clock. Encoded rate 25 versus
the ETH helper's 15 remains unresolved for physical seconds. Dataset-local
coordinates and native steps stay as such. Do not shift video timing to optimize
future prediction. No supplied velocity, destination/group label or future
endpoint enters these checks.

## Consequence for the Next Experiment

This moves the bottleneck from inability to access images to whether past
appearance has predictive value under a declared native-index mapping. A next
fit-only probe can be registered with centered patches and explicit visibility
masks, retaining the original trajectory comparator and primary metric. Missing
patches need a trajectory-only fallback; matching confidence must not be treated
as pose confidence. Compare appearance versus trajectory under the same scene
folds rather than choosing a search radius from outcomes. Do not promote these
moving controls to evidence that stationary body direction is inferable.

A new forecasting experiment has not run in this turn. Independent calibration
and confirmation sites, useful candidate forecasts, joint-decision lift and the
pending prospective primary-metric decision remain gaps. Current study negatives
and historical exposure remain unchanged. This is not a world-model success,
deployment or submission-readiness result; Stage5C and SMC remain off.

## Reproducibility

Both runs completed locally in approximately 6.90 and 12.59 seconds; these are
image-processing times, not neural training. Registrations bind source videos,
annotations, helper code, tests and decisions. Completion receipts bind aggregate
reports and local controls/pairs. The paired summary verifies both runs and exact
control identities before comparing. Fourteen relevant tests pass, including
synthetic translation, future-tail invariance, crop support, missing-pair handling
and row-identity mismatch rejection. The full legacy test suite was not rerun.
No current process from this diagnostic remains active.
