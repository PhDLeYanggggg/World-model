# Coupled Other-Recording Cost Retrieval: Fitting-Only Control

## Material Passport

Planned fresh_run: exact k32 retrieval and cost prediction on the original six
fitting contexts, followed by complete fixed-roster scoring and replay.
Cached_verified: frozen forecasts, 72 paired completion-cost heads and source
packets. Not_run at registration: pilot, full retrieval/readout, independent
evaluation, new neural or trajectory training and deployment.

## Question And Difference From Earlier Neighbors

The old local-cost analogue used selected donors, separately estimated two
harm quantities and applied reference offsets. Its loss-aligned variant
violated EH<=H and did not repair risk prediction. Do not relabel that failure
as untried. The current control instead predicts all five costs for the current
seven-action bank from common, all-causal donor neighborhoods, without offsets.
It tests a nonparametric alternative to the recent MLPs; several representation
and target details differ from the older analogue, so their contrast does not
isolate moment coupling as the unique cause.

## Fixed Inputs And Donors

Use the original fitting packets only; do not load inner-held, independent
selection, calibration, confirmation or DroneCrowd rows. Freeze all forecasters
and original physical-fitting feature normalizers. Form one 398-dimensional
descriptor by averaging the six physical actions' standardized 380 motion plus
18 static-region fields. This is a specified compressed distance representation,
not all possible history information and not a learned neural metric. Neural
fields do not set distance or normalization for the physical control.

All past-eligible queries are evaluated, regardless of future-label support.
For each query recording, exclude its entire original recording from the
donor bank. Retrieve exactly 32 Euclidean neighbors from the remaining causal
rows, with one common neighbor set for all actions. Source-balanced weights
are inverse causal-donor source counts, specified without outcome support.
No k, metric, threshold, action amplitude or source search. Every donor and
query identity and descriptor hash is retained privately or by aggregate hash.

The model's fitting-only normalizer and upstream forecasts were exposed to
the fitting cohort. This is NOT whole-pipeline OOF or independent testing,
even though each prediction excludes its own recording's donor labels.

## Coupled Cost Construction

Observed target columns remain B,H,R,ER,EH, and unknown evaluation labels stay
NaN. Encode B/E,H/E,EH/E using the causal candidate envelope, and R/s,ER/s
using the original positive fitting cost scale. Keep reference columns common
across all actions. Convex positive donor averaging preserves EH<=H and
ER<=R; decoding uses the actual query envelope and scale, with B<=R enforced
before selection. No independent reference-offset correction is added.
Only causally real donor actions contribute action fractions. All causal
donors contribute the common reference estimates. Insufficient known donors
produce unsupported actions and fallback, not invented labels or safe zeros.

Four fixed modes share the retrieval measure:

1. local_known: mask unknown donor losses, normalize remaining weights.
2. local_bounds: include separate unknown endpoints [0,E,0,0,E].
3. global_bounds: same weights/recording exclusion/target but no distance.
4. shuffled_bounds: same neighbors; jointly shuffle known encoded targets
   within each donor recording, preserving unknown support and coupled rows.

These are proxy conditional means, not calibrated population risk bounds.
The unknown endpoints are not observed future errors. Fitting labels are
donor model state only; query futures/masks never enter distance, gating or
inference. A regression test changes the entire query recording's labels and
requires identical predictions. Train-only cost scales are unchanged.

## Engineering Pilot And Full Roster

Before primary scoring, time 128 causal queries in the first recording of
the full first context. Repeat neighbor retrieval exactly, verify moment
constraints and measure memory. Do not score query outcomes in the pilot.
Project full retrieval time from the real lookup, preserving the full target
roster even when slow. Use native arm64 CPU4/workers0, exact chunked search,
process lock, private heartbeat, 10 GiB disk reserve and 12-hour boundary.
Neighbor checkpoints are private, bound to query/donor/descriptor hashes,
support --resume and have a 512 MiB budget. No new CREATE job is warranted
unless the pilot demonstrates a need; reconcile jobs first if one is needed.

All six original contexts and four source packets per context are required.
Each of four modes is compared with frozen known-only and completion heads
for original controller seeds17/29/43: 288 source/mode/control-seed groups.
The retrieval policy itself is deterministic and repeated across comparator
seeds; this is NOT three independently trained new models. Freeze descriptor,
neighbor and prediction hashes before scoring that context. A separate process
recomputes the complete retrieval and readout exactly, not just cached metrics.

## Readout And Advancement

Retain physical-only and mixed banks, neural-only, actual frozen binary policy
and floor. Reproduce original paired-head policy metrics and action hashes.
Report full known ADE/FDE, full-recording completion risk, hard/easy subsets,
regret, unknown selections and same-count utility. Empty selected risk remains
not_estimable. The 2% complete all/easy recording gate is unchanged; no bad
recording or undefined denominator is removed. Count matching is utility only.

Primary local_bounds must satisfy complete mixed-bank risk, positive full and
matched completion-lower utility CIs versus the fresh known-only MLP control,
a positive mixed-minus-physical completion-lower CI and a positive full
completion-lower locality CI over global_bounds. Report all four modes even
if one secondary looks better. Use the existing 3,000 fitting-locality bootstrap
after averaging dependent context/control-seed views. No confirmatory inference,
deployment promotion, or opening of independent roles follows automatically.

## Boundaries

SOURCE is oblique detector-track midpoint pixels, obs8/pred12/stride12 raw
frames. Static polygons are proxies, not verified walkability. No future
endpoint, central velocity, test statistics or test-endpoint goals enter input.
No metric/effective-second, human-gold, true-3D, foundation or new world-dynamics
claim. Stage5C and SMC remain off; A-conference readiness is not established.
