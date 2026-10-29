# Joint Risk Repair: Coherent, Still Not Predictively Sufficient

## Material Passport

2026-10-29. Fresh local fitting on original TRAIN inner-fit; exposed upstream
models/preprocessing/selection. Not an independent test or new neural dynamics
model. Protocol and code committed as008fe6b8 before pilot. obs8/pred12,
rawstride12,pixel/local detector-silver. No held/independent role opening.

## What Ran

Native arm64 CPU4/interop1/workers0. Pilot PID84860 exit0:23.545532s,
5,306,286,080B peak RSS,7,572 queries supported. Full PID85170 exit0:
189.291540s,6,623,805,440B peak RSS,72/72 views. Same62,902 repeated
query occurrences (61,335 known/1,567 unknown), full external reference support.
Six contexts,three existing head seeds,four sources per context. This is not
62,902 independent examples or three new forecaster training seeds.

Full inference/retrieval/local-fit replay PID85722 exits0 and matches every
group and summary exactly,verified05:24:36 UTC. Separate receipt arithmetic
checks864 arm-target records,443 reference banks,72 donor objectives,144 prior
controls and8 conditions.21 source/context/seed slices,6 support strata and
figure receipt rechecked;figure visually inspected.79 related tests pass2.72s,
not the entire historical suite. Engineering reproduction is not replication
on independent data.

## Registered Primary Result

Equal-view source-normalized signed-risk MSE, NOT ADE/FDE:

| Arm | All risk MSE | Easy risk MSE | Easy>all violations |
|---|---:|---:|---:|
| Original head | 0.079458745 | 0.000309373 | 0 |
| Previous independent cost | 0.089637254 | 0.000442400 | 9019 |
| Joint cost only | 0.089642302 | 0.000435658 | 0 |
| Recording mean alpha | 0.078732500 | 0.000325425 | 0 |
| Shuffled alpha | 0.083404108 | 0.000347115 | 0 |
| Nested-support blend, PRIMARY | 0.080533858 | 0.000326962 | 0 |

Joint constraints remove the contradictory moments but improve easy MSE only
1.52% over independent fitting. The primary reduces prior easy MSE by26.09%,
yet remains5.69% worse than the original head; all MSE remains1.35% worse.
Only1/18 paired context/head-seed comparisons beats original and independent.
The magnitude screen FAILS (6/8 conditions pass). Conditional assignment also
fails: primary beats shuffled alpha but is0.47% worse than recording-mean alpha
on easy MSE. Mean alpha is not promoted posthoc; its easy MSE also loses original.

All12 source means,all6 context means andall3 head-seed means have worse easy
MSE than original. No source or undefined ranking view is dropped.

## What Was Learned

- Coupling is an engineering/statistical consistency fix, not sufficient risk learning.
- 3,183 queries have32 neighbors from just one recording. Their primary alpha
  is0, returning original predictions exactly. Unknown query labels stay unknown.
- 27,025/62,902 queries (42.96%) have alpha0;6,942 have alpha1;28,935 lie
  between. Mean alpha0.30294. These are risk-estimate blend strengths, not
  deployment intervention or safety-certified switching rates.
- Recording ESS is below2 for22,032 queries. Only4,816/9,019 prior violations
  occur there (53.40%); another4,203 occur at higher ESS. Low effective support
  is not a complete explanation of the logical inconsistency.
- The primary's predicted easy-harm mass is24.190950 versus observed95.560642.
  Of predicted mass,23.553876 (97.37%) lies on non-event queries. Shrinkage reduces
  excess mass but does not identify rare positive easy harm well enough.
- Optimized donor loss falls8693.823584 to8147.906203. This follows from fitting
  alpha and does not prove transfer to the query. The query readout still fails.

## Decision

Keep the existing deployment unchanged. No new neural trajectory model, policy
execution, risk-budget certification or held-role evaluation. Stage5C/SMC off.
Next test the separate all/easy donor-loss conflict of a shared alpha, then
consider a prespecified donor-easy-constrained objective if supported. Do not
replace the primary with the best observed control, tune query outcomes, or
open independent data to rescue this fitting failure.

[Detailed slices](slice_diagnostics.md),[failure analysis](failure_analysis.md),
[gate](evidence_gate.md),[reproduction](reproducibility.md).
