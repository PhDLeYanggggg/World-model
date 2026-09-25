# Execution and Provenance

Date: 25 September 2026. Local native arm64 environment, CPU4/inter-op1/workers0.
No DataLoader multiprocessing, MPS probing, remote training or NumPy substitute
for model training. This phase has no new training; it operates on frozen scores.

## Registered Attempt and Repair

The original pilot (PID51991) finished in 2.293s, including source loading.
It created causal decisions for 202,146 indexed rows and read no new outcomes.
The complete original attempt (PID52141) stopped with exit1 at the registered
common-support check, after 15 partial decision groups, before new readout.
This was an explicit correctness failure, not a hang or a timeout downgrade.

A fresh causal-score-only audit found two unequal-support groups. The original
files and partial archives were preserved. The pre-readout amendment (commit
53eb3ff4) adds full/common anchors and an explicit support component. No risk
limit, outcome definition or selected anchor changed.

## Completed Amended Runs

| Phase | PID | UTC span | Result |
|---|---:|---|---|
| Decide all groups, then evaluate | 52536 | 10:37:51-10:38:43 | Exit0; 53.080s including load; 216 views. |
| Full decision/metric replay | 52679 | 10:39:52-10:40:44 | Exit0; 53.870s including load; exact analysis reproduction. |
| Separate sorting/coordinate verifier | 52925 | 10:43:40-10:45:22 | Exit0; 216 arrays, 864 reductions, 108 decompositions. |

All 36 amended decisions were saved before new metric evaluation. Parent72
controls exactly reproduce. The separate verifier shares the audited data and
prediction banks, but independently implements scalar sorting, coordinate
errors, error reductions and additive contrasts. This is computational
cross-checking, not independent statistical replication.

Analysis SHA256:
`9b4f98e253b15e92d0a1a27cd45c013144cf4814759df8a8f4942414a703c9fc`.
Parent hurdle analysis SHA256:
`29bb7dc9a39bda42699094b7176e313934766e2202e1d3b74b19e6845fb1b331`.

The [completion receipt](completion_checks.json) binds the analysis, reporter,
summary, safety summary and test report. [Replay](replay.json) and
[separate verification](separate_verification.json) are retained. The final
scoped suite passes 227 tests in 36 files. It is not the full legacy suite.

## Storage and Resume

Public outputs contain code, configuration, aggregate metrics, reports and an
aggregate SVG. Row-level decision archives, prediction banks, private analysis,
PNG preview and all checkpoints remain local and excluded from Git. The large
locality table is split into ADE/FDE aggregate CSVs without rounding away data.
Existing unrelated staged and unstaged files remain untouched.

Private identity, heartbeat, event log and immutable decision archive paths:
`data/stage_cvpr2027_experiments/european_hurdle_coverage_v1/support_v2/`.
Restarting an unchanged phase checks hashes and reuses/reproduces immutable
artifacts. A changed binding is an error, not permission to overwrite evidence.
The original failed attempt remains in its parent directory.

CREATE was not needed for this local diagnostic. The previous study's read-only
queue observation at 09:47:01 UTC is only cached, timestamped context, not a new
live availability or job-completion assertion. No remote jobs were changed.

All twelve source localities are opened development; reserved roles remain
closed. No deployment change, independent final-test claim, seconds/metric/
physical-safety claim, Stage5C execution or SMC.
