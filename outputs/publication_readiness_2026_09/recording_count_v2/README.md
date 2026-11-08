# Recording-Aware Count Supervision

**Outcome: no qualifying controller.** This is a development experiment
on seven previously exposed OpenDD locations (202 distinct recordings),
not independent confirmation. The eight observed and 12 forecast
native steps, original CV-relative 2% recording risk, and treatment of
undefined easy support are unchanged.

The previous additive action scorer fixes the recipient and amplitude
menu for counts 0, 1 and 2 at each query. For supervision on each donor,
the additive scorer was refitted with both that donor and the outer
location excluded; its easy cutoff excludes both as well. A query-local
head learns the best signed-risk-penalized local utility. A second head
learns counts from an integer plan maximizing the lesser conservative
gain against CV and OLS4 for the entire donor recording. The teacher
requires original risk, a nonempty easy reference, strict gain against
both references and an aggregate edit quota. Verified plans were
rescored with the original evaluator. Certified no-positive and
no-support plans emit no-edit training targets; unresolved plans are
omitted from recording supervision. All 3,636 donor recording-seed
appearances include 2,290 positive witnesses and one unresolved plan;
these are overlapping training appearances, not independent outcomes.

At evaluation the two online heads have the same chronological token
budget: start empty at each recording, add 0.2 credits per eligible query,
store at most two, spend one per edit. Only current observations and past
credits enter the decision. The original additive baseline and its
nested-menu implementation differ in one amplitude across all views;
the archived decoder reproduces exactly, and both new heads share the
same nested menu. The initial `v1` run stopped on this discrepancy.

| Count rule | Edits | Risk passes | Risk pass and beat both | Undefined easy | Gain vs CV (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen hurdle counts + additive ranking | 5,214 | 104 | 63 | 201 | +0.026 |
| Online query-local | 1,768 | 87 | 55 | 291 | +0.051 |
| Online recording-aware | 780 | 83 | 57 | 400 | +0.026 |
| Query-local, no token cap | 12,100 | 60 | 32 | 72 | -0.163 |
| Recording-aware, no token cap | 1,477 | 113 | 57 | 322 | +0.007 |

Counts are over 606 recording-seed evaluations. Direct online gain
for recording-aware versus query-local is -0.045 pp, with a 95%
location-bootstrap interval [-0.279, +0.157] after averaging seeds.
Fewer interventions do not resolve the support constraint.

**Exact-count attribution.** For each recording-seed pair, use the
smaller of the two online totals and solve a label-blind score-maximizing
allocation under the same chronological token constraints. The choice
uses all query histories in the recording, so this is retrospective,
not an online result. Both arms use exactly 780 edits, with nonzero
matches in 339/606 recording-seed evaluations. Recording-aware gives
57 versus 53 dual passes but loses direct conservative forecast gain:
**-0.076 pp [-0.165, -0.009]**. The better pass count does not imply
better forecasting or universal risk control.

The experiment does not separate the limits of the nested recipient menu
from those of the chronological budget. Quantify their joint ceiling
before investing in another controller. No change to deployment,
Stage5C, or SMC. DroneCrowd stays at 15 calibration groups and 30 sealed
confirmation groups.

```sh
.venv-pytorch/bin/python scripts/run_m3w_recording_count.py
.venv-pytorch/bin/python scripts/run_m3w_recording_count.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_recording_count_matched.py
```

The registered `v2` execution is `fresh_run`: native arm64 CPU Torch,
four threads, zero workers, 63 pair-excluded additive rankers and 42
count heads, 65,100 updates. The attribution is `fresh_run`; the
zero-update checkpoint replay is `cached_verified`. Independent
confirmation is `not_run`. [Readout](analysis.json) |
[Exact-count attribution](matched_diagnostic.json).
