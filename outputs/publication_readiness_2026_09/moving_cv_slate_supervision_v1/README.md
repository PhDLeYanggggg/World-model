# Whole-query supervision under the original recording rule

## Question and design

Can supervision from an entire feasible recording teach a model which
edits to make together, rather than which individual paths look good?
The analytic solver found 8, 5 and 4 verified training recordings
across the three source-role folds. In those recordings I replaced
one query at a time at the same edit count and verified the new
recording's risk and conservative gain. A gain-only additive scorer,
a feasibility-first additive scorer and a feasibility-first joint
scorer used the same training draws. The joint scorer also has more
parameters. Teacher actions and future outcomes are not inputs at inference.

## Results

All 27 fits (3 heads x 3 folds x 3 seeds) ran locally in PyTorch.
There were 27,000 optimizer updates. Every matched arm made exactly
54,748 edits across 72 recording/seed views, with identical counts
in every query. Results below are held fitting-source development
results, not a new independent evaluation.

| Exact-count arm | Original recording-risk passes | Undefined easy views | Worst known signed easy change |
|---|---:|---:|---:|
| Frozen control | 14/72 | 19 | +61.90% |
| Geometric OLS4 agreement | 0/72 | 18 | +133.37% |
| Gain-only additive | 2/72 | 19 | +86.35% |
| Feasibility-first additive | 4/72 | 20 | +31.10% |
| Feasibility-first joint | 8/72 | 20 | +86.35% |

Feasibility labels changed 335 of 3,324 training pair preferences.
With the same additive architecture, they reduced worst easy harm
versus gain-only but reduced the paired full-grid lower gain by
3.122 pp (95% source-bootstrap interval: -6.351 to -0.817 pp).
Adding the joint head reduced that contrast by another 8.566 pp
(interval: -17.322 to -1.173 pp), even though its original risk
pass count rose from four to eight. These lower contrasts account
for missing outcomes; they are not estimates of the unobserved error.
The paired intervals resample the nine source groups 3,000 times
after aggregating seeds. **No calibration cell qualified (0/9).**
CV remains the fallback; zero edits do not certify recording risk.

## What limits this particular test

The exact-count, six-slate hindsight diagnostic found 5/72 verified
dual-reference and original-risk witnesses. A further 24 were
risk-infeasible, 27 had a nonpositive upper bound and 16 lacked
easy support. There were no unresolved solves in this run.
This is a diagnostic of the combined shortlist and frozen query
counts, not a claim that only five views are solvable in general.
The [earlier uncapped full-bank result](../moving_cv_pooled_coverage_v1/README.md)
verified 53/72 using a different feasible set. A useful next
comparison is to cross a larger slate library with training-only
query budgets, vary one restriction at a time, and first show that
the new menu admits dual-reference solutions before training a ranker.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/build_m3w_cv_slate_teachers.py
.venv-pytorch/bin/python scripts/run_m3w_cv_slate_supervision.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_fixed_slate_capacity.py
```

Training and the capacity diagnostic are `fresh_run`. The donor bank,
historical comparison and the exact no-update checkpoint replay are
`cached_verified`. The local runtime and required preceding outputs
are documented in the [Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
The capacity calculation uses outcomes only for retrospective scoring;
it cannot produce a selectable model or threshold. The 30 DroneCrowd
confirmation groups were not opened. Stage5C and SMC were not run.
Forecasts are native observation-8/prediction-12 in dataset-local
pixels. They are not metric, seconds-level or physical-safety results.

[Protocol](protocol.md) | [Frozen readout](readout.json) |
[Fixed-menu capacity](fixed_menu_capacity.json) |
[Manuscript v133](../evidence_manuscript_v133/README.md)
