# Selected Harm By Released-Label Support

Fresh_run: all18 original G-role model pairs are inferred and all three
selected sets are decomposed. A second full execution reproduces every
partition. Cached_verified: original/extended models and unpartitioned
costs. No training update, new H score or threshold selection occurs.

## Numerator Contributions

All entries summarize the same444 repeated recording/pipeline views,
not444 independent scenes. Unknown EH is a conservative missing-label
contribution, not measured forecast error. Partial means available-label
evaluation only, not a complete twelve-step trajectory.

| Selected set | Median unknown share of EH | Whole easy violations | Observed-only easy violations / defined views | Whole violations also observed |
|---|---:|---:|---:|---:|
| extended_fixed_original_selection | 63.55% | 324 | 252/444 | 252 |
| extended_own_selection | 64.26% | 307 | 223/443 | 223 |
| original | 63.55% | 324 | 252/444 | 252 |

## Predicted / Target EH Ratios Within Each Partition

These are evaluation-only label partitions of the unchanged decisions.
No future availability mask is an inference feature. Undefined target
mass stays not estimable, never zero risk.

| Selected set | Complete labels | Partial labels | Wholly unknown |
|---|---:|---:|---:|
| extended_fixed_original_selection | 0.015161 | 0.030384 | 6.80783e-05 |
| extended_own_selection | 0.008336 | 0.010831 | 1.42016e-06 |
| original | 0.008077 | 0.011744 | 3.65983e-05 |

## Execution And Limits

Original PID63885: 154.281s, peak RSS12,221,333,504bytes.
Separate replay PID64229: 157.571s, peak RSS12,259,966,976bytes.
Every original selected row count and unpartitioned predicted/target
mass matches exactly; the complete/partial/unknown contributions sum
to that same mass. This is descriptive fitting evidence, not a new
primary, independent confirmation or uniquely identified cause.

Observed-only ratios cannot replace the original complete risk gate.
A view that fails only after conservative bounds is not thereby safe;
a view that also fails on available labels cannot be repaired by merely
dropping unknown rows. Neither subset permits future-validity filtering.

```sh
.venv-pytorch/bin/python scripts/run_m3w_duration_support_diagnosis.py register
.venv-pytorch/bin/python scripts/run_m3w_duration_support_diagnosis.py run
.venv-pytorch/bin/python scripts/run_m3w_duration_support_diagnosis.py verify
.venv-pytorch/bin/python scripts/report_m3w_duration_support_diagnosis.py
```

Unchanged interrupted execution supports run --resume. Completed phases
are immutable. Private inputs/checkpoints are required. SOURCE remains
exposed development, image pixels/raw frames and silver labels, not
metric/seconds, human gold or new dynamics evidence. Original failures
and deployment stand; Stage5C/SMC remain off and the goal is unmet.
