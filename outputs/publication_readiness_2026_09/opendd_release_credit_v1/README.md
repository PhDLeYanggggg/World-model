# Timing The Same Neural Edit Credits

The [candidate-advantage experiment](../opendd_candidate_advantage_v1/README.md)
found only 88 original-risk witnesses at the exact query counts.
Here I retain the same frozen ego-only EqMotion forecast for each row
and use the previous hardness count as the stream of released edit
credits. I do not re-use the older analytic-shape timing result.

## Exact Same-Total Capacity

Every label-aware schedule spends the same number of credits in each
recording. An edit may move only after release, at most 2 or 8 queries
later, with a local cap of current release plus one extra edit, bounded
by legal rows. For cumulative releases `R` and spends `E`, every prefix
satisfies `R[t-d] <= E[t] <= R[t]`, and the final totals are equal. FIFO matching proves
sufficiency for equal unit credits: the j-th spend cannot precede the
j-th release by the upper inequality, or miss its deadline by the lower
inequality. I use these integer prefix constraints inside the unchanged
2% risk program, maximizing the smaller of the two lower gains.

| Allowed delay (query indices) | No credits | No reachable easy | Risk infeasible | No positive dual gain | Qualifies |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 217 | 85 | 208 | 8 | 88 |
| 2 | 217 | 27 | 123 | 16 | 223 |
| 8 | 217 | 15 | 87 | 20 | 267 |

Zero delay reproduces every prior classification. Each positive
witness passes the exact original scorer and count checks; none is
unresolved. This is an outcome-aware bound for this neural candidate
bank and delay rule, not a feasible causal policy. Timing alone
cannot help recordings with zero releases under this stream.

## Causal Spending

The immediate reference is the previous gain-and-risk ranking.
All arms preserve that ranking in each query, using only its current
score and previously released credits. Fixed wait holds a credit at
least one query. Value wait defers if the risk-adjusted score is
nonpositive. Support wait additionally requires predicted easy support
while the product of estimated previous no-support probabilities is
at least 0.5. The product is a heuristic, not a calibration guarantee.
The deadline is two queries. No policy sees the end of the recording;
credits that cannot be spent before expiry or end remain unused.

| Policy | Edits | Unused credits | Undefined views | Risk passes | Joint passes | Matched immediate passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Immediate | 3,323 | 0 | 401 | 13 | 9 | - |
| Fixed wait | 3,065 | 258 | 371 | 23 | 15 | 9 |
| Value wait | 2,977 | 346 | 411 | 10 | 7 | 8 |
| Support wait | 2,792 | 531 | 401 | 23 | 14 | 8 |

The matched immediate control spends **only the credits the waiting
policy actually used**, on their original rows. It is retrospective
and matches the number of edits in every recording, not their
radius or query. It does not certify the full policy.

| Direct lower gain vs matched immediate | pp [95% interval] |
| --- | ---: |
| Fixed wait | -0.118 [-0.386, +0.127] |
| Value wait | +0.005 [-0.096, +0.096] |
| Support wait | -0.383 [-0.832, +0.039] |

Units are percentage points of common observed CV error mass.
The 606 views repeat 202 exposed recordings at three seeds.
The 3,000 bootstrap draws resample seven sites after averaging
seeds. Support wait also loses to CV directly, -0.239 pp
[-0.471, -0.028]. The improved counts for fixed or support waiting
are not evidence of improved forecast utility or an all-recording pass.

## Decision And Reproduction

The timing limitation is real, but these fixed causal rules do not
recover it. I do not advance a policy. A useful next comparison is
to learn the relative value of waiting from double-excluded forecasts
and charge expiry explicitly, rather than treat estimated easy
support alone as a reason to wait.

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_opendd_release_credit.py
```

`fresh_run`: exact timing solves and causal/realized-count comparisons.
`cached_verified`: frozen neural bank and prior controllers; rerun
agrees with the immutable readout. Seventeen focused tests passed.
No new training or CREATE job. Original 2% rule and 15/30 DroneCrowd
roles unchanged; calibration and confirmation not run. Native 8/12
annotation steps and dataset-local coordinates, not seconds or
metric claims. No deployment, Stage5C or SMC.

[Registration](registration.json) | [Site and seed results](analysis.json)
