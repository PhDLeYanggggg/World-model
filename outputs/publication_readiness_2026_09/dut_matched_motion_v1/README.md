# Does The DUT Neural Advantage Survive A Stronger Motion Control?

**OLS4 is the stronger DUT accuracy baseline.** The old complete-case
readout showed a large raw EqMotion gain over CV and damped CV. It
did not include OLS4. I added OLS4 and tested whether the neural path
helps on exactly the same objects and at the same edit size as motion.

## What I Compared

I kept the previously exposed [DUT](https://arxiv.org/abs/1902.00487)
roster: 27 recordings at two physical sites, with the same duplicate
annotation quarantine. The time-only rule takes the first query
and advances by at least 20 raw frames. This gives 469 nonoverlapping
eight-observation/twelve-prediction queries and 21,731 targets:
20,667 complete, 1,012 partial, and 52 without future labels.
All remain in inference. The original full-cohort result is not
replaced by this separate comparison.

All three original source-trained EqMotion seeds and bounded-fraction
heads are frozen. Each control is computed from the same past
positions. I matched the strict neural edits' recipients, query times,
and maximum per-row distance from CV. The corresponding damped and
OLS4 directions have nonzero support on all 1,978 nominations across
the three seeds, so no intervention was removed. The original strict
neural result and its matched counterpart are identical.

| Disjoint-query forecast | Complete ADE | Endpoint FDE | Worst site/seed known easy change vs CV |
|---|---:|---:|---:|
| CV | 3.836 | 6.868 | 0.00% |
| Damped CV, 0.05 | 3.633 | 6.628 | +586.52% |
| OLS4 | 1.956 | 3.392 | +80.94% |
| Raw EqMotion | 2.253 | 3.782 | +108.19% |
| Strict EqMotion | 3.778 | 6.720 | 0.00% |
| Matched damped CV | 3.778 | 6.740 | 0.00% |
| Matched OLS4 | 3.751 | 6.712 | 0.00% |

ADE and FDE use native dataset coordinates. The learned strict
rule makes 1,978/65,193 edits (3.03% across repeated seeds). None of
the matched arms has positive selected easy reference mass in any
of the 81 recording-seed views. Zero signed error change on known
easy rows is **not** an estimable selected-risk pass. The original
2% rule and source roles are unchanged. The separate native CV
precision-check alias in the JSON is not a second intervention arm.

## Direct Paired Results

| Neural forecast vs comparator | Complete-path ADE gain | Full-grid lower to upper gain, pp |
|---|---:|---:|
| Raw vs damped CV | +37.92% | [+33.628, +38.048] |
| Raw vs OLS4 | -15.21% | [-9.468, -6.243] |
| Strict vs damped CV | -4.00% | [-5.890, -1.582] |
| Strict vs OLS4 | -93.13% | [-50.587, -44.272] |
| Matched vs damped CV | -0.03% | [-0.096, +0.022] |
| Matched vs OLS4 | -0.72% | [-0.759, -0.664] |

The full-grid endpoints are outcome-completion bounds using the
common observed CV error denominator, not confidence intervals.
For matched OLS4, even the upper gain for the neural direction is
negative in both sites (-1.084 and -0.245 pp, seed-averaged).
The separate 3,000-draw paired site bootstrap for the lower endpoint
is [-1.259, -0.258] pp. It describes two sites, not a precise population
interval. Raw OLS4 has lower ADE on both pedestrians and vehicles in
each of the six site/seed views. On the hard subset, matched OLS4
also has lower observed ADE than matched EqMotion in all six views.

## Full-Roster Check

I independently reconstructed the old 9,147-query population from
past positions, recomputed OLS4, and checked all 27 recording target
counts, complete counts, known-step counts and native CV ADE/FDE
sums against the bound original receipts. On the same **399,729
complete targets** (420,364 past-eligible), the site-equal comparison is:

| Forecast | Complete ADE | Complete FDE | Evidence |
|---|---:|---:|---|
| OLS4 | 1.957 | 3.404 | Fresh analytic forecast |
| Raw EqMotion | 2.250 | 3.781 | Verified original neural aggregates |

OLS4 wins all six site/seed ADE views on the full roster. This check
does not reconstruct pointwise missing-label bounds from aggregate
neural errors. The earlier benefit over CV is real on its stated
comparison, but it does not imply a stronger neural forecaster or
a better intervention policy. I did not promote OLS4 to a deployed
fallback, change thresholds, or open independent confirmation.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_dut_matched_motion.py
.venv-pytorch/bin/python scripts/run_m3w_dut_matched_motion.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_dut_full_ols4.py
.venv-pytorch/bin/python scripts/build_m3w_dut_matched_replay.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_dut_matched_motion.py tests/test_m3w_dut_readout.py tests/test_m3w_frozen_policy_chain.py
```

Native arm64 CPU, four Torch threads and no loader workers. The
fresh disjoint readout took 307.54 s; fresh-process replay reproduced
all 27 recording predictions and scores in 306.78 s. Twenty-nine
focused tests passed. Per-recording atomic checkpoints
and heartbeat are under `data/stage_cvpr2027_experiments/dut_matched_motion_v1/`;
rerun the same command after confirming the original process exited.
No new training or CREATE job. The anonymous packet contains only
unnamed site/seed scalars; it reconstructs 21 descriptive intervals.

[Registration](registration.json) | [Disjoint analysis](analysis.json) |
[Full baseline check](full_ols4.json) |
[Manuscript v119](../evidence_manuscript_v119/README.md)
