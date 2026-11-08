# Is the First Edit Blocked by Credit or by the Forecast Bank?

I changed when the first edit can be made, holding the OpenDD
development roster, nomination, candidate shapes, gain scores,
CV and OLS4 comparators, missing-step debit and 2% rule fixed.
Credit still accrues at one fifth of an edit per eligible query.
The modified policy can make one edit before its first credit
accrues, then must repay it before another edit. It uses no
recording length or future input. If a recording ends before five
eligible queries, its one possible edit is an explicit terminal
subsidy. On longer recordings the accrual-based terminal upper
bound is unchanged, but realized counts can differ.

A second policy requires the existing positive past-prefix score
for the advance. Once enough credit is available, both variants use
the original decision rule. Neither variant retunes scores or thresholds.
The recorded outcome is the full-policy comparison, not equal-count
evidence or an estimate of the effect of intervening on the real world.

| Policy | Edits | Risk passes | Undefined easy | Risk pass and beat CV and OLS4 | Paired lower gain vs anchor, pp (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen anchor | 1,457 | 112 | 334 | 77/606 | reference |
| Advance | 1,557 | 131 | 311 | 95/606 | -0.020 [-0.100, +0.036] |
| Advance with past-prefix gate | 1,495 | 120 | 324 | 91/606 | +0.010 [-0.005, +0.028] |

The full-grid paired lower gain subtracts prediction disagreement at
missing steps and uses observed CV error mass for normalization. The
intervals bootstrap seven locations 3,000 times after averaging three
seeds. There are 202 distinct recordings; the 606 recording-seed views
are not independent. Neither policy demonstrates better overall
forecast utility, and neither passes the full recording gate.

## The 105 Views That Could Not Accrue One Credit

The anchor edits none. Advance edits 41 and passes 3 joint checks;
prefix-gated advance edits 18 and passes 1. Neither comparison implies
safety at the same edit count. Zero-edit views still have undefined
selected risk and cannot count as passes.

| Exact one-edit hindsight ladder (views out of 105) | Fixed first nominee per query | Every legal row and three amplitudes |
| --- | ---: | ---: |
| Any candidate | 105 | 105 |
| Nonzero edit on a known, positive easy reference | 58 | 78 |
| Pass the original risk rule | 35 | 63 |
| Also beat CV | 21 | 45 |
| Also beat OLS4 | 15 | 30 |

For a single edit the edited row must itself supply the positive
easy denominator. The evaluator then checks all three original risk
ratios and strict conservative full-recording gain over both baselines.
The audit enumerates each admissible candidate and verifies one witness
per feasible recording with the unchanged scorer. It never sends
future-based choices to the controller.

This separates three limitations: 27 of 105 views have no editable
easy reference even in the full menu, 48 more fail either risk or
dual-baseline utility even with hindsight, and the causal rule finds
only 3 of the 15 feasible fixed-nominee views. Expanding the nomination
pool doubles capacity, but 75 views still cannot succeed with one edit
from this bank. Of the 30 full-bank witnesses, 21 come from one location.
More first-credit threshold tuning is unlikely to remove this gap.
Test candidate paths against OLS4 on exposed data before fitting
another budget controller.

```sh
.venv-pytorch/bin/python scripts/run_m3w_first_credit.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_first_credit.py
```

`fresh_run`: 21 site-seed readouts, about two seconds for the
compute phase, native arm64 Torch, four threads and no data workers.
The first attempt stopped at anchor verification: the straight
nomination paths are not the actual prefix-selected emitted paths.
The second registration binds that failed attempt and uses the
original emission code. The anchor then reproduces at all 21 cells.
`cached_verified`: checkpoint hashes, training rosters, original
anchor and exact resume from all 21 score checkpoints. No new training,
deployment or calibration. Independent confirmation is `not_run`;
DroneCrowd's 30 groups stay sealed. No Stage5C, SMC, metric or seconds claim.

[Registration](registration.json) | [Results](analysis.json).
