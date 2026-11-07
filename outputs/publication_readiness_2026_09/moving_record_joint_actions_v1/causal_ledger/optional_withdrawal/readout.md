# Causal Insertion With Optional Withdrawal

**Result: no recording gain; withdrawal was not exercised.**
The two-stage model learns a reachable insertion and a conditional
withdrawal, reserving a quarter of each mini-batch for positives
where available. It may withdraw only an original action, in the
same or a later query after insertion. The matched control uses
the exact same insertion without withdrawal.

| Arm | Donor-calibrated folds | Insertions | Withdrawals | Missing easy reference | Signed easy violations | Recording passes |
|---|---:|---:|---:|---:|---:|---:|
| Original direct | not applicable | 0 | 0 | 35 | 8 | 2/102 |
| Independently calibrated insertion | 2/42 | 4 | 0 | 35 | 9 | 2/102 |
| Matched insertion | 2/42 | 4 | 0 | 35 | 9 | 2/102 |
| Optional withdrawal | 2/42 | 4 | 0 | 35 | 9 | 2/102 |

One fold enables a withdrawal cutoff but no held withdrawal
triggers. Both the matched and optional policies have the same
decisions; their paired difference of zero is not a demonstration
of equivalence or safety. The paired completion-lower change
against direct is -0.000297 pp (source bootstrap 95% interval
[-0.000891, 0.000000]). No recording pass is added and no policy
is deployed.

## Where the Policy Stopped

All four frozen insertions occurred outside the 35 views that
lacked easy reference. Two have observed easy reference, but
none is a new non-loss pass either by itself or when paired with
any one legal withdrawal. Thus the frozen insertions did not
present a repair opportunity for the withdrawal head.

For diagnosis only, forcing the highest-scored first eligible
insertion in each of 33 eligible missing-reference views reaches
observed reference in eight. Only one of those first insertions
is a standalone non-loss rescue and one other is repairable
with a same-query withdrawal, using labels. No strictly later
withdrawal witness was found for those first choices. The forced
probe is not the calibrated policy and is not safety evidence.
This localizes the practical bottleneck to insertion retrieval
and calibration before a possible withdrawal can matter.

## Reproduction

Registered in `d89c2351` before fitting. With verified private
parents:

```sh
.venv-pytorch/bin/python scripts/run_m3w_optional_withdrawal.py fit
.venv-pytorch/bin/python scripts/run_m3w_optional_withdrawal.py score
.venv-pytorch/bin/python scripts/run_m3w_optional_withdrawal.py replay
.venv-pytorch/bin/python scripts/audit_m3w_optional_withdrawal.py
```

All 84 native arm64 PyTorch fits finished 220 updates in about
90 seconds locally, with four compute threads, no loader workers,
checkpoints and heartbeats. The 102 development views repeat 34
recordings across three frozen forecast seeds and 14 source
groups. Replay is exact for the scalar report. Checkpoints and
decisions stay local. The original 2% and support rules remain
unchanged, with no new independent evaluation, metric,
seconds-level, Stage5C or SMC claim.
