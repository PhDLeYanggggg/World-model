# Support-Preserving Motion Exchange

**The marginal risk guard preserves the fixed-half passes, but adds
no established forecast gain.** All policies edit the same 30,075
target-seed rows and keep the same sum of realized displacement radii
within every query. No selected-easy reference is lost.

[Frozen protocol](protocol.md) | [Full metrics](analysis.json) |
[Outcome-aware upper envelope](oracle_headroom.json) |
[Direct exact-action follow-up](../opendd_direct_exchange_v1/README.md)

21 fresh tree heads, seven held OpenDD development locations and
three seeds. Train-side predictions are from pair-excluded incumbent
producers. Every policy starts at fixed half. For a candidate pair,
move one quarter of the smaller incumbent radius from one target to
the other, preserving support. The learned guard accepts exchanges
with predicted gain and no predicted increase in any original risk
numerator. Actual safety is computed only after inference.

| Policy | Conservative gain vs half, pp [95% CI] | Queries exchanging | Passes /606 |
| --- | ---: | ---: | ---: |
| Fixed half | Reference | 0 | 547 |
| Past hindcast | -0.0107 [-0.0288, +0.0072] | 9,714 | 543 |
| Learned utility | -0.0008 [-0.0053, +0.0035] | 5,513 | 547 |
| Learned risk guard | +0.0007 [-0.0011, +0.0032] | 1,904 | 547 |
| Matched past hindcast | +0.0002 [-0.0027, +0.0031] | 1,904 | 547 |

Intervals resample seven locations after averaging seeds (3,000
paired draws). The last control uses the guard's active queries and
exact transferred radii. The guard-to-matched utility contrast is
unestimable: four required site-seed views make no exchange. In the
full fixed-half comparison, no-exchange views legitimately contribute
zero difference. The original risk result is unchanged for every
recording-seed view: 547 pass, 29 fail with defined risk and 30 are
undefined. The primary development criterion is not met.

## Headroom is present

An exact outcome-aware enumeration checks every realized action in
the registered menu. It is not a causal predictor and does not set
thresholds. Among 9,717 eligible query-seed views, 9,193 admit some
strictly positive lower gain and 8,550 admit one with no increase in
any of the three risk numerators. The guarded hindsight envelope
gains +0.082 pp [0.066, 0.093] and passes 551/606; the utility-only
envelope gains +0.087 pp [0.076, 0.097]. Thus the marginal head's
failure is not explained by absence of useful actions in its menu.
Even the guarded envelope cannot meet the full 606-view requirement.

## Reproduce

~~~sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_support_exchange.py register
.venv-pytorch/bin/python scripts/run_m3w_opendd_support_exchange.py run
.venv-pytorch/bin/python scripts/run_m3w_opendd_support_exchange.py replay
.venv-pytorch/bin/python scripts/audit_m3w_opendd_exchange_headroom.py
~~~

The head receipt verifies cached model identities; a separate no-fit
run reproduces the result exactly. Native arm64 CPU, four Torch
threads, zero workers. No new source, deployment, Stage5C, or SMC.
