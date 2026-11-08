# Can the Same Intervention Budget Be Used at Better Times?

I froze the original 1,457 edit slots, including the count and the multiset
of peak CV-relative displacement radii in each recording. I changed only
when a slot can be used; eligible recipient assignment and the six-shape
observed-prefix rule are identical across arms. An edit still uses one
distinct agent, respects its native CV-to-OLS4 radius, and leaves at least
one unedited agent at that query. All witnesses pass the original streaming
ledger (0.2 earned per eligible query, maximum balance 2, cost 1).

## Label-Aware Capacity, Not a Model Result

| Timing window, in query indices | Joint witnesses | No eligible easy | Risk-infeasible | Nonpositive upper bound | No edit slots / no easy |
|---|---:|---:|---:|---:|---:|
| Fixed | 206 | 83 | 92 | 51 | 172 / 2 |
| Only delay up to 2 | 298 | 32 | 55 | 47 | 172 / 2 |
| Only delay up to 8 | 324 | 20 | 44 | 44 | 172 / 2 |
| Advance or delay up to 8 | 338 | 3 | 45 | 46 | 172 / 2 |

The solver retains every original slot, then maximizes the smaller of
missing-step-debited gains against CV and OLS4 subject to the unchanged
2% recording risk checks. Every positive incumbent was reconstructed
and re-evaluated in the original evaluator. No solve was unresolved;
all fixed witnesses were recovered with each enlarged timing window.
Of 118 additional witnesses under delay-8, 92 already appear with
delay-2. Allowing advance adds only 14 beyond delay-8. Advance of a
future slot cannot be used as a causal rule. Delay-only capacity also uses
future labels for its choices and is not a deployable policy.

The 606 views repeat 202 recordings across three frozen seeds and seven
exposed OpenDD locations. Of the 172 zero-edit views, 105 cannot accrue
even one edit credit under the original ledger. Another 14 have enough
credit but no easy candidate at a payable query even if nothing is spent
before it; 53 have at least one reachable easy candidate (not necessarily
a risk-feasible edit). With less than one credit, any policy under this
ledger emits zero edits, so the original selected-easy denominator is zero
regardless of training quality. This is a budget limitation, not a risk pass.

## Causal Readout

At release, a slot may be spent when the past-only score is positive. The
wait policy gives it up to eight later queries; unspent slots expire. The
matched diagnostic retains exactly the slots the wait policy emitted but
replays them at their original recipients and times. It is retrospective and
does not replace the full policy's 2% test.

| Policy | Edits | Joint passes | Undefined easy risk |
|---|---:|---:|---:|
| Frozen prefix shape and roster | 1,457 | 77 | 334 |
| Prefix, immediate | 1,267 | 73 | 353 |
| Prefix, wait | 1,443 | 74 | 334 |
| Prefix, matched replay | 1,443 | 78 | 336 |
| Learned score, immediate | 610 | 36 | 507 |
| Learned score, wait | 1,128 | 54 | 459 |
| Learned score, matched replay | 1,128 | 58 | 383 |

Compared with the equal-emission diagnostic, waiting changes conservative
gain by -0.092 pp [-0.255, +0.014] for prefix and -0.015 pp [-0.046, +0.010]
for the frozen learned score. The 95% intervals resample locations 3,000 times
after averaging seeds; gain is relative to common CV error mass. Relative
to the full frozen roster, both wait rules lose paired gain. The better
hindsight capacity is not recovered by changing when a positive local
score first appears. A separate [release-value experiment](../release_wait_v1/README.md)
fits the choice to wait directly.

This is closer to constrained deferral than reject-option classification:
there is an explicit fallback cost on the same trajectories, but actions
also compete for chronological credit and a recording needs observed easy
support. See [Mao, Mohri and Zhong, ICML 2024](https://proceedings.mlr.press/v235/mao24d.html)
for multi-expert regression deferral; this experiment does not claim a
general-purpose deferral algorithm or a theoretical risk guarantee.

```sh
.venv-pytorch/bin/python scripts/run_m3w_slot_timing.py
```

Fresh solve and readout in 15.71 seconds on native arm64 CPU Torch, four
compute threads, no training and no CREATE job. The same command
reuses completed per-recording oracle receipts and checks the immutable
aggregate while regenerating causal outputs. Raw native obs8/pred12 only;
no independent confirmation access, deployment change, Stage5C or SMC.

[Registration](registration.json) | [Aggregate readout](analysis.json).
