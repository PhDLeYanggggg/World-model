# Protecting the Support Edit

The last matched-budget ranker lost easy-reference coverage when it
changed which tracks were edited. This experiment asks whether
freezing the support edit makes the gain edit easier to improve.
The support recipient and trajectory are unchanged. The 943 edits,
gain query and displacement radius match the original. Before
support is observed, the protected arm may change gain direction
but not recipient. Co-observed support is allowed. Only 74 of
598 gain queries have observed support.

## OpenDD Development Results

| Arm | Joint positive | Risk pass | Undefined | Old easy reference lost | New easy reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 138 | 172 | 295 | 0 | 0 |
| Unrestricted gain | 119 | 158 | 301 | 20 | 14 |
| Protected gain | 129 | 165 | 295 | 1 | 1 |
| Protected, context masked | 131 | 165 | 296 | 1 | 0 |
| Protected, guarded | 136 | 171 | 295 | 0 | 0 |

Counts span 606 exposed recording-seed views. The original per-recording
2% rule is used; an undefined easy reference is not a pass. The protected
head loses -0.04442 percentage points in conservative utility versus the
original (95% paired site bootstrap [-0.11633, -0.00350]). Its effect
versus the retrained context-masked arm is -0.00417 pp
[-0.01292, +0.00081]. The guard makes 25 direction changes, no recipient
changes, and yields -0.01495 pp [-0.04953, +0.00580] versus the original.
No learned arm improves on the original in both utility and risk.

Protection reduces reference loss but cannot guarantee it: in some
recordings the original gain recipient is the only edited track with
an easy reference. One old reference was lost and one new reference
appeared, so the protected arm's unchanged undefined count conceals
turnover. This addresses an observed reference-support mechanism,
not the under-sampled state-action uncertainty studied in
[SPIBB](https://proceedings.mlr.press/v97/laroche19a), and imports no
safe-improvement guarantee.

## Fixed-Support Timing Capacity

The exact audit keeps the support path, original gain radius, and
original edit count. It enumerates legal gain recipients and three
directions at the original or a later query, and rescores witnesses
with the original evaluator. At the original query it finds 185
protected passes. Allowing later queries raises this to 283 while
preserving old defined references; unrestricted delay reaches 416.
These are hindsight ceilings, not an online policy or a risk pass
guarantee. A causal policy still has to predict whether waiting will
improve the remaining opportunity before it expires.

## Reproduction and Status

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_protected_gain.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_protected_gain.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_opendd_protected_gain_clock.py
```

- `fresh_run`: 42 fits with 500 updates each, native arm64 Torch, four
  threads, no loader workers; checkpoints and heartbeats every 100 steps.
  The training run took 44.3 seconds (PID 75506). The exact clock audit is
  separate and uses future labels only for hindsight scoring.
- `cached_verified`: original role allocation, forecasts and proposal
  lineage; checkpoint replay with zero optimizer updates. The stored
  [analysis](analysis.json) and [timing capacity](clock_capacity.json)
  bind the registration and parent receipts.
- `not_run`: independent confirmation and CREATE training. The approved
  DroneCrowd split is [15 calibration groups and 30 reserved confirmation
  groups](../dronecrowd_calibration_readout_v1/protocol.md). This OpenDD
  result does not qualify a new policy for calibration or confirmation.

The next model change to test on development is gain-slot deferral
with exact support preservation and a matched-budget early-use control.
No deployment or submission-readiness claim, no Stage5C, and no SMC.
