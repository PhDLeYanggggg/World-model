# Athens: rehearse both readout contracts before selecting dates

The [frozen inputs and matched proposals](../pneuma_prefix_pilot_v1/README.md)
remain unscored. I exercised the evaluation on fabricated positions using
the same six-member, four-date structure. A separate read-only preflight
verified that the real runner can load the existing pilot receipts and
recognise the date grouping; it did not open future positions.

All four arms edit two distinct recipients per query and share the
same displacement radius. Which recipients are edited can differ by arm.
The primary comparisons are between matched forecasts; there is no claim
that two edits would be released. The reader also reports CV, OLS4 and
median velocity as absolute accuracy references.

## Decisions not yet made

The two options have different interpretations. Neither is registered
as the real experiment by the rehearsal:

| Option | Label access | Readout |
| --- | --- | --- |
| Four-date zero-shot | All four dates only after the donor operating rule and date roles are published | Equal-budget accuracy, paired relative motion, missing-label bounds and the unchanged 2% risk audit; no Athens-fitted release |
| Two dates for calibration, two for confirmation | Only the first two dates during calibration; the last two stay closed until the calibration freeze is published | Same matched comparison, plus a distinct calibrated-release result |

For a zero-shot operating rule, the [donor-only receipt](donor_cuts.json)
verifies easy and hard CV ADE cutoffs of 2.834019 and 8.006736 in
the donor coordinate units. Each of the 21 OpenDD exclusion pairs
contributes once (the repeated seed cutoffs agree). The recipe takes the
componentwise median of those prior fold cutoffs. It is a proposed
transfer of a difficulty definition, not an Athens quantile; the
prospective registration must bind this exact receipt if zero-shot
is selected. The 2% risk ratio is unchanged. In a calibrated readout,
cutoffs instead come from complete, positive-CV-error calibration
examples; the release threshold uses the mean frozen query score per
member and stops before the first tied score group containing a risk
failure.

The same day is one aggregation unit even when it has three simultaneous
drone members. Average seeds within member, members within date, then
show per-date effects and a descriptive 3,000-draw date bootstrap.
Neither design supports a claim of replication across cities. A required
date with undefined evidence is reported, not silently omitted. Available
ADE/FDE, tail and worst member error, intervention rate, and risk support
remain distinct from the paired conservative and optimistic bounds.

## Verified in the rehearsal

1. Registration opens no outcomes, and an unpublished registration or
   operating rule cannot trigger the real join.
2. The synthetic zero-shot path uses six confirmation members on four dates.
   The synthetic calibrated path uses four calibration members and two held
   members, with the dates kept separate.
3. An interrupted scoring run can continue from immutable member receipts;
   replay of a complete run reads no labels again.
4. An incorrect edit radius is rejected. The same-date grouping and
   missing-label bounds are covered by regression tests.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/prepare_m3w_pneuma_donor_cuts.py
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/rehearse_m3w_pneuma_confirmation.py
```

The real runner accepts `register`, `calibrate` and `confirm` with an
explicit author-approved `--protocol` file. There is no default real
configuration. It verifies that the exact registration and operating
rule are on the remote before it can join future coordinates.
[Rehearsal numbers](rehearsal.json) are fabricated and are not results for
pNEUMA. No real roles were assigned, no real futures opened, and no
deployment or submission changed.
