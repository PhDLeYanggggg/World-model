# Can Reallocating Small Forecast Edits Catch OLS4?

**Answer on Hannover: no, even at twice the total budget.** This is a
new capacity calculation on already examined data, not a new model,
confirmation result, or proof that OLS4 is always more accurate.

## Comparison

I retained the registered Hannover roster: three clips from one
acquisition campaign, 162 nonoverlapping observation-8/prediction-12
queries, 1,171 pedestrian targets, and three frozen OpenDD seeds. Of
the targets, 960 have complete futures, 194 partial, and 17 none.
Three seeds do not create independent sites. The source hashes and
existing split and reference definitions are unchanged.

For each recording and seed, I summed the peak CV-relative displacement
across all rows of the frozen half-radius forecast. I call that total
`B`. Unlike the [earlier per-row cap](../optional_reference_bridge_v1/README.md),
this audit can pool the displacement and give more of it to one row
than that row had originally. I compared:

1. An optimistic **risk-free bound** permitting any forecast path inside
   each row's assigned CV ball. I checked both the original recipients
   and the entire roster.
2. A **finite-bank hindsight solver** with the original eligible roster,
   optional frozen directions, and 25/50/100% blends from CV to OLS4.
   Its budget is the sum of chosen row peaks. It chooses at most one
   action per row, maximizes the lower of the CV and OLS4 utility gains,
   and keeps the original per-recording 2% risk, signed easy loss and
   easy support rules. It does not require matched edit counts. The
   original scorer verifies the witnesses after optimization.

| Total budget | Arbitrary-path upper gain vs OLS4 | Finite-bank maximin upper gain | Verified gains above both references with risk pass |
|---|---:|---:|---:|
| Original `B` | -40.14 to -22.51 pp | -41.40 to -23.79 pp | 0/9 |
| Twice original `B` | -37.06 to -19.38 pp | -39.74 to -22.11 pp | 0/9 |
| No cap (previous finite-bank audit) | Not run | +6.18 to +10.04 pp | 9/9 |

The arbitrary-path ranges use all recipients; the original recipient
restriction gave the same bounds at both budgets. Finite-bank runs
returned solver optima at numerical tolerance and risk-supported
incumbents in all 18 cases. None was a positive dual-reference witness.
All gains use observed CV error mass as the common denominator and
charge disagreement on missing steps. These are not observed-label
ADE/FDE improvements; reversing the reference is not sign-symmetric
when labels are missing.

## Why The Upper Bound Holds

Start with CV everywhere. On an observed step, a CV-centered move of
radius `r` can reduce CV's error by at most the smaller of `r` and
the CV error. On a missing step it can reduce the OLS4 disagreement
debit by at most the smaller of `r` and the CV-to-OLS4 distance.
Projection onto the ball attains these limits separately on every
step, since this relaxation imposes no path dynamics. Summing yields a
concave piecewise-linear return for each row. Allocating budget in
descending marginal return order gives the exact relaxed OLS4 upper
bound. The calculation may use future labels to price the ceiling,
but none of these allocations is a deployable input or policy.

Even in this relaxation, merely reaching zero against OLS4 requires
8.43 to 15.01 times `B`. That is a necessary budget in this scoring
setup, **not** an admissible deployment threshold. A stronger forecast
reference or larger proposal may change capacity, but still has to
pass the original risk rule using a past-only policy. Tuning who
receives the same small edits cannot close this gap on these clips.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/audit_m3w_hannover_shared_radius.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_shared_radius_capacity.py tests/test_m3w_optional_dual_capacity.py
```

The runner uses the existing Hannover archive, frozen OpenDD
checkpoints and cohort cache. It verifies their lineage; no weights
are trained and no forecast choices are exported. The
[registration](registration.json) fixes the budget and scorer, and
[analysis.json](analysis.json) contains the per-clip, per-seed bounds.
DroneCrowd's 15/30 roles and confirmation seal, deployment, Stage5C
and SMC are unchanged.
