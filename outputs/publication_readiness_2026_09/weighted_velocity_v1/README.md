# Learned observation weights for full-strength velocity

**Decision: retain the existing policy.** Full-strength weighted
velocity improves over a fixed robust fit on the whole development
roster, but does not establish an improvement over OLS4 or pass the
original risk check. It adds no new dual-reference capacity witness.

## What changed

The 8 observed positions define a weighted straight-line estimate;
the 12-step path starts at the actual last observation and uses only
the fitted slope. The network predicts positive weights from signed
and unsigned motion residuals. It does not predict unrestricted
future offsets. The fixed robust comparator downweights residuals
iteratively, in the spirit of [robust M-estimation](https://doi.org/10.1214/aoms/1177703732);
it is not Huber's original location estimator. Other controls are OLS4,
OLS8 and the earlier quarter-strength nominee.

The paired neural arms use identical architecture, initialization,
batches and 600-update budgets. The treatment adds non-canceling harm
against CV and OLS4 and an explicit disagreement debit on wholly
unlabeled rows to the same observed ADE objective. Targets and masks
are used only for supervision and evaluation. Each fold has six fit,
two inner calibration and one outer development source group, with no
overlap within a fold. All views are from the previously exposed
MovingDroneCrowd++ development roster.

## Matched comparisons

| Direct comparison | Conservative gain difference, pp | Paired 95% source CI |
|---|---:|---:|
| Same recipients: dual loss minus ADE | -0.255 | [-0.565, -0.008] |
| Same recipients: dual minus quarter | -1.191 | [-4.552, +1.450] |
| Whole roster: dual minus fixed robust | +3.383 | [+0.602, +6.371] |
| Whole roster: dual minus OLS4 | +0.069 | [-2.821, +2.542] |

Both learned arms change exactly 95,769 paths in the matched comparison
across 72 recording-seed views. OLS4 makes 198 fewer nonidentity changes
on the same recipients because some OLS4 and CV paths coincide.
Whole-roster arms are forecasting diagnostics, not rate-matched
deployment policies. Neither learned arm passes any original risk
check, and no option qualifies in any of 27 inner fold-seeds.
At matched recipients, the dual loss reduces the worst signed easy
degradation from 129.41% to 117.80%, still far above 2%. The whole-roster
dual head gains +2.229 pp on observed labels versus OLS4, but most of
that gain disappears after debiting the missing labels. Intervals
use 3,000 paired source bootstrap draws after averaging training seeds.

## What the missing labels imply

For a wholly unlabeled row, let `d(A,B)` be the maximum waypoint
distance between two trajectories. The two registered conservative
debits for prediction `P` satisfy
`d(P,CV) + d(P,OLS4) >= d(CV,OLS4)` by the triangle inequality.
With the same CV error denominator and source aggregation, the
right side contributes **11.551 pp** across this roster. The two
debits of the matched dual model total 11.623 pp, only 0.071 pp above
that floor. Its observed gains sum to 6.104 pp; eliminating all excess
debit at unchanged observed gain still leaves **-5.447 pp**. This
is a necessary condition on the aggregate score, not a claim about
unknown true trajectories or every individual source. No risk rule,
label mask or pairwise completion bound was changed.

## Candidate capacity

I added the two learned paths and the fixed robust path to the original
full-strength menu. At the same original query counts and under the
unchanged dual-reference rule, outcome-aware optimization retained
24/72 original witnesses (eight recordings) and found **no new ones**.
The other views were 45 risk-infeasible and three negative-upper-bound;
none was unresolved. This diagnostic uses future labels to choose
actions, and says nothing about a deployment policy. Improving
observed forecast gain or acquiring verified missing-label support
is now more promising than reducing disagreement within this family.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/run_m3w_weighted_velocity.py
.venv-pytorch/bin/python scripts/run_m3w_weighted_velocity.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_weighted_velocity_debt.py
.venv-pytorch/bin/python scripts/audit_m3w_weighted_velocity_capacity.py
```

Native arm64 Torch: 54 fits, three actual seeds, four threads, zero
loader workers; 63.74 s fresh and 34.78 s no-refit replay. Atomic
training checkpoints every 100 steps and solver receipts per recording
stay local. Forty-five targeted tests passed. No official calibration
or confirmation was opened, no deployment changed, and Stage5C and
SMC remain off. Coordinates are image pixels, not verified metric or
physical time.

[Training and paired analysis](analysis.json) | [Triangle debit](debt.json) |
[Capacity and witness receipts](capacity.json) |
[Working supplement v48](../evidence_manuscript_v48/README.md).
