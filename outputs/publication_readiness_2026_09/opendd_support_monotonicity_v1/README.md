# Adding Support Is Not a Monotone Risk Operation

I replayed the fixed OpenDD gain and easy-support actions on the
previously exposed development roster. The comparison is gain alone
versus gain plus the original support action, restricted to the 345
recording-seed views where the second slot was used. The first actions
and candidate forecasts did not change. This is an action-count
diagnostic, not a count-matched utility comparison or a new policy.

| Original state | Undefined after support | Over 2% after support | Pass after support |
| --- | ---: | ---: | ---: |
| Undefined | 86 | 66 | 102 |
| Over limit | 0 | 38 | 1 |
| Pass | 0 | 17 | 35 |

Of the 91 views with the selected-easy ratio defined before and
after, the ratio increased in 30, decreased in 31, and did not change
in 30 (tolerance `1e-12`). Seventeen moved from at most 2% to above 2%;
one crossed in the other direction. The analogous all-row ratio has
28 upward and 13 downward threshold crossings. These are matched
finite observations, not estimates based on 345 independent videos.

For a defined ratio, adding an action with harm increment `h`
and easy reference increment `b` changes `H/B` to
`(H+h)/(B+b)`. The direction depends on the sign of `hB-Hb`,
not on the action count alone. For `B=0`, the first ratio is
undefined; no numeric zero or successful calibration is assigned.
This explains why the unchanged recording rule does not in general
form a monotone family merely by ordering policies by action count.
The [Conformal Risk Control theorem](https://openreview.net/attachment?id=33XGfHLtZg&name=pdf)
uses monotone, bounded, exchangeable losses and controls expected loss;
it is not an automatic certificate for this two-policy recording test.
The 2% cap is within-recording degradation, not a 2% probability
that a new recording will fail.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_opendd_support_monotonicity.py
```

`fresh_run`: exact risk transition audit. `cached_verified`:
original causal predictions and matched schedules. No new model
training in this audit. The original risk rule, deployment, and sealed
DroneCrowd confirmation are unchanged.
