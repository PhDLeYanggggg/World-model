# Can an Optional OLS4 Bridge Reach Both Strong References?

The Hannover transfer failed for two separate reasons: mandatory edits
spent risk on unknown outcomes, and the fixed CV-centered radius was
too small to match OLS4 on average. Here I let each existing eligible
recipient abstain to exact CV or choose one action. I compare the six
frozen half-radius candidates, CV-to-OLS4 bridges at 25%, 50%, and 100%,
and their union. A separate control caps each new bridge at that row's
original half radius. Unrestricted bridges can spend more displacement;
the comparison is not matched in radius or number of edits.

I solve a per-recording binary optimization over this action bank. It
maximizes the smaller full-grid conservative gain against CV and OLS4
on the common observed CV error denominator. It retains the original
2% CV-relative all/easy bounds, signed easy loss limit, and nonempty
easy reference requirement. CV fallback is not counted as an edit or a
safety pass. Candidate benefits include triangle-inequality debits on
all missing steps. Risk is the unchanged recording rule, which debits
the peak disagreement for wholly unknown edited rows. I re-score every
claimed positive witness using the original risk code and both direct
gain calculations. The solver cannot see these labels at inference;
this whole experiment is a hindsight capacity audit, not a policy.

| Source | Optional frozen only | Add radius-clipped OLS4 | OLS4 bridge only, no cap | Union, OLS4 bridge no cap |
|---|---:|---:|---:|---:|
| OpenDD | 513/606 | 513/606 | 513/606 | 564/606 |
| Hannover | 0/9 | 0/9 | 9/9 | 9/9 |

Cells count re-scored witnesses with strictly positive utility against
both references and a pass on the original risk rule. OpenDD uses the
same seven exposed locations and 202 recordings; Hannover uses the
three previously scored clips from one provisional acquisition group.
Three seeds per view do not create independent recordings. None of the
four comparisons is unresolved at the declared numerical tolerance.

Optional frozen edits recover risk feasibility in 9/9 Hannover views,
but the maximin upper bounds rule out beating both references. Unclipped
bridges unlock all nine, with union witnesses gaining 6.18 to 10.04 pp
over OLS4. The clipped bridges unlock none. On OpenDD the union adds
51 views; 12 defined views still have no strictly positive bound, and
30 views still lack easy reference support. The clipped bank retains all
original witnesses without increasing the positive count. This isolates
the value of allowing larger corrections in this menu; it does not
show that a larger neural residual or a causal selector will succeed.

**Decision.** Stop optimizing tiny shape changes as the only route to
stronger accuracy. The next test should train a past-only gain/harm
policy for the expanded action menu on held-location development folds,
compare to motion-only and same-count controls, and preserve fallback.
The original evaluation, deployment and sealed DroneCrowd confirmation
are unchanged. Hannover is now exposed and is not a tuning source or
independent confirmation. No Stage5C or SMC.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_optional_reference_bridge.py
.venv-pytorch/bin/python scripts/audit_m3w_optional_reference_bridge.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_clipped_reference_bridge.py
.venv-pytorch/bin/python scripts/report_m3w_optional_reference_bridge.py
```

[Aggregate counts and full-output hashes](readout.json) |
[Optional bridge registration](registration.json) |
[Clipped control registration](clipped_registration.json).
Detailed solver diagnostics are generated locally and are not published.
