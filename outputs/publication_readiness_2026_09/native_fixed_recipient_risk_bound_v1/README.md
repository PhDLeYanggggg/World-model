# Most Fixed Recipient Views Cannot Be Repaired By Reranking

The diagnostic gives the optimizer access to labels and minimizes
each risk component separately over the frozen motion recipients.
For each component it can keep the selected motion path or use
any of the three legal learned replacements. It cannot change
recipients or select a different motion alternative. Where even
the best component exceeds the original rule, no joint choice
in that slate can pass. This is a necessary-condition proof,
not a causal policy, and the remaining views are not certified.

| Neural replacement counts | All-harm bound >2% | Easy-harm bound >2% | Signed easy bound >2% | Missing reference | Any reason / 102 |
|---|---:|---:|---:|---:|---:|
| Frozen exact count per query | 86 | 90 | 16 | 15 | 99 |
| Free within the same recipients | 77 | 83 | 3 | 15 | 93 |

Reasons can overlap. The 99 views cover 33 distinct recordings;
the 93 cover 31. Both span all 14 source groups. Three and nine
views remain unruled out, respectively, not proved feasible.
The exact-count minima independently use the k smallest cost
increments after each row's best legal neural option is found.
The free-count minima can keep the anchor whenever it is better.
The selected reference masses stay fixed because the same rows
continue to differ from CV. No label-aware choices are published.

This explains why a better increment ranker cannot, by itself,
rescue most of this operating point. The earlier all-recipient
hindsight result is not contradicted: that experiment could change
recipients and select from a broader bank within query caps.

~~~sh
.venv-pytorch/bin/python scripts/audit_m3w_fixed_recipient_risk_bound.py score
.venv-pytorch/bin/python scripts/audit_m3w_fixed_recipient_risk_bound.py replay
~~~

The first run produced the aggregate but failed to record its receipt
because of a missing import. The revised registration fixes the
receipt only; the verified rerun reproduces the exact original
aggregate. No external access, threshold change, Stage5C or SMC.

[Protocol](protocol.md) | [Revised registration](registration_v2.json) |
[Metrics](analysis.json) | [Replay](replay_execution.json)
