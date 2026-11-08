# Fixed Menu and Streaming Budget Capacity

This is a hindsight diagnostic on seven previously exposed OpenDD
development locations, with 202 distinct recordings. The complete analytic
bank is deterministic; its 202 results are repeated only to align with
the 606 recording-seed views for the learned nested menu. No inference
performance, deployment or independent confirmation follows.

Every arm retains the three analytic CV-to-OLS4 bridge amplitudes as its
source, the per-query cap of two edits (leave at least one agent on CV),
the total quota of one edit per five eligible queries, and the original
CV-relative 2% recording risk. The nested menu fixes the first and
second nominees and requires that using the second also uses the first.
The chronological arms also start with zero credits, earn 0.2 per eligible
query, store at most two, and spend one per edit.

| Candidate menu | Total quota only | Total quota + chronological credit |
| --- | ---: | ---: |
| All legal bridge actions | 423/606 | 369/606 |
| Frozen nested recipients and amplitudes | 383/606 | 330/606 |

Each number counts a verified witness that passes the original risk rule
and gives strictly positive conservative gain over CV and OLS4. Timing
alone costs 54 full-menu witnesses; restricting candidates at the same
timing costs 39 verified witnesses (38 ruled out, one unresolved).
In the strict nested/chronological arm, 30 views have no easy reference,
78 have zero quota, 14 have no legal easy edit, 105 are risk-infeasible,
48 have an upper bound at or below the positive-gain criterion, and one
is unresolved. A timeout or unverified incumbent is never a negative
certificate. See the [paired counts](analysis.json).

The original online recording-aware head passes 57/606. Its 273-view gap
to the strict, independently rescored witness count is larger than the
combined losses from menu and credit restrictions (93 views). Labels
permit this hindsight gap; there is no implication that a causal policy
can learn all of it. Test budget-consistent training before changing the
quota or risk rule.

**Exact credit formulation.** Let `b[t]` be retained credit after spending,
`a[t]` the added credit, `k[t]` the number of edits, and `C` the storage cap.
The linear constraints are `b[t] >= 0`, `b[t] + k[t] <= C`, and
`b[t] + k[t] <= b[t-1] + a[t]`, starting from zero. The actual chronological
balance is the maximum available balance; by induction it is at least
any balance satisfying the inequalities, so every integer edit plan
admitted by the solver can be replayed chronologically. The reverse
direction uses the actual retained balance. The code uses units of one
fifth of an edit to replay every returned witness exactly.

The capacity calculation is `fresh_run`. Candidate checkpoint inputs are
hash-verified; no training, remote job, or confirmation read was needed.
The DroneCrowd 15/30 split, deployment, Stage5C, and SMC remain unchanged.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_streaming_menu_capacity.py
```
