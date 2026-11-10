# How stable are the available OpenDD exchanges?

I held the 606 views, 780 edits, and original risk and utility checks
fixed. Of 7,198 legal exchanges, 431 change an incumbent failure
into a pass. They cover 121 recording-seed views.

| Requirement on at least one rescue | Views | Candidate actions |
| --- | ---: | ---: |
| Full-roster pass | 121 | 431 |
| At least 0.25 pp of worst-risk reserve | 116 | 397 |
| At least 0.50 pp of worst-risk reserve | 108 | 373 |
| At least 1.00 pp of worst-risk reserve | 98 | 308 |
| Risk passes after each of three background deletions | 105 | 271 |
| Risk and utility pass after each deletion | 93 | 216 |
| Stable, full gain > 0.10 pp and risk reserve >= 0.50 pp | 27 | 44 |
| Selected easy support in at least two separate queries | 39 | 152 |

The deletions remove each chronological third in turn, while keeping
both queries involved in the exchange. This holds the edited-row count
equal between the two plans on each retained subset. All 431 full-roster
witnesses have three supported deletions. Across their 1,293 deletions,
1,003 pass, 132 have undefined risk, 111 have insufficient utility and
47 violate a measured risk condition. Inside the model's fixed 101-view
intervention mask, 31 of the 44 rescuable views still have a stable
rescue.

Most rescuable views are not explained by an infinitesimal distance
to the original 2% threshold. The opportunities do shrink when I ask
for both stable outcomes and material gain. This stress test retains
the action endpoints and is not evidence that the effect replicates
on different future trajectories. Its role is to distinguish a
context-sensitive pass from a pass that can survive removing
unchanged parts of the recording.

[Protocol](protocol.md) | [Grouped readout](readout.json)

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_exchange_stability.py
```
