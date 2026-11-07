# Does Keeping More Joint Path Assignments Help?

I kept the eight-person shortlist, original motion models, fixed costs
and per-query switch counts. Rather than commit to one path per person
before comparing pairs, I enumerated all legal assignments. The exact
score includes the same OLS4-relative graph energy as the prior joint
decoder; the matched additive control drops only the cross term.
The 10,770,597 assignments include 8,342 that satisfy the original
predicted local screen. These are not independent evaluation units.

| Operating choice | Switches | Original recording passes / 102 | All-risk upper bound exceeds 2% | Undefined easy risk |
| --- | ---: | ---: | ---: | ---: |
| Original single choice then screen | 72 | 4 | 12 | 88 |
| Expanded additive, same active queries | 72 | 3 | 13 | 88 |
| Expanded joint, same active queries | 72 | 4 | 12 | 88 |
| Screen all assignments before choosing | 2,194 | 4 | 40 | 62 |

**Matched result.** At the same active queries and counts, exact joint
search changes one query relative to the original. Its paired
conservative ADE contrast is -0.000024 pp [-0.000072, 0.000000].
Coupling changes two queries compared with the matched additive search,
which is too little evidence for a separate interaction contribution.
At the larger ungated matched budget of 9,675 switches, exact search
changes 615 queries. Observed pair ADE improves by 0.052 pixels
[+0.027, +0.081], but its full-grid completion lower contrast is
-0.094 pixels [-0.252, +0.008]. Conservative trajectory utility is
-0.010 pp [-0.082, +0.052] versus the original ungated decoder.

**Screen before choosing.** This comparison deliberately lets the
activation count change. It improves conservative utility over CV
by +0.045 pp [+0.009, +0.094] but remains below OLS4 by -9.779 pp
[-21.964, -0.515]. The 2,194 interventions are 0.583% of all target
instances. More queries satisfy the predicted screen, but more
recordings violate the actual total-risk limit; four pass overall
as before. The signed easy net cap alone passes and cannot replace
the other risk and support requirements.

**Decision:** Keep the original policy. This test shows that the
placement of a predicted local screen affects how much the model
intervenes. It does not establish that joint search controls recording
risk. The next test should calibrate cost on the actual
post-search actions, not on the pre-search population or a single
old proposal, without reopening the independent confirmation split.

All decisions are frozen before the scoring phase. The 102 views are
34 exposed recordings across three forecaster seeds, grouped into 14
sources for 3,000 paired bootstrap draws. Distances use image pixels
and eight observed/twelve forecast annotation steps. No external
readout, deployment change, Stage5C or SMC.

[Protocol](protocol.md) | [Aggregate results](analysis.json) |
[Freeze](freeze_execution.json) | [Score](score_execution.json) |
[Replay](replay_execution.json)

```sh
.venv-pytorch/bin/python scripts/run_m3w_joint_path_portfolio.py freeze
.venv-pytorch/bin/python scripts/run_m3w_joint_path_portfolio.py score
.venv-pytorch/bin/python scripts/run_m3w_joint_path_portfolio.py replay
```
