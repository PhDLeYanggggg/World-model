# Which Track Should Receive a Support Edit?

I kept the original 598 gain edits and the 345 second-slot queries
fixed. Every two-slot policy made exactly 943 edits with the same
amplitudes, 0.25 and 0.10 respectively. I trained one head per held
site and seed to estimate three quantities for every legal recipient:
conservative CV gain, local feasible support, and excess loss.
The head uses current observed features and differences from the
frozen easy recipient. Only the training labels use future outcomes.
Donor forecasts and easy heads exclude both donor and evaluation
sites. Donor menus use all qualifying queries, not the realized first
gain edit; this is a local surrogate, not a recording-level witness.

The guarded policy keeps the easy recipient unless an alternative
has higher predicted support, no worse predicted gain, and no worse
predicted harm. The unguarded and easy-only arms use the same trained
head. No thresholds were selected on held readouts. These are OpenDD
development comparisons, with 202 previously exposed recordings,
seven sites and three seeds.

| Second-slot choice | Changed | Below easy threshold | Undefined | Risk pass | Joint pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen easy | 0 | 0 | 295 | 172 | 138 |
| Frozen gain | 235 | 227 | 341 | 142 | 113 |
| Learned gain | 233 | 224 | 342 | 137 | 110 |
| Learned support, all legal | 100 | 92 | 303 | 166 | 128 |
| Learned support, easy only | 8 | 0 | 294 | 173 | 139 |
| Guarded support | 27 | 25 | 296 | 172 | 138 |

The primary guarded comparison yielded **+0.00160 pp
[-0.00019, +0.00499]** against frozen easy. Its joint count did not
change. Learned gain improved **+0.03710 pp [+0.01181, +0.06489]**
against frozen easy, but lost 28 joint passes and had 47 more
undefined views. Learned gain versus CV was +0.16571 pp
[+0.03519, +0.28478]. There is no direct interval against frozen gain
in this experiment; the two estimates versus CV do not establish
that one gain head is better than the other. Intervals average
seeds within each site and resample sites 3,000 times. The conservative
contrasts need not be transitive. None passes the full risk rule.

The wide support rank used 92 recipients the old rule would not admit,
yet selected fewer observed easy recipients (228 vs 242) and fewer
local feasible events (97 vs 110). The guard limited the loss of
support but did not recover extra passes. The preceding exact
same-query diagnostic adds 40 potential joint passes over easy;
31 of those are at one site and eight at a second site. This limits
what a better ranker at the same clock can do across the roster.
A stronger next comparison should train for the **increment in
whole-recording support given the first edit**, not just rank
local events. Coverage and gain should be reported separately.

This is a cost-sensitive decision relative to an existing forecast,
related to [learning to defer](https://proceedings.mlr.press/v119/mozannar20b.html).
Unlike [regression with a reject option](https://proceedings.neurips.cc/paper/2020/file/e8219d4c93f6c55c6b10fe6bfe997c6c-Paper.pdf),
the comparison here does not abstain or threshold absolute uncertainty:
it preserves a fixed intervention count and evaluates an entire
recording against CV and OLS. Those distinctions motivate the experiment,
not a new theoretical guarantee.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_support_admission.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_support_admission.py --replay
.venv-pytorch/bin/python scripts/freeze_m3w_dronecrowd_calibration_roles.py verify
```

`fresh_run`: 21 arm64 Torch fits and 10,500 updates. The runner saves
resumable checkpoints and heartbeat every 100 steps. `cached_verified`:
the trajectory bank, original clock and zero-update replay. `not_run`:
independent confirmation. The 15/30 DroneCrowd roles and original 2%
and undefined-risk rules are unchanged. No deployment promotion,
Stage5C execution, SMC or paper submission.
