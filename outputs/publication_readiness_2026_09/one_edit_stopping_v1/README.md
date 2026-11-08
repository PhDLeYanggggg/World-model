# Can an Online Stop Rule Find a Better First Edit?

The prior exact one-edit audit found 120 feasible views at the first
eligible query and 441 at any eligible query. I tested whether a
past-only stop rule can turn that hindsight gap into forecast gain.
The candidate paths, OpenDD development roles, eight observed and
twelve requested native steps, CV/OLS4 comparators, missing-step
debits and original 2% recording check are unchanged.

## Fit and Compare

The same new value head ranks candidates in all three arms. It trains
on every eligible window, with balanced draws by source, recording,
window and candidate. The head predicts the original one-edit
recording qualification as a training target; the full recording
offset and future outcomes are never inputs. For each held location,
the two tunable arms use out-of-fold predictions from the six donor
locations, excluding both the held site and the predicted donor from
each producer. Pool three seeds to choose one setting per held site.
Select by macro-average donor joint-pass rate, tie by conservative
gain against CV and fixed grid order.

- **Immediate:** edit at the first eligible query.
- **Fixed delay:** skip the donor-selected number of eligible queries.
- **Learned stop:** edit at the first query whose top score reaches the
  donor-selected threshold. No forced edit at the last query.

The fixed delay chooses 0, 1 or 2 skipped windows; learned stop chooses
0.10 at six sites and zero at one. These scores are not independently
calibrated risk probabilities. A recording with no edit retains CV;
its selected-risk denominator is undefined under the original rule.

| Policy | Edits | Risk passes | Undefined easy | Joint passes / 606 | Short joint passes / 105 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Immediate | 606 | 96 | 427 | 63 | 3 |
| Fixed delay | 597 | 93 | 438 | 60 | 3 |
| Learned stop | 580 | 116 | 399 | 70 | 6 |

Despite seven more joint passes, learned stop loses **-0.422 pp
[-1.037, -0.041]** in conservative utility to immediate on the
full roster. On the **580 views where both edit once**, it still
loses **-0.417 pp [-1.035, -0.036]**. On the 574 views matched with
fixed delay, the effect is **-0.233 pp [-0.813, +0.144]**. Matching is
retrospective and is only a utility comparison; risk is always
evaluated on the full roster. The intervals use 3,000 paired location
draws after averaging seeds. The 606 views are 202 recordings,
not 606 independent samples.

## Why Selecting More Passes Is Not Enough

In every donor fold where the primary selection chooses a nonzero
threshold, it increases the donor pass rate at the cost of CV gain.
A secondary check requires a threshold to preserve both donor
outcomes. It returns to immediate in six of seven folds. The held
joint count remains **63/606** and conservative utility relative
to immediate is **-0.014 pp [-0.042, 0]**. This is a post-primary
development diagnostic with donor-only threshold choices, not a
reclassification of the primary result.

The exact one-edit capacity still reproduces: 120 at the first
eligible query, 441 at any query and 321 later-only. The availability
of a better action does not imply the learned score knows when
to wait. No operating policy qualifies. Do not promote another
threshold variant as a remedy; a stronger route needs candidate or
risk-target improvements that preserve utility on held locations,
followed by independent calibration before any confirmation.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_one_edit_stopping.py
.venv-pytorch/bin/python scripts/run_m3w_one_edit_stopping.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_one_edit_guard.py
```

84 fresh native arm64 Torch fits: 21 outer fits and 63 double-excluded
fits, 33,600 optimizer updates, four compute threads, no data workers,
checkpoint and heartbeat every 100 steps. The primary score replay
reused all 84 models with zero updates and identical aggregate scores.
29 targeted tests passed. No CREATE submission, deployment, metric or
seconds claim, Stage5C or SMC. The 30 DroneCrowd confirmation groups
remain sealed.

For a standalone statistical replay, build the anonymous scalar bundle:

```sh
.venv-pytorch/bin/python scripts/review_m3w_one_edit_stats.py \
  outputs/publication_readiness_2026_09/one_edit_stopping_v1 \
  --package data/stage_cvpr2027_experiments/one_edit_stopping_v1/anonymous_scalar_replay.zip
```

Verified from a clean extraction: `python replay.py data.json` needs
only NumPy. It recomputes the five bootstrap intervals, pass counts,
donor-setting choices and expected 84 exclusion roles from scalar
receipts. It does not reproduce model fitting, verify raw training
membership or add independent evidence. The local ZIP SHA-256 is
`3fd26d5fc203d068d9185d9eec6e790f40cf07d40bca14cd2fc2149307274848`.

[Registration](registration.json) | [Scores](analysis.json) |
[Checkpoint receipts](head_freeze.json) | [Secondary diagnostic](donor_guard/analysis.json).
