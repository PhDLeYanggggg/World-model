# Causal Reference Bridges: The Learned Risk Screen Abstains Too Often

I trained a small action head to choose among the six frozen half-radius
paths and three larger CV-to-OLS4 bridges. It observes only eight past
positions and the causal candidate paths. It regresses full-grid gain
over CV, the change in gain against OLS4 relative to CV fallback, and
the positive harm and reference masses for all and easy rows. The
query rule can make up to two edits and keeps at least one CV fallback.
It searches single edits and pairs using predicted risk; this prediction
screen is not the official, recording-level 2% certificate.

I ran 21 native Torch fits, 14,700 updates, three seeds and seven held
OpenDD locations. No evaluated location contributes to the head,
refiners or thresholds. Donor producers also exclude their own location;
donor refiners exclude the outer location, not necessarily the donor.
No Hannover evaluation, held-location tuning or confirmation access.

| Policy | Edits | Lower gain vs CV (pp) | Original risk passes | Positive vs both and risk pass |
|---|---:|---:|---:|---:|
| Fixed tangent (larger switch count) | 30,075 | +0.3057 | 522/606 | 427/606 |
| Independent predicted utility | 10,898 | +0.0236 | 416/606 | 268/606 |
| Joint predicted risk | 64 | +0.0010 | 4/606 | 4/606 |
| Independent, matched query counts | 64 | +0.0008 | 4/606 | 4/606 |
| Motion-ranked, matched query counts and action families | 64 | +0.0004 | 10/606 | 8/606 |
| Joint, OLS4 bridges only | 22 | +0.0030 | 3/606 | 1/606 |

Joint edits just 53 query-seed cases, eleven with two edits. It takes
only one bridge, at 25%; it does not use the larger options. Its direct
gain over matched independent is +0.00016 pp [-0.00015, +0.00063].
The motion control ranks recipients by past-only CV-to-OLS4 backtesting,
while inheriting the same query counts and action-family counts;
if reassignment is infeasible it keeps the joint assignment. Neither
count match guarantees equal displacement or realized risk. Independent
gain versus CV is +0.0236 pp [-0.0906, +0.1191]. The 3,000 bootstrap
draws resample seven locations after seed averaging. The joint rule has
602 undefined easy-risk views, not 602 safe abstentions.

## Separate the query cap from prediction error

An exact hindsight optimization now enforces the *same* maximum per
query as the learned policy, in addition to the unchanged recording rule.
It verifies positive utility against both references in **561/606**
views (unlimited oracle: 564/606). Thirty views lack easy support and
15 more have no strictly positive maximin bound. The count cap is not
the main reason that the learned policy fails.

| Same query decision rule | Edits | Risk passes | Dual positive and pass |
|---|---:|---:|---:|
| Predicted gain and risk | 64 | 4/606 | 4/606 |
| Replace gain only by outcome-derived values | 91 | 8/606 | 8/606 |
| Replace risk only by outcome-derived values | 10,867 | 479/606 | 347/606 |

These replacements are counterfactuals for diagnosis, not inference
features or deployable models. They show that the fitted risk screen
blocks useful edits much more than the fitted gain estimates do in this
particular decision rule. Correcting risk alone still falls short of the
exact ceiling and does not imply all views could pass with a learned
head. The next experiment should train directly on the signed excess
relative to the fixed 2% boundary and use donor-only calibration,
rather than weaken the official rule or enlarge the model.

No deployment, original protocol change, Stage5C, SMC or independent
confirmation. The original DroneCrowd confirmation groups stay sealed.

```sh
.venv-pytorch/bin/python scripts/run_m3w_reference_bridge_policy.py
.venv-pytorch/bin/python scripts/run_m3w_reference_bridge_policy.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_bridge_policy_bottleneck.py
```

[Held-location results](analysis.json) | [Exact cap and counterfactuals](bottleneck.json).
