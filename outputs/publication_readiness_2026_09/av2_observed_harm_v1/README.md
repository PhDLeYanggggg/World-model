# AV2: conditional harm improves probability error, not release

The [support study](../av2_projection_support_v1/README.md) found
that knowing an easy reference exists does not establish acceptable
harm. I trained a separate head for **observed positive harm conditional
on support**, keeping the fixed position forecast, selected actors,
stepwise edit budget, support head and old gain/harm score unchanged.

## Training and comparisons

The head uses past motion, causal forecasts and the fixed projection.
On supported source-excluded donor projections, it learns whether
either original observed-positive-harm ratio exceeds 2%, with auxiliary
regression on the log of both ratios. The absence of support is not
a zero-harm target. This loss does not use missing-step completion or
signed easy net harm; the original scorer still requires those checks.

The three arms are the old transferred cost, the actor-support score
from the prior study, and that same actor-support score multiplied by
the new probability of no observed positive harm. The conditional
probability is only meaningful on supported views. Applying it as a
multiplier on all views is the policy ablation tested here.

The nested direction, EqMotion and action checkpoints each exclude
their scored donor city, calibration and held sources. The new head
uses 116,286 supported options across 18 fits, of which 94,148 have
observed harm. Seeds 17, 29 and 43 use four donor cities, a separate
calibration city and a held development city. Confidence is frozen
before scoring. The unchanged original all-qualified nonempty prefix
chooses each threshold on calibration only.

## Conditional estimation and equal-count choice

There are 340 held development scenarios, 1,020 seed views and
387 supported views. Among the supported views, 213 have observed
positive harm. The new probability's city-weighted Brier is 0.283
vs 0.316 for the pooled donor-rate baseline; paired change -0.0336
(95% city CI [-0.0570, -0.0087]). Mean fold AP for observed harm
is 0.637 for reverse actor-support score, 0.670 for the new probability,
and 0.677 for the reverse combined score. The risk tail is not reliable:
all five supported seed views with predicted harm below 0.2 actually
have observed harm. Five seed views are not five independent scenarios.

| Score, same top-five count (90 views) | Qualifies | Risk fails | Observed harm | No easy reference |
|---|---:|---:|---:|---:|
| Transferred cost | 31 | 45 | 35 | 10 |
| Actor support | 31 | 46 | 39 | 6 |
| Actor support + conditional harm | 28 | 45 | 24 | 21 |

Against actor support, adding harm reduces the observed-harm count by
15 but increases unsupported selections by 15. The paired effects
are -16.67 pp for observed harm (95% CI [-34.44, -2.22]) and
+16.67 pp for no reference ([+7.78, +25.56]). Qualification changes
by -3.33 pp ([-13.33, +4.44]). This is substitution, not evidence
that the combined score meets risk at the same intervention count.

Restricting both scores to actually supported views **after scoring**
retains the same 90-view diagnostic count. Observed harm falls from
41 to 35 (-6.67 pp [-24.44, +12.22]); qualification rises from
33 to 36 (+3.33 pp [-10.00, +14.44]). Neither contrast is conclusive.
A second hindsight diagnostic filters to known support and no observed
harm, then ranks by the unchanged actor-support score. At its common
available budget of 86 views it qualifies 58, versus 30 for actor
support. This is not a deployable rule or an optimal upper bound.

## Actual release

| Score | Qualifying calibration prefix views | Held releases / qualifies | Held risk fails | Held dual-floor fails |
|---|---:|---:|---:|---:|
| Transferred cost | 6 | 6 / 3 | 3 | 1 |
| Actor support | 6 | 8 / 5 | 3 | 2 |
| Actor support + conditional harm | 9 | 17 / 4 | 8 | 9 |

Of the eight risk failures for conditional harm, five have observed
positive harm and three are unsupported. The 17 releases cover 1.67%
of held views. Their known ADE/FDE are 0.13195/0.27382, with worst
released p95 ADE 1.119 in native coordinates. Same-set conservative
CV/OLS4 gain is +3.097/+2.700 pp. Full policy gain with CV fallback
is +0.0450 pp (paired city 95% CI [-0.0086, +0.1436]).
Neither gain replaces the original risk condition. No new release
policy is deployed.

The next test should enforce predicted support as a separate constraint
before ranking by conditional harm and gain. It should keep the exact
risk rules and compare the same position forecasts at matched counts.
Do not interpret fewer observed violations as progress when missing
references increase by the same amount.

## Reproduce

Native arm64 CPU, four Torch threads, zero workers. Fit and freeze took
44.66 s (783 MB peak RSS), 400 updates per head. Atomic checkpoints
every 50 updates save model, optimizer and RNG. Rerun `fit` to resume;
the heartbeat identifies the active fold and PID. Score and audit
reuse existing held scalar and support receipts without reopening raw
held outcomes. No new CREATE job was needed.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_observed_harm.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_observed_harm.py score
.venv-pytorch/bin/python scripts/audit_m3w_av2_observed_harm.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2_*.py tests/test_m3w_cv_anchor_budget.py tests/test_m3w_cv_joint_feasibility.py
```

[Freeze](freeze.json), [calibration](calibration.json), [results](summary.json),
[conditional probability audit](conditional_calibration.json),
[supported-only ranks](supported_only_ranks.json).
No official AV2 validation/test, DroneCrowd confirmation, manuscript,
submission, deployment, Stage5C or SMC change.
