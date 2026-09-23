# A Less Conservative Scene-Risk Bound, Not a New Safety Certificate

## Material Passport

2026-09-23. `fresh_run`: targeted primary-paper reading, an isolated numerical
implementation, analytical sample-count calculations, 27 exact two-point null
checks, 20,000 synthetic family-wise replications and a joint-action counterexample.
`cached_verified`: exact replay of the new diagnostic. `not_run`: any real
forecast, model training, calibration, independent confirmation or deployment.
No SDD/DroneCrowd coordinates, labels, forecasts or checkpoints were opened.

The previous round established annotation-level window support and past/label
separation. That was material progress, but it did not establish independent
physical sites. While source admission remains unresolved, this round improves
the statistical feasibility implementation without assigning new data roles.
The legacy calibration runner, its Hoeffding screen, risk functionals and all
frozen experiments remain unchanged. This alternative is **not wired into the
scientific protocol** and must not be used to relabel an earlier failed gate.

## Source-Checked Method

Learn then Test treats calibration as testing a fixed family of risks. Its
Proposition 1 supplies Hoeffding-Bentkus p-values for bounded losses; Proposition
2 controls family-wise error with Bonferroni. We implement these existing results,
not a new M3W theorem. Independent, identically distributed calibration units and
a calibration-independent policy family remain requirements.
[Angelopoulos et al., arXiv:2110.01052v5, sections 1.1 and 2.1-2.3](https://arxiv.org/abs/2110.01052v5).

For n scene losses in [0,1], observed mean r and candidate population mean a:

```text
p_HB(a) = min(1,
              exp(-n * KL(min(r,a) || a)),
              e * BinomialCDF(ceil(n*r); n,a)).
error_level = delta / (number_of_policies * number_of_risks).
```

Invert the test for an upper mean bound. Fractional scene losses require the
ceiling and e factor in the binomial term; treating them as exact Bernoulli
counts is not this method. The implementation retains a conservative numerical
bracket and does not give learned zero-intervention samples the analytical
exemption of an identically baseline-only policy. The paper also covers more
powerful testing orders; this diagnostic retains simple simultaneous correction.

Conformal Risk Control's main theorem instead concerns expected risk for a
monotone family of loss functions. It is not automatically a high-probability
certificate for arbitrary joint-action thresholds.
[Angelopoulos et al., arXiv:2208.02814v4, sections 1.1 and 2.1](https://arxiv.org/abs/2208.02814v4).
Read scopes and bibliographic identities are recorded in [analysis.json](analysis.json);
this was a targeted check, not an exhaustive literature review or full-paper review.

## Fresh Feasibility Arithmetic

All numbers below assume **zero observed bounded loss**, independent same-population
scenes, delta=0.05 and the indicated frozen family. They are optimistic arithmetic,
not achieved losses or a recommendation to change tolerance. In particular,
**a bounded-risk tolerance of 0.02 is not 2% relative easy-case ADE degradation**.

| Policies | Risks | Illustrative bounded-risk tolerance | Existing Hoeffding scenes | HB scenes |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.02 | 3,745 | 149 |
| 1 | 1 | 0.10 | 150 | 29 |
| 1 | 1 | 0.20 | 38 | 14 |
| 5 | 2 | 0.02 | 6,623 | 263 |
| 5 | 2 | 0.10 | 265 | 51 |
| 5 | 2 | 0.20 | 67 | 24 |
| 10 | 2 | 0.02 | 7,490 | 297 |
| 20 | 2 | 0.02 | 8,356 | 331 |

For one policy and one risk, the zero-loss HB upper bound is
`1 - delta^(1/n)`. It is 0.3930 at n=6, 0.0950 at n=30, 0.0419 at n=70 and
0.0264 at n=112. These example counts are **not assignments of DroneCrowd clips
or source-described scenarios to calibration**. Even assuming all 112 clips were
independent would not certify the illustrative 0.02 risk with this screen.
Actual calibration would use only its independently reserved portion and may
observe nonzero loss. Neither statement is an impossibility theorem for all
possible risk definitions or statistical methods.

This changes the planning interpretation of the older support table: thousands
of scenes were a limitation of that particular loose bound, not an established
data requirement of every method. It does not repair the absence of independent
scene identities, fitting ancestry or calibration/confirmation separation.

## Numerical and Synthetic Checks

- **42 scoped tests passed**, including 19 new cases. Coverage includes limits,
  invalid inputs, fractional losses, multiplicity, bounds rescaling, duplicate
  cluster IDs, fitting overlap, missing values, numerical inversion and the
  existing intervention/support checks. The frozen legacy screen is reused only
  for validation/reference comparison, not modified.
- **27 exact null checks passed** over binary and nonbinary two-point bounded
  losses, three sample sizes and three error levels. These are finite fixtures,
  not a proof for all distributions; theoretical validity comes from the cited
  assumptions/result, not a passing test suite.
- With 300 synthetic independent units, ten frozen policy-risk pairs, true risk
  0.02001 and illustrative tolerance 0.02, exact independent-pair false acceptance
  is 2.3012%; perfectly identical pairs give 0.2325%. Their general union bound is
  2.3254%, below delta=5%. The 20,000-replicate simulations give 2.255% and 0.215%.
  Correlation between policies is allowed; dependence between calibration units
  is a different issue. These are not M3W deployment failure rates.
- A dependence counterexample repeats each of six Bernoulli scene outcomes
  1,000 times. At true risk 0.20, tolerance 0.10 and delta=0.05, the valid
  scene calculation falsely accepts with probability 0 in this fixture; treating
  6,000 copies as independent falsely accepts with probability **26.2144%**.
  A tighter bound does not make window pseudo-replication valid.

The numerical audit replays exactly. Analysis SHA256:
`683a31959109b8e38f105b14787a3dcc1ccb39cbdfe8d3847b5a114d8d17bd2e`.
It uses NumPy/SciPy only, with no Torch fit, GPU or CREATE job.

```bash
.venv-pytorch/bin/python scripts/audit_m3w_scene_risk_bounds.py --verify
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_scene_risk_bounds.py tests/test_m3w_joint_intervention.py tests/test_m3w_calibration_support.py
```

## Joint Risk Need Not Be Monotone

An executable two-agent, one-step witness uses baseline coordinates (0,0)/(4,0)
and candidate coordinates (4,0)/(0,0). Choosing both candidates, one candidate,
or neither produces pair distances 4, 0 and 4. A distance-below-one proxy is
therefore 0, 1, 0 even though interventions decrease from two to one to zero.

This is an elementary constructed counterexample, not a physical collision test,
not measured M3W performance, and not an assertion that every risk functional is
nonmonotone. It shows why a joint selector cannot inherit a monotonicity assumption
from switch counts alone. A fixed-family risk test is compatible with such a
nonmonotone family; its empirical benefit still requires the approved real study.

## What This Does Not Solve

The empirical easy criterion remains at most 2% relative degradation in every
evaluated site/seed, with exact-zero cases handled separately. Neither a clipped
mean harm risk nor a harm-event probability certifies that ratio or a worst-site
property. Missing future labels still cannot be silently counted as successes.
Changing a risk definition, aggregation or tolerance needs explicit protocol
approval; this diagnostic changes none of them.

The core contribution must remain whether learned gain/harm and joint decisions
improve forecasts at matched intervention or risk budgets. Implementing LTT/CRC
is not itself novelty and does not rescue the failed neural comparisons.

Next priorities are unchanged in order: resolve independent physical-site/camera
evidence, pin the annotation/observation protocol, approve separate scientific
roles, and then choose the applicable calibration method before reading its
outcomes. The image-audit authorization remains pending. No repeated question,
download, test-set threshold tuning or new SDD search was initiated.

The project remains a pixel/dataset-local, raw-frame 2.5D trajectory/world-state
study, not a verified metric/seconds, true-3D, foundation or submission-ready
result. No new best model, safety certificate, independent confirmation,
Stage5C execution or SMC activation is claimed.
