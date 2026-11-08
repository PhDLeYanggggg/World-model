# Direct Signed Bridge Risk: More Interventions, No Reliable Gain

I kept the original gains, candidate paths, neural-action risk estimates
and query decoder fixed. For the three analytic CV-to-OLS4 bridges I
trained two signed values: all harm minus 2% of its reference mass, and
easy harm minus 2% of its reference mass. The direct rule admits a query
combination when the corresponding predicted sums are nonpositive.
This is a prediction screen, not a recording-level risk certificate.

The donor-calibrated rule adds an equal-location mean of per-location
60th-percentile out-of-fold residuals, separately for each bridge and
risk component. An inner risk fit and its easy cut exclude both the
outer scored location and the inner calibration location. All risk
fitting and calibration uses only analytic candidates; no learned
forecaster or original gain head enters those calculations. The full
risk head uses the six donors, with neither its fit nor its correction
using the outer location. The 60th percentile is fixed, not tuned, and
does not establish coverage. The frozen original gains are used only
for outer inference. This remains OpenDD development.

There are 147 native Torch fits and 58,800 updates (seven outer locations,
three seeds, six inner plus one full head per cell). Full scores use the
unchanged per-recording CV risk test and separate full-grid gain bounds.

| Policy | Edits | Risk passes | Dual-positive passes /606 |
|---|---:|---:|---:|
| Original predicted risk | 64 | 4 | 4 |
| Direct signed risk | 1,227 | 59 | 37 |
| Direct, matched-count independent | 1,227 | 111 | 63 |
| Donor residual calibration | 1,916 | 65 | 36 |
| Calibrated, matched-count independent | 1,916 | 165 | 98 |
| Fixed tangent, unrestricted count | 30,075 | 522 | 427 |

The calibrated head uses 1,440 bridges: 1,432 at 25%, eight at 50%,
none at 100%. It has 397 undefined easy-risk views, not safe abstentions.
Its conservative utility versus CV is +0.0337 pp [-0.0211, +0.1003];
the direct comparison against tangent is -0.3556 pp [-0.4571, -0.2574].
Versus its count-matched independent control it is +0.0249 pp
[-0.0247, +0.0856] but has fewer risk passes. Intervals resample seven
locations 3,000 times after seed averaging. Matching counts does not
match radius, recipient or action family.

## Admission diagnosis

For 9,595 legal bridge actions with predicted positive increment
against both references, 5,497 have nonpositive observed excess in
both components individually. This is a candidate-level check, not a
recording certificate or a claim about independence.

| Risk head | Correctly admitted | Incorrectly admitted | Incorrectly rejected |
|---|---:|---:|---:|
| Original | 38 | 12 | 5,459 |
| Direct | 830 | 525 | 4,667 |
| Donor calibrated | 1,164 | 711 | 4,333 |

Replacing only the analytic bridge risk by its outcome-derived value
raises dual-positive passes to **221/606**. This is diagnostic, not an
inference feature. The refit alleviates abstention but not the ordering
of useful, safe admissions. Unconditional residual correction admits
more true cases and more false cases, with no gain in dual-positive
passes. A better next test should calibrate on the **selected** actions
and explicitly model easy support or a safe-action hurdle, with proper
donor separation. Do not loosen the original 2% rule or fit to Hannover.

No deployment, independent confirmation, Stage5C or SMC.

```sh
.venv-pytorch/bin/python scripts/run_m3w_signed_bridge_risk.py
.venv-pytorch/bin/python scripts/run_m3w_signed_bridge_risk.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_signed_bridge_admission.py
```

[Registration](registration.json) | [Training and policy results](analysis.json) |
[Admission diagnosis](admission.json).
