# Donor-Only Risk and Gain Calibration

**Outcome: no qualifying online controller.** This comparison keeps
the frozen nominated CV-to-OLS4 actions, eight observations, twelve
native forecast steps, original 2% CV-relative recording rule and online
credit budget. The seven OpenDD outer locations have development exposure
from earlier studies. No new thresholds were selected on their outcomes.

For an outer location and a calibration donor, the moment predictor is
trained without either site. For each source used to train this predictor,
its nomination ranker excludes the two held sites and the nominated
source itself. Risk cutoffs are computed on the corresponding training
locations only. Calibration fits use equal total weight per donor site.
They learn the sign probability and conditional positive magnitude of
all excess, easy excess and net easy change. The soft policy multiplies
probability by magnitude. The sign ablation only applies the resulting
penalty when probability is at least 0.5. A separate gain readout uses
the factorized predictions and per-prefix observed forecast errors.
Final outer inference uses only the frozen predictors and the fitted
donor readout, not the outer outcomes.

| Policy | Online edits | Online joint pass | Joint pass at 1,457 matched edits |
| --- | ---: | ---: | ---: |
| Frozen motion gain only | 1,457 | 61/606 | 60/606 |
| Frozen observed-prefix gain | 1,340 | 56/606 | 68/606 |
| Frozen factorized risk | 1,614 | 63/606 | 73/606 |
| Calibrated soft risk only | 1,616 | 58/606 | 75/606 |
| Calibrated sign-filtered risk only | 1,711 | 58/606 | 70/606 |
| Calibrated gain only, frozen factorized risk | 1,659 | 48/606 | 64/606 |
| Calibrated gain and soft risk | 1,681 | 59/606 | 51/606 |

Joint pass means original risk pass and positive conservative gain against
CV and OLS4. The matched readout uses the gain-only controller's exact
recording totals and token constraints; it sees all scores in each
recording without labels and is retrospective, not online. Soft risk
versus frozen factorization gives +0.001 pp [-0.049, +0.069] online and
-0.039 pp [-0.088, +0.004] matched. Intervals are 3,000 location-bootstrap
draws after averaging seeds; 606 views are 202 distinct recordings.

**Action probabilities do not control recording risk.** At the fixed
0.5 threshold, the calibrated easy sign has 0.65% false positives and
0.51% recall, versus 94.95% and 97.46% for the raw expected-excess zero
sign. These are different score types, not calibration curves for the
same classifier. Easy-sign Brier score is 0.1657 versus 0.1619 for donor
prevalence and 0.1588 for the original direct event logit. Low false
positives here result from rarely firing, not improved discrimination.
Gain MAE falls from 0.00995 to 0.00835 in normalized units; the online
gain readout nevertheless loses -0.080 pp [-0.150, -0.016] to the prefix
control. Its factorized-risk head is unchanged, but the prefix control
has no risk penalty, so the matched component attribution remains the
contrast to the frozen factorized arm: -0.094 pp [-0.175, -0.028].

**Why not a conformal guarantee?** Even for nested intervention sets,
selected easy-reference masses of 1 each and positive harms of 0.03,
0, and 0.06 give ratios 3%, 1.5%, and 3% at counts one, two, and three.
The original full evaluator passes only the middle set. This is tested
with nonidentity forecasts and complete labels. [Conformal Risk Control](https://openreview.net/pdf?id=33XGfHLtZg)
requires exchangeable, monotone losses for its main guarantee, unlike
this empirical ratio readout. [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html)
studies risk over accepted predictions; here a rejected edit still emits
the original forecast, and easy reference is part of the denominator.
Neither citation is a guarantee for this controller.

`fresh_run`: 105 triple-excluded nomination rankers, 63 pair-excluded
moment models, 117,600 native arm64 CPU Torch updates, and 21 donor
readouts over three seeds. `cached_verified`: upstream controls and
replay with zero neural updates and deterministic donor readout refits.
No risk-rule change, deployment, Stage5C, SMC, or independent confirmation.
[Readout](analysis.json) | [Setup](registration.json).

```sh
.venv-pytorch/bin/python scripts/run_m3w_donor_risk_readout.py
.venv-pytorch/bin/python scripts/run_m3w_donor_risk_readout.py --replay
```
