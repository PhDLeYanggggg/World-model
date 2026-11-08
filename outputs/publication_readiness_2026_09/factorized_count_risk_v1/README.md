# Factorized Risk: Better Matched Pass Count, Not a Qualifying Policy

On the same seven previously exposed OpenDD locations, the preceding
[fixed-count attribution](../support_value_attribution_v1/README.md)
found that true harm was more useful to the joint pass count than true
easy support alone. We therefore estimate nonnegative known all/easy
harm, missing-label debit, all/easy reference mass and easy benefit
separately. A hurdle predicts occurrence and positive size. Reconstruct
signed all/easy risk using the same 2% coefficients and score weights.
The actions, features, online credits and easy cutoff provenance are
unchanged. Thresholds were not tuned on held outcomes.

| Policy | Online edits | Online joint pass | Joint pass at 1,457 matched edits |
| --- | ---: | ---: | ---: |
| Frozen gain only | 1,457 | 61/606 | 60/606 |
| Frozen gain and directly fitted signed risk | 1,346 | 57/606 | 60/606 |
| Replace only risk with factorized estimate | 1,614 | 63/606 | 73/606 |
| Replace only gain | 1,399 | 56/606 | 61/606 |
| Replace both | 1,637 | 53/606 | 69/606 |

Joint pass means original risk pass and positive conservative gain
against both CV and OLS4. The retrospective match fixes realized totals
for every recording and seed, observes all query scores without labels,
and obeys the same token constraints; it is not online. The matched
risk-only replacement versus directly fitted signed risk has direct
forecast gain **-0.013 pp [-0.091, +0.057]**. The equal-location difference
in joint pass fractions is **+5.98 pp [+0.58, +15.56]** (3,000 location
draws after averaging seeds). Across all 606 views it adds 13 passes.
Four of seven location averages increase and three tie; the largest
relative increase comes from `rdb7`, with one distinct recording, so
the equal-location effect falls to +1.42 pp when it is omitted. These
are exposed development data, not independent risk control.

**Risk prediction is still poorly calibrated in sign.** Pooled legal
action diagnostics (not independent observations) give the following
mean absolute errors in the same normalized units:

| Risk target | Direct signed | Factorized | Always zero |
| --- | ---: | ---: | ---: |
| All excess, all actions | 0.00979 | 0.00581 | 0.00487 |
| Easy excess, all actions | 0.00598 | 0.00138 | 0.00112 |
| Easy excess, true positive actions only | 0.00837 | 0.00486 | 0.00519 |

Factorization improves magnitude error among positive easy-risk actions
but does not beat a zero prediction on average. At zero expected excess,
its easy-risk recall is 97.5% and false-positive rate is 95.0%; the
direct signed head has 54.3% and 52.4%, respectively. Better magnitude
fits are not enough to interpret the sign as a usable risk gate. Keep
the partial matched pass improvement and the failed utility result
separate; do not replace the current policy.

`fresh_run`: 21 native CPU Torch fits, 14,700 updates, three seeds, four
threads, no workers, and a risk readout audit. `cached_verified`: exact
frozen online controls and checkpoint replay with zero updates.
No original rule change, deployment, Stage5C, SMC or independent
confirmation. [Scores](analysis.json) | [Risk audit](risk_readout.json).

```sh
.venv-pytorch/bin/python scripts/run_m3w_factorized_count_risk.py
.venv-pytorch/bin/python scripts/run_m3w_factorized_count_risk.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_factorized_risk_readout.py
```
