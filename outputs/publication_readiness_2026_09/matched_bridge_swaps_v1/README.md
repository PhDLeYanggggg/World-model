# Same-Count Bridge Swap Learning

**Outcome: the joint ranker did not qualify.** This is a three-seed
development comparison on seven previously exposed OpenDD locations,
not independent evidence. The held location is excluded from fitting
and risk-cut estimation in every cell. The eight-observation,
12-prediction-step protocol and original CV-relative 2% recording
risk are unchanged. Undefined easy support is not a pass.

The frozen hurdle supplies the number of edits for each query. Every
arm preserves it exactly, including zero. Learned alternatives differ
only in the recipients and their CV-to-OLS4 amplitudes. During training
two legal configurations with the same count are sampled per query.
The target is their relative utility, including conservative gain
against both references, signed-risk penalties and an easy-support
term. The local loss does not certify recording-level risk. A symmetric
pair term sees the two causal action embeddings and query size; it does
not encode explicit inter-agent geometry. Additive ranking removes that
term; joint point regression retains it but uses absolute proxy targets.
This follows the general motivation of [decision-aware training](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2020.3922),
not the SPO+ loss or its guarantees.

| Policy | Risk passes | Risk pass and beat both | Gain vs CV (pp) | Direct gain vs independent (pp, 95% CI) |
| --- | ---: | ---: | ---: | ---: |
| Frozen independent | 85/606 | 54/606 | -0.197 | reference |
| Frozen hurdle | 85/606 | 54/606 | -0.192 | +0.005 [-0.004, +0.019] |
| Joint point regression | 104/606 | 61/606 | -0.127 | +0.024 [-0.133, +0.294] |
| Additive ranking | 104/606 | 63/606 | +0.026 | +0.200 [-0.041, +0.661] |
| Joint ranking (primary) | 94/606 | 58/606 | +0.057 | +0.228 [-0.029, +0.692] |

All arms make 5,214 edits. Joint ranking changes 1,728 recipients
and changes only the amplitude on 26 additional retained recipients.
Intervals are location bootstrap (3,000 draws) after averaging seeds.
The two-recording `rdb2` contributes +1.572 pp at the location level,
but neither recording passes both gates in any seed. Omitting it leaves
+0.004 pp. The two largest locations both favor the independent control
on the direct gain metric, and the additive ablation passes more risk
checks. There is no reliable joint-model advantage.

**Exact count hindsight.** With the same bank and frozen count vector,
integer optimization verifies feasible gains against both baselines in
276/606 recording-seed evaluations. There are 160 risk-infeasible,
49 without a legal easy edit at the frozen counts, 15 without any easy
reference event, 75 with zero switches, and 31 bounded below strict
positive gain. Zero-switch cases are counted separately and first.
The prior cap-only 170/202 distinct
recording witnesses correspond to 510/606 views on this deterministic
bank. This gap concerns count allocation, not a change to the 2% rule.
The remaining 276 vs 58 gap also leaves room for better recipient choice.

**Provenance.** Native arm64 CPU Torch, four threads, zero workers:
63 fits, 44,100 fresh updates. All checkpoint hashes, held scores and
frozen controls reproduce with zero updates. Exact capacity was computed
fresh using future labels only for the diagnostic. DroneCrowd remains
15 calibration groups and 30 sealed confirmation groups. No change to
deployment, Stage5C or SMC.

```sh
.venv-pytorch/bin/python scripts/run_m3w_matched_bridge_swaps.py
.venv-pytorch/bin/python scripts/run_m3w_matched_bridge_swaps.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_matched_bridge_swap_capacity.py
```
