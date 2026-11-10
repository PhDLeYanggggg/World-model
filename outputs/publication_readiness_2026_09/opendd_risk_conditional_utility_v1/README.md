# OpenDD: rank utility among plans that pass risk

I kept the forecast menu, recording rule and realized budgets fixed. Both
new arms start from the previous preference checkpoints and take 300
updates. Their first stream sees the same ordinary preference pairs. The
control draws more ordinary pairs; the utility arm instead compares two
plans that both pass the original risk checks and differ in conservative
gain. Both plans have the same count and movement cap. All training labels
are from donor locations, not the held location being scored.

| Scores used by the same offline allocator | Edits | Original risk passes / 606 | Risk and both references / 606 |
| --- | ---: | ---: | ---: |
| Frozen count scores | 780 | 88 | 60 |
| Original pairwise head | 780 | 95 | 59 |
| Ordinary continuation | 780 | 91 | 55 |
| Risk-conditional utility continuation | 780 | 94 | 58 |

Conditional training adds three dual passes versus ordinary continuation,
all from one location. The equal-location difference is +0.112 percentage
points (95% location interval [0.000, +0.337]); the corresponding paired
conservative gain is -0.00197 pp (95% [-0.01659, +0.01418]). The new head
also falls below the original pairwise head on paired conservative gain:
-0.01087 pp (95% [-0.02136, -0.00290]). It does not beat the frozen 60
passes. That is not evidence for changing the allocation or release policy.

The seven held locations and three seeds give 21 cells. The two continuations
fit 42 checkpoints (12,600 updates) in 45.9 seconds on native arm64. I used
6,112 utility pairs across fitting roles, 125 to 369 per cell; pairs recur
across folds and are not independent recordings. All pairs were rechecked
with the original risk and trajectory scorer. No matched score allocation
was unresolved. Zero-update checkpoint replay reproduced the result in
24.0 seconds. No external outcomes were opened and no threshold was tuned.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/run_m3w_opendd_risk_conditional_utility.py --replay
```

Omit `--replay` to fit or resume. The [protocol](protocol.md),
[registration](registration.json), [checkpoint manifest](head_freeze.json)
and [readout](readout.json) record fitting roles and per-location results.
The [v177 manuscript](../evidence_manuscript_v177_decision_ablations/README.md)
reports this alongside the earlier controls, not as independent confirmation.
