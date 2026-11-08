# A common shrinkage factor does not rescue the frozen choices

I kept the selected recipients and forecast directions fixed and searched the entire affine amplitude range from OLS4 to each chosen path on exposed development. For each arm, the conservative gain is concave in the common amplitude and has a negative right derivative at zero. This proves that no positive common amplitude improves the registered conservative point contrast on this fixed roster, not that other directions or policies cannot improve it.

| Frozen directions | Right slope at zero (pp per amplitude) | Best common amplitude | Rowwise hindsight gain (pp) |
| --- | ---: | ---: | ---: |
| OLS hindcast | -1.349 | 0 | +0.816 |
| OLS motion | -1.576 | 0 | +0.623 |
| CV hindcast | -1.321 | 0 | +0.845 |
| Unlearned hindcast | -1.499 | 0 | +1.050 |

The OLS-hindcast slope's paired source interval is [-3.296, -0.143]. On its 9,675 directions, 6,099 individual directions have positive initial slope. The hindsight rowwise calculation can turn off the other directions and chooses amplitude with labels. It reaches +0.816 points [0.551, 1.170] against OLS4 on the common CV denominator. It is an optimistic diagnostic, not count matched, not a deployable policy, and not a risk certificate.

The full-amplitude score reproduces the parent contrast; the optimization and aggregate replay match exactly. Eight targeted tests passed. No external data, new threshold, deployment, Stage5C, SMC, metric units or seconds claim.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_ols_replacement_amplitude.py score
.venv-pytorch/bin/python scripts/audit_m3w_ols_replacement_amplitude.py replay
```

[Protocol](protocol.md) | [Results](analysis.json) | [Replay](replay_execution.json)
