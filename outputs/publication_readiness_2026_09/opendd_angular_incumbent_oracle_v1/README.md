# How much can direction alone recover at the original recipients?

I kept the original 780 recipients and their stepwise CV-relative
norms. The finite oracle chooses an angle using the future labels;
it always includes the unchanged path. The stepwise bound rotates
independently toward each observed position and keeps the original
at missing steps. For a fixed norm, the nearest point is the target
direction projected onto the allowed arc. Because the loss adds
over steps, this gives an analytic upper bound on the utility of
the smooth bounded angular head, not a trained forecast.

| Utility-optimal candidate family | Paired lower gain, pp | Worst location, pp | Views with gain > 0.10 pp | Dual passes of the selected paths |
| --- | ---: | ---: | ---: | ---: |
| 13 constant angles including zero | +0.05573 | +0.01320 | 165 | 158 |
| Five analytic directions including original | +0.05045 | +0.00119 | 152 | 159 |
| Independent per-step direction, bounded to 90 degrees | +0.06766 | +0.03075 | 182 | 174 |
| Independent per-step direction, unrestricted | +0.08442 | +0.03075 | 202 | 181 |

The finite angle menu is useful even at the worst location in
hindsight. The observed lack of model gain is not simply a lack
of feasible angular changes. The range also puts the opportunity
in perspective: the equal-location conservative gain in this
fixed-size regime is small even with full future labels. The
dual-pass counts describe utility-optimal paths; the oracle does
not maximize risk passes or certify a release.

[Protocol](protocol.md) | [Grouped readout](readout.json)

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_angular_incumbent_oracle.py
```
