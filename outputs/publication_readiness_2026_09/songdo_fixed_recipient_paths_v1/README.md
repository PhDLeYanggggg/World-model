# Songdo: change the path rule without changing the vehicles

The [recipient attribution](../songdo_recipient_attribution_v1/README.md)
showed where the original score difference occurred, but did not test
what happens if both rules must edit the same vehicles. I crossed the
original independent and joint recipient sets with the two path rules.
All four cells use the same candidate bank, query, two edits and
peak radius. The original diagonal choices replayed in all 480 views;
the crossed choices were frozen at `2bbe0901b` before rereading labels.

| Paired contrast | Conservative trajectory gain (pp) | 95% source bootstrap (pp) | Views with different paths / 480 |
| --- | ---: | ---: | ---: |
| Joint path rule on independent recipients | +0.00001411 | [-0.00004528, +0.00006648] | 344 |
| Joint path rule on joint recipients | +0.00000358 | [-0.00000177, +0.00001189] | 70 |
| Joint recipients with independent paths | -0.00044577 | [-0.00150891, +0.00023435] | 455 |
| Joint recipients with joint paths | -0.00045672 | [-0.00147047, +0.00019423] | 455 |
| Original joint minus independent | -0.00044218 | [-0.00149870, +0.00023659] | 460 |

The point differences from swapping recipients are larger than those
from switching path rules, although the source intervals cross zero.
On independent recipients, the joint path rule improves the available
relative-motion score by +0.00003231 native units (95% interval
[+0.00000875, +0.00006438]); its conservative relative-motion contrast
is -0.00002463 ([-0.00004294, -0.00000856]). The latter is a lower bound,
not a conclusion that the full-grid effect is harmful. The corresponding
conservative contrast on joint recipients is -0.00000657.

| Recipients | Path rule | Original risk passes / 480 | Risk and both utility passes / 480 |
| --- | --- | ---: | ---: |
| Independent | Independent | 179 | 169 |
| Independent | Joint | 174 | 166 |
| Joint | Independent | 71 | 55 |
| Joint | Joint | 69 | 53 |

The original 2% recording checks and both references are unchanged.
These are matched-proposal passes, not newly released views. No new
threshold was selected. Observed trajectory effects sum along either
route through the factorial comparison, but lower bounds with
missing targets do not generally add.

The [formal 40-flight calibration and 160-flight confirmation split](../songdo_calibrated_20261110/protocol.md)
is unchanged. For this control I use only the already-scored confirmation
flights, averaging three seeds within flight and grouping flights by 14
linked source components for 3,000 paired bootstrap draws. This is a
retrospective control, not a new external test. The strongest remaining
question is whether recipient selection works on new scenes. No
calibration or new source outcomes were opened for this control.

[Protocol](protocol.md) | [Readout](readout.json) |
[v193 manuscript](../evidence_manuscript_v193_songdo_fixed_paths/README.md)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_songdo_fixed_recipient_paths.py freeze
.venv-pytorch/bin/python scripts/run_m3w_songdo_fixed_recipient_paths.py score
```

The local freeze took 7 seconds and the full readout 78 seconds.
Both phases keep per-flight receipts and can be resumed with the same
commands. The scoring phase verifies the published forecast freeze.
