# Can map and peer history identify better exchanges?

I kept the forecast paths, 780 realized edits, movement caps and original
2% recording rule fixed. The model could only exchange one chosen edit
for another legal edit. In addition to motion features, I tested the
publisher's static vehicle-area map and the closest simultaneous peer's
observed relative motion. The negative control rotates both descriptor
blocks within recordings and equal-support strata. It uses the same
active network inputs and the same donor draws as aligned context. The
training labels, candidates and budgets did not change. At least one of
the two affected queries had matched modality support in 7,187 of 7,198
legal exchanges; both did in 6,653.

The fresh local run trained 105 heads (five arms, seven held locations,
three seeds), with 42,000 updates in 94 seconds. The checkpoint replay
took 13 seconds with no updates. This is an offline development
comparison, not a new external test.

| Input | Exchange views (own rule) | Dual passes (own rule) | Risk passes (own rule) | Dual passes (same 101 views) | Risk passes (same 101 views) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen, no exchange | 0 | 60 | 88 | - | - |
| Motion | 112 | 50 | 79 | 46 | 77 |
| Motion + map | 85 | 51 | 84 | 48 | 83 |
| Motion + peer | 141 | 47 | 73 | 49 | 81 |
| Motion + both, aligned | 101 | 47 | 79 | 47 | 79 |
| Motion + both, misaligned | 130 | 56 | 83 | 50 | 78 |
| Uniform legal exchange (exact expectation) | - | - | - | 51.21 | 83.80 |

Counts use all 606 views. A dual pass requires the original risk rule and
positive utility against both CV and OLS4. In the matched comparison,
every arm exchanges on exactly the 101 views selected by aligned
features, without looking at outcomes to select the views. The random
control enumerates all legal exchanges at those same views.

| Paired conservative gain of aligned input (pp) | Point | 95% location interval |
| --- | ---: | ---: |
| Versus motion, own rules | +0.002479 | [-0.001378, +0.007199] |
| Versus misaligned, own rules | +0.007455 | [+0.000030, +0.019327] |
| Versus frozen, own rules | -0.003175 | [-0.006802, -0.000138] |
| Versus motion, same swaps | -0.000370 | [-0.000859, 0] |
| Versus misaligned, same swaps | +0.000895 | [-0.000226, +0.002912] |
| Versus random, same swaps | -0.001039 | [-0.002291, -0.000128] |

The paired intervals use 3,000 location draws after averaging seeds.
The apparent advantage over misaligned context does not survive fixing
the swap mask. Against matched chance the equal-location dual-pass
difference is -0.632 pp (95% interval [-1.559, -0.026]). These descriptors
do not justify a new release rule. The result is not a claim that all
maps or interaction features are ineffective.

[Protocol](protocol.md) | [Arm-specific readout](readout.json) |
[Matched readout](matched_swap_readout.json) |
[Anonymous replay](../context_review_supplement_v1/README.md)

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/run_m3w_opendd_exchange_context.py --replay
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_exchange_context_matched.py
```
