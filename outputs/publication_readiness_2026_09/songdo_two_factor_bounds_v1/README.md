# Songdo: allocate the joint-versus-independent contrast

I reused the four [fixed-recipient proposals](../songdo_fixed_recipient_paths_v1/README.md)
on the 160 already scored Songdo flights. I averaged both orders of
changing recipients and changing paths, so the observed contributions
sum to the original joint-minus-independent contrast. The shared
unknown future is bounded using the smaller of two distance matchings.

All entries below are percentage points on the original observed CV
denominator. The envelope varies missing positions in this sample.
The bootstrap instead resamples the 14 source components after
averaging the three seeds and the flights within each source.

| Component | Observed contribution | Missing-label envelope | 95% source bootstrap for the observed contribution |
| --- | ---: | ---: | ---: |
| Recipient choice | -0.00028113 | [-0.00045125, -0.00011101] | [-0.00102032, +0.00022780] |
| Path choice | +0.00000909 | [+0.00000885, +0.00000934] | [-0.00001635, +0.00003351] |
| Original total | -0.00027203 | [-0.00044218, -0.00010188] | [-0.00102733, +0.00025093] |

The unfavorable point estimate mainly reflects whom joint ranking
edits, not which paths it assigns to the same recipients. Neither
component has a source interval that resolves its sign. The bound
removes redundant uncertainty in the constructed cancellation test,
but did not tighten either component on these Songdo predictions.
The total bound is computed directly, not by summing component bounds.

The original total and its envelope reproduce on all 480 flight-seed
views and all source means. This is fresh retrospective arithmetic
and target joining on already scored flights; not a new confirmation
or a calibration change. The original release decision stands.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_songdo_two_factor_bounds.py
```

The second run reuses hash-bound per-flight scalars and does not need
another target join. [Protocol](protocol.md) | [Readout](readout.json).
