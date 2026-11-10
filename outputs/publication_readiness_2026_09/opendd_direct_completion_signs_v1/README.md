# OpenDD: what the observed and conservative results identify

I used the same direct pairs, fixed CV denominator and seven location
means as the count-matched, movement-matched and learned-visibility
studies. For each pair, the worst-case missing-step debit is the
observed contribution minus the lower endpoint. Adding the same debit
to the observed contribution gives the upper endpoint. No raw labels
were reopened and no scores were changed.

| Contrast | Observed | Sample completion envelope | 95% source interval for lower | 95% source interval for upper |
|---|---:|---:|---:|---:|
| Support vs original, count matched | +0.272 | [-1.029, +1.574] | [-1.652, -0.437] | [+0.119, +3.737] |
| Support vs original, count and movement matched | +0.591 | [-0.388, +1.570] | [-0.798, -0.072] | [+0.402, +3.422] |
| Learned missingness vs donor step profile, count and movement matched | -0.014 | [-0.120, +0.093] | [-0.229, -0.034] | [+0.030, +0.185] |

Units: 0.001 percentage points of observed CV error. The envelopes
are over possible positions for the missing steps in the scored sample.
The intervals use 3,000 paired location resamples; they are not intervals
over imputed outcomes. All three sample envelopes cross zero under every
leave-one-location-out deletion (21/21). Thus none identifies the sign
of the unrestricted complete-label contrast. A negative lower endpoint
is evidence against a conservative gain, not proof of worse realized ADE.

The bounds are sharp for unrestricted additive trajectory comparisons:
placing an unknown position at the treatment forecast attains the upper
per-step error difference, and at the reference forecast attains the
lower. The construction does not require either completion to be a
physically coherent trajectory and is not used for risk ratios.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_opendd_completion_signs.py
```

[Protocol](protocol.md) | [Endpoint and deletion results](readout.json).
`fresh_run`: upper endpoints, paired intervals and deletion analysis.
`cached_verified`: the original direct contrasts, including their lower
endpoints. `not_run`: new raw outcomes, training or external evaluation.
Original risk rule and deployment unchanged.
