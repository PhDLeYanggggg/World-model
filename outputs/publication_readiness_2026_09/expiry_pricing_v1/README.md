# Pricing the Loss of an Expired Edit

I kept the release times, slot radii, causal first acceptance and actual
two-query scheduler unchanged. The head uses only self history, current
and recent observed opportunities, and ledger state. Donor nomination
and count producers exclude the held location and the donor receiving
their predictions. The same fitted head supplies four readouts:

- **Direct:** predict the net gain of waiting.
- **Mixture:** predict the probability of an actual move and gains
  conditional on moving and on expiry.
- **Priced:** use the same conditional move gain, but charge the worse of
  the conditional expiry estimate, the unconditional withdrawal estimate,
  and zero. Do not credit a predicted gain from expiry.
- **Geometry:** replace the learned expiry estimate with the negative
  mean distance between the original edit and CV. By the triangle
  inequality this is a bound on the expired edit's gain, including
  missing steps. It does not bound the gain of a successful move or
  certify the recording risk.

With at most one changed slot per recording, the full paired difference
decomposes into a moved-slot contribution and an expired-slot
contribution. A retrospective emission match retains the actually
emitted slots at their original times and radii. It evaluates utility
only, not full-policy risk.

| Readout | Accepted / moved / expired | Joint passes / 606 | Undefined easy | Direct gain vs anchor (pp, 95% CI) | Expiry contribution (pp) |
|---|---:|---:|---:|---:|---:|
| Frozen anchor | 0 / 0 / 0 | 77 | 334 | reference | 0 |
| Direct | 43 / 32 / 11 | 76 | 336 | +0.000005 [-0.000386, +0.000402] | -0.00013639 |
| Mixture | 36 / 24 / 12 | 76 | 336 | +0.000230 [0, +0.000539] | -0.00008542 |
| Priced | 34 / 23 / 11 | 77 | 335 | +0.000176 [0, +0.000401] | -0.00000415 |
| Geometry | 34 / 22 / 12 | 76 | 336 | +0.000086 [-0.000047, +0.000304] | -0.00008542 |

The asymmetric price reduces the aggregate expiry loss by about 97%
relative to the direct readout without reducing the number of expired
slots. It makes nine fewer successful moves, and the paired effect
against direct is only +0.000139 pp [-0.000140, +0.000557]. Worst single
expiry is -0.00100 pp for priced and -0.00171 pp for direct (each
normalized by the corresponding location-seed CV mass). Both mean
utility and the same-emission diagnostic for priced have an interval
that reaches zero against their references. Matching the anchor's 77
joint passes does not make the policy qualify. The bounds, cutoff and
undefined convention remain unchanged.

The three seeds reuse 202 recordings across seven exposed locations.
Intervals are paired location bootstraps, 3,000 draws after seed averaging.
This is an ablation of one head and decision rule family, not a tuning
search on independent test data. I do not deploy the priced rule or
claim that small, low-coverage gains solve the independent evidence gap.
The next useful step is to increase exposure to payable opportunities
on different camera sites without treating new observation coverage
as evidence of forecasting superiority.

```sh
.venv-pytorch/bin/python scripts/run_m3w_expiry_pricing.py
.venv-pytorch/bin/python scripts/run_m3w_expiry_pricing.py --replay
```

21 fresh arm64 CPU Torch fits, 16,800 updates in 31.24 seconds, four
compute threads, no DataLoader subprocesses. Checkpoint and PID heartbeat
every 100 updates. DroneCrowd's 30 confirmation groups stay sealed; no
Stage5C, SMC, metric or seconds-level claim.

[Registration](registration.json) | [Analysis](analysis.json) |
[Checkpoint receipts](head_freeze.json).
