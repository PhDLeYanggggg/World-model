# Recipient exchanges and peak displacement budgets

I started with the frozen half-peak policy and let it trade one
edited row for an unedited row per query. The incoming path follows
OLS4, scaled down to the outgoing radius. The edit count and sum
of peak displacement stay equal in each query. Geometry ranks the
exchange by OLS4 agreement; the trained model estimates the dual
conservative gains and risk changes. A gated variant requires both
predicted gains to be positive and no predicted risk increase.
All three use the same shortlist.

## Fresh training and matched results

Nine native PyTorch fits use 12,600 updates across three training
seeds and three source-role rotations. Each held arm edits 54,748
rows. The half-peak anchor reproduces the previous risk and utility
readout. Bootstrap intervals resample the nine source groups 3,000
times after averaging seeds.

| Held policy | Exchanges | Original risk passes / 72 | Worst known signed easy change | Paired lower gain vs anchor (pp) |
|---|---:|---:|---:|---:|
| Half-peak anchor | 0 | 18 | +30.25% | 0 |
| Geometry | 2,382 | 19 | +4.81% | -0.041 [-0.101, -0.006] |
| Learned | 1,154 | 18 | +30.25% | -0.005 [-0.012, -0.0005] |
| Learned with gate | 894 | 18 | +30.25% | -0.003 [-0.008, +0.0002] |

Geometry improves the worst known easy value, but it is still over
2%. It does not produce a conservative utility gain. Its point
estimate versus OLS4 is -55.338 pp, and the gated model's
conservative pair-distance change versus the anchor is -0.004
pixels (-0.008 to -0.0004). **Calibration: 0/9.** No deployment.

## Is there any solution at the same budget?

The hindsight envelope for the one-exchange menu excludes OLS4
gains in 69/72 views before applying risk. I then allow arbitrary
paths, any recipients, and any number of exchanges while retaining
the total radius and (where specified) the edit count. The table
reports **excluded** views out of 72, not observed policy passes.

| Peak budget relative to half-peak anchor | Per-query count fixed | Per-query count free | Per-recording radius and count pooled |
|---|---:|---:|---:|
| 0.5x | 68 | 68 | 68 |
| 1x | 67 | 67 | 67 |
| 2x | 67 | 67 | 66 |
| 4x | 64 | 64 | 57 |
| 8x | 57 | 57 | 54 |
| Unbounded | 33 | 0 | 28 |

The last column pools the original edit count as an upper limit
across queries, not an unconstrained count. With no radius limit,
zero-count queries remain closed only for the fixed query-count
column. Positive envelopes do not imply a legal, realizable or
risk-feasible plan.

The bound follows the standard
[Lagrangian relaxation principle](https://pubsonline.informs.org/doi/10.1287/mnsc.27.1.1).
Each row improves the OLS4 numerator by at most the sum of its
radius-clipped distances, with the original query weights. Pricing
radius and taking the top count gives an upper bound. Pooling allows
reallocation across queries; it does not solve the original risk
problem. At the same total displacement it fails to open additional
views in this roster. The next model should therefore vary the
amount of displacement with predicted gain and risk. Simply
reordering recipients at the old budget is not a promising repair.
The original 2% rule should remain the deciding constraint.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/train_m3w_cv_matched_exchange.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_exchange_peak_envelope.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_exchange_record_pool.py
.venv-pytorch/bin/python scripts/build_m3w_cv_matched_exchange_replay.py
```

Training and envelopes are `fresh_run`; parent paths and no-update
checkpoint replay are `cached_verified`. Remove `--replay` to train
or resume after checking PID and heartbeat. The anonymous NumPy
archive reproduces the scalar policy table, not training or the
path-level envelopes. These are native observation-8/prediction-12
steps in dataset-local pixels. No new A outcomes, deployment,
Stage5C, SMC or independent confirmation; the 30 DroneCrowd
confirmation groups remain sealed.

[Protocol](protocol.md) | [Training results](readout.json) |
[Query envelope](peak_envelope.json) | [Recording envelope](record_peak_envelope.json) |
[Manuscript v138](../evidence_manuscript_v138/README.md)
