# Ordering Sensitivity Does Not Establish A Useful Repair

All three original joint-control models were recomputed over their full own-B
fitting populations:6197/5521/50748 rows,52/137/898 queries and5767/3715/38579
known allowed proposals. These are training rows, not independent observations
or extra test scenes. The original identity outputs match frozen metrics.

Changing only neighbor-slot order changes44.953%,11.971%,3.956% of known
proposal decisions on average across seven fixed nonidentity shifts. However,
averaging all eight normalized output views worsens MSE from0.009564/0.008399/
0.612722 to0.616367/1.844052/2.222340. Joint decision cost also worsens in all
three contexts. The prospective screen requires at least two improved contexts
and is not met:0/3. No new trained model or deployment follows.

The original model uses a deterministic nearest-neighbor order, which is a
valid input convention. Permuting it also moves values across slot-specific
normalization statistics. The perturbation can therefore create inputs unlike
training, even though the same geometric neighborhood and masks are retained.
The observed failure does not prove that permutation-invariant training would
fail; nor does sensitivity alone prove an implementation defect or explain
the earlier real-scene errors. No shift is selected retrospectively.

The metric implementation initially disagreed with its own identity control;
the guard rejected that run before analysis. The repair reuses the established
metric rather than changing its meaning. See [repair record](failure_recovery.md).
The corrected result and all three contexts replay exactly.

The practical next model question is the relation between a causal candidate
rollout and its own past motion: the existing explicit basis summarizes past,
floor and candidate largely separately. A future controlled feature change
must check existing schema first, retain canonical neighbor ordering, retain
the original loss/risk/support rules and compare to a matched control. This is
not evidence that such features will help, and that experiment is not_run here.
No projection, neighbor-averaging or threshold sweep is justified by this run.

## Evidence Status

- `fresh_run`: frozen-model own-B perturbation and original-metric checks.
- `cached_verified`: trained checkpoints, preprocessing, labels and source roles.
- `not_run`: new feature training, new V/H results and independent confirmation.

Original24-group primary remains NA. Missing support and empty controls do not
pass; no identity-fallback certification is introduced. Native obs8/pred12
image coordinates are not metric, seconds, true3D, foundation or human gold.
Stage5C and SMC remain disabled; the broader research goal is not complete.

## Replay

```bash
.venv-pytorch/bin/python scripts/replay_m3w_neighbor_order.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_neighbor_order.py tests/test_m3w_neighbor_order_replay.py -q
```

The replay needs the original private cache and model files. GitHub contains
only source, reports and hashes, not those assets. It must reproduce metrics
without replacing original execution receipts or changing public file contents.
