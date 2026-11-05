# Zero-Target Blocking Outputs

I traced existing fitting errors to the four outputs used by the frozen
acceptance rule. Models, thresholds and labels were not changed in this audit.

| Context | False-reject cost | Zero-channel blockers only | Nonzero blockers only | Both |
|---|---:|---:|---:|---:|
| critic0 | 0.00742947 | 0.00288060 | 0.00166763 | 0.00288125 |
| critic1 | 0.00403215 | 0.00167404 | 0.00039041 | 0.00196770 |
| critic2 | 0.02805911 | 0.01361692 | 0.00151898 | 0.01292320 |

These are disjoint query-weighted normalized fitting costs, not ADE or safety
percentages. Zero-only blockers account for about39%,42%,49% of false-reject
cost. Their row counts are1,066,258,1,948. Other rows have multiple blockers;
per-output counts in analysis.json overlap and must not be summed as samples.

For the largest context, the easy decision output blocks19,991 target-safe
proposals whose easy target is exactly zero. Risk MSE still supervises zero
targets, but the observed signed-cost term has zero weight at target zero.
At inference, a positive score nevertheless vetoes the full conjunction.
This explains a specific training-signal gap, not all generalization failure.
Nonzero blockers and unknown conservative supervision remain important.

The next matched experiment changes only the added loss: learn the sign of
the complete all/easy, risk/decision conjunction. It will not exempt zero
targets from the deployment rule or expose outcome-derived easy labels at
inference. Original risk MSE and unknown-bound losses remain unchanged.

## Verification And Limits

Fresh own-B diagnostic PID4456 completed33.69s; exact replay PID4519 completed
33.40s. All six existing model attributions reproduce exactly. Five focused
tests across decision-semantics and zero-channel diagnostics pass. Parent
assets are cached_verified; no new model, V or H was evaluated here.

Original2% all/easy risk, easy-net and support rules stay unchanged. No identity
certification; empty controls/missing support stay NA. Old failed results
remain failures. Original24-group primary is NA, not passed. Reserved Moving,
DUT, DroneCrowd and H stay closed. Native obs8/pred12 image coordinates, not
metric/seconds/true3D/foundation/human gold. Stage5C/SMC remain disabled.

```sh
.venv-pytorch/bin/python scripts/diagnose_m3w_zero_channel.py --verify
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_zero_channel.py
```
