# Fixed Cost Refit Execution

Registration`1a9d26ed`was pushed before training. Verified nested prerequisite
results were committed as`b03e849b`. Preflight session12598/PID75798 exited0
after841binding checks. Pilot49228/PID75878 completed100updates; full53690/
PID75911 resumed it and exited0 after all36heads,108,000updates and27,648,000draws.
Recorded cost-head fitting time76.64547seconds excludes upstream prediction
training and data/verification overhead. Nativearm64, CPU4/inter-op1/workers0.

Evaluation33089/PID76128 froze all decisions before outcome reductions and
exited0. Replay2774/PID76188 and separate arithmetic35308 exited0. Post-readout
same-population forensics69680 exited0. Scoped suite89556:91passed in4.08seconds.
No new training, cache, evaluation or verification process remains live from
this experiment. No CREATE job was submitted or remote scheduler state claimed.

```bash
.venv-pytorch/bin/python scripts/run_m3w_eqmotion_cost_refit.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_eqmotion_cost_refit.py --view coupa_seed17 --arm direct_native --stop-at 100
.venv-pytorch/bin/python scripts/run_m3w_eqmotion_cost_refit.py --resume
.venv-pytorch/bin/python scripts/run_m3w_eqmotion_cost_refit.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_eqmotion_cost_refit.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_eqmotion_cost_refit.py
.venv-pytorch/bin/python scripts/audit_m3w_eqmotion_cost_refit.py
```

The pilot command is for a fresh identity with no completed fit, not an instruction
to restart a completed run. Existing checkpoints require`--resume`; complete
receipts are hash-checked and reused. Do not edit registered files or delete
identities to bypass checks. The process lock precedes training-view assembly.
Private prediction, supervision and checkpoint arrays remain outside Git.

The joint primary gate fails despite recovered empirical easy preservation.
No threshold tuning, seed selection, closed-role readout, deployment or
independent confirmation. A separately written verifier is not an independent
research group. Unknown labels remain unknown. See`conclusions.md`for numerical
results and the remaining methodological gap.
