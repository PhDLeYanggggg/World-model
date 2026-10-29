# Did the Rational Repair Remove the Measured Obstruction?

Fresh72/72 fit-only views completed64.016528s, PID53106 exit0. All18 new models
were already frozen. Full-range prediction hashes and fixed cohorts replay;
rational autograd forwards match ordinary forward. No new fit, held inference,
threshold change or independent role in this diagnostic.

| Fixed cohort | Head | Rows | Exact-zero easy J | Near-zero easy J | Positive easy underprediction | At exact-zero J | At near-zero J |
|---|---|---:|---:|---:|---:|---:|---:|
| full-range-selected | full-range |58707|54254|54254|79.541375|76.590339|76.590339|
| full-range-selected | rational |58707|231|40174|78.718189|0.001224|53.142671|
| rational-selected | full-range |48990|44015|44015|68.974516|65.951858|65.951858|
| rational-selected | rational |48990|241|37758|69.267837|0.001224|60.678836|

The fixed full-range cohort has1,612 unknown outcomes; the rational cohort has
1,265. Unknown rows remain in inference and derivative counts, not outcome sums.
Near-zero means L1 output/logit Jacobian <=1e-6 times the causal harm cap.
All amounts are source-normalized repeated occurrence sums, not independent n,
ADE/FDE, probabilities or physical safety. All eligible and anchor cohorts remain
in summary.json, alongside all/easy component accounting.

Exact-zero obstruction is substantially removed. Tiny derivatives and selected
underprediction remain; the held registered outcome gate still fails. This is
not grounds to deploy or to claim the full hypothesis was fixed.

```bash
.venv-pytorch/bin/python -m scripts.check_m3w_rational_gradients check
```

Obs8/pred12 raw stride12, detector-track pixel/dataset-local evidence only.
No independent roles, metric/seconds/human-gold/true3D/foundation claims.
Stage5C and SMC disabled.
