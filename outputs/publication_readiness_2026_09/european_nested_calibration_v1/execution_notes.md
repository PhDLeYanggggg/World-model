# Execution and Reproduction

## Fresh and Cached Work

- fresh_run:18 inner Transformer fits,54 score-head fits,72 source calibration
  maps,72 outer policy views,all checkpoint and decision replays,transport audit.
- cached_verified:nine final four-locality predictors and their inference banks;
  hash-bound source geometry,labels,folds and earlier control-code dependencies.
- not_run:reserved selection/calibration/confirmation,DroneCrowd readout,new
  joint optimization,CREATE execution,Stage5C,SMC and deployment promotion.
  No new remote authentication was attempted in this experiment. Earlier CREATE
  observations are not a current scheduler or asset inventory.

## Runtime

Native arm64 `.venv-pytorch/bin/python`,Torch CPU,4 compute threads,1 inter-op
thread,workers0.18 predictors have88,514 parameters each;54 heads have22,914.
No NumPy replacement for neural fitting,DataLoader multiprocessing or runtime
resource probing. All requested training processes finished normally.

Producer pilot PID36794,training/prediction/replay PID36827. Head preflight
PID39574,pilot39666,training/calibration/readout39696,replay40172. Atomic
checkpoints and UTC events remain in the ignored private experiment directory.
Both100-update pilots resumed inside the registered budget.

Summed predictor fit time:1,731.753506s. Summed head fit time:84.969341s.
These are measured fit-loop times,not total elapsed time including hashing,
prediction,replay,reporting or machine idle periods. Producer pilot1.784429s;
head pilot0.091062s. No precision,thread or model-size change was used to obtain
a more favorable result.

Each predictor drew256,000 training rows. Across18 predictors,129,241 draws had
unknown future labels and contributed zero supervised loss under the unchanged
sampler. Head sampling excluded unknown labels:512,000 supported draws per head.
These repeated draws are not distinct observations. All six heads in each of
nine fold/seed groups use identical sampled rows and sampling RNG endpoints.

## Verification

18 producer and54 head checkpoints exactly reproduce sampled predictions on
4,096 rows each. All216 outer decisions independently reconstruct from causal
scores,frozen maps and past-motion masks. Metrics fully recompute. The432
role/rule diagnostic records retain1,728 locality entries,including failures.

All196 tests in28 scoped files pass; the full legacy suite was not run. The
exact test list,log digest,model evidence digests and process checks are in
[completion_checks.json](completion_checks.json). The revised all-view figure
was rendered and inspected for readability.

Analysis SHA256:`c74bf6a7b578eb13ed641bd0775a94e1aad3fdbbb8511c93e64a4e78fe324f0d`

Summary SHA256:`356d80028183d038ad122ffcf29a8d78967e294f8e25f42dc040f8f97b253ea1`

## Reproduction

Follow [operation_zh.md](operation_zh.md). Existing model runs require explicit
`--resume`; matching complete artifacts are verified rather than silently
refitted. The producer bank must finish before head preparation;all heads must
finish before calibration;all calibration maps must freeze before readout.

After reporting,run:

```bash
.venv-pytorch/bin/python scripts/complete_m3w_european_nested_calibration.py
```

No raw data,prediction bank,checkpoint,large cache,video/image,third-party data
or virtual environment belongs in Git. Only code,registration,aggregate evidence
and documentation are published. Unrelated staged work is left untouched.

Data remain source-development released detector tracks,image pixels,
obs8/pred12 rawstride12. No independent confirmation,calibrated physical safety,
metric/seconds,human-gold,true3D,foundation,Stage5C or SMC claim.
