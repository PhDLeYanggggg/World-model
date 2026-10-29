# Corrected Runtime Ledger

Code/protocol/repair registration commit2ea5884c was pushed before new fitting.
All v1 artifacts remain intact; its unintended encoder-frozen run is not part
of the corrected comparison and has no held outcome readout.

v2 real pilot:PID59910 exit0,300updates including exact50+50resume/direct100,
21.264seconds,2.579GBpeak RSS. Native arm64 Torch2.12.0/NumPy2.4.6,
CPU4/inter-op1/workers0. Both arms have12,322trainable parameters; actual encoder
updates and unchanged base outputs are verified. The pre-run estimate435.32s
with1.5margin is superseded by measured full training381.689s:PID60147 exit0,
36fits/72,000updates,2.952GB peak RSS. Training froze at commit18ab3abc before
readout. PID61062 readout exit0,72views in95.521s; PID61456 full replay exit0.
The scientific screen failed;60scoped tests pass3.49s. See verification.md.

The public entrypoints now target v2:
```bash
.venv-pytorch/bin/python -m scripts.run_m3w_query_context_risk pilot
.venv-pytorch/bin/python -m scripts.run_m3w_query_context_risk train
.venv-pytorch/bin/python -m scripts.run_m3w_query_context_risk train --resume
.venv-pytorch/bin/python -m scripts.run_m3w_query_context_risk eval
.venv-pytorch/bin/python -m scripts.run_m3w_query_context_risk check
.venv-pytorch/bin/python -m scripts.verify_m3w_query_context
.venv-pytorch/bin/python -m scripts.verify_m3w_query_context --check
.venv-pytorch/bin/python -m scripts.plot_m3w_query_context
```
Only use resume on an interrupted incomplete phase; a completed receipt is
immutable. Do not run duplicate training. Checkpoints and event/heartbeat logs
are under data/stage_cvpr2027_experiments/european_query_context_v2. They are not
committed. At the12h boundary resume the same budget, not a smaller substitute.
The full72,000update roster must freeze before evaluation.

The earlier CREATE read-only inventory verifies24TRAIN packets; scheduler queue
observation remains unavailable after squeue timeout. Local streaming succeeds.
No new remote submission, auth change or resource modification. Preserve the
10GiBlocal reserve and256MiBper-run checkpoint cap.

Fresh_run means corrected head training. Cached_verified means parent forecasts,
models, preprocessing and packet lineage. Independent selection/calibration/
confirmation, outer development, new trajectory training and deployment are
not_run. Obs8/pred12 rawstride12 detector-silver pixels only; Stage5C/SMC off.
