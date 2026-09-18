# Reproducing the Source-Site Diagnostic

Registration committed and pushed before fitting: `a91e5795`.
Config: `configs/m3w_source_site_probe_v1.json`.
SHA256: `a2038307eab36cc62ae783eedf3f359747965f2ddcc17cf4f661d83273fab74d`.
Fixed runner/model/input/schema/test hashes are bound in that registration.

Thirty real Torch models:two input arms,five held physical SDD source sites,
three seeds. Each2,000updates,batch64,43,849parameters. No model/threshold search.
The source-site folds are internal diagnostics of already exposed originaltrain40
only, not an official split reassignment. Main forecast protocol and sealed roles
are unchanged. The source label is annotation-coordinate change, not human gold.

Use native arm64 `.venv-pytorch/bin/python`. The entrypoint rejects Rosetta/x86_64
before importing Torch. Four compute threads,oneinter-op,noDataLoaderworkers.
No NumPy replacement,resource probing or MPS requirement. Each checkpoint saves
model,optimizer,RNGs,trainingIDs,drawcounts,step and loss/gradient trace atomically.

First verify that no writer is live: inspect the private `heartbeat.json`, PID
and `run.log` under `data/stage_cvpr2027_experiments/source_site_probe_v1/`.
Do not start concurrent writers. Reusing the same command resumes exact state.

```sh
.venv-pytorch/bin/python scripts/run_m3w_source_site_probe.py --registration configs/m3w_source_site_probe_v1.json
```

The pilot used `--trial mask_only_bookstore_seed17 --stop-at 100`:4.49fitseconds,
no held scoring. Those100updates are inside the fixed first model budget, not
an extra experiment. Checkpoint and heartbeat every200updates during fullrun.

Only after the full writer finishes:

```sh
.venv-pytorch/bin/python scripts/run_m3w_source_site_probe.py --registration configs/m3w_source_site_probe_v1.json --replay
.venv-pytorch/bin/python scripts/verify_m3w_source_site_probe.py --registration configs/m3w_source_site_probe_v1.json
.venv-pytorch/bin/python scripts/analyze_m3w_source_site_probe.py --registration configs/m3w_source_site_probe_v1.json
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_source_site_probe.py tests/test_m3w_source_site_analysis.py tests/test_m3w_source_visual_start.py tests/test_m3w_source_visual_start_analysis.py tests/test_m3w_source_visual_information.py tests/test_m3w_source_start_probe.py tests/test_m3w_source_start_probe_integrity.py tests/test_m3w_observed_unit_frame_v2.py tests/test_m3w_sdd_auxiliary.py
```

Prediction replay must be exact. Verifier checks30budgets,15paired training
streams,finite parameters/gradients,source-only training membership and identical
normalizers. Completed resume must preserve91immutable local artifacts plus
report with0addedfits/updates. These commands are not evidence of completion;
the actual receipts establish completion.

Primary information contrast is equal physical-site mean of window Brier gain
from RGB. Seed losses are averaged,not probabilities.2,000physical-site-block
resamples describe only five exposed sites;trainingfolds overlap. Within-site
video-block and equal-agent sensitivity are supplementary and not independent
new-scene confirmation. Constant training-prior controls stay in the table.
No held-site label is used in training normalization or inference calibration.

Full legacy test suite is not represented as rerun by these focused tests.
Public artifacts contain aggregates,code,configs and original scientific plots.
Rawdata,caches,images,per-row predictions andcheckpoints remain local andignored.
No trajectory lift,deployment,metric/seconds,true3D,foundation,Stage5C orSMC claim.
