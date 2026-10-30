# Reproducing the current-query risk experiment

## Material Passport

Full training and72-group policy readout are complete. Separate full replay
agrees exactly on all72 groups,144 fixed-query controls and the canonical
summary;verification.json records the receipt. Do not infer replay from a
training refit or a metric archive. Private source arrays,features and checkpoints are
required for full inference/training reproduction and are deliberately not in
Git. Public aggregate reduction does not substitute for those assets.

Use native arm64 `.venv-pytorch/bin/python`,CPU4,interop1,workers0. The entry
point rejects Intel/Rosetta before Torch import. No resource probing,DataLoader
multiprocessing,NumPy training fallback,Stage5C execution or SMC.

## Original execution sequence

```sh
.venv-pytorch/bin/python scripts/run_m3w_current_policy_critic.py register
# Commit scientific code,config,protocol and registration before real fits.
.venv-pytorch/bin/python scripts/run_m3w_current_policy_critic.py pilot
.venv-pytorch/bin/python scripts/run_m3w_current_policy_critic.py train
.venv-pytorch/bin/python scripts/report_m3w_current_policy_critic.py training
.venv-pytorch/bin/python scripts/run_m3w_current_policy_critic.py eval
.venv-pytorch/bin/python scripts/run_m3w_current_policy_critic.py verify
.venv-pytorch/bin/python scripts/report_m3w_current_policy_critic.py complete
```

Completed immutable pilot/training/readout outputs refuse replacement. For a
genuinely interrupted unfinished phase,first inspect its PID,heartbeat,log and
checkpoint,then use the same command with`--resume`. Do not restart a healthy
job or change bindings under resume. Checkpoints are atomic every100 updates,
heartbeat every50 updates plus30-second query-selection events. The registered
healthy-run boundary is12h,with10GiB free-disk reserve and256MiB model cap.

## Public aggregate check

```sh
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_current_policy_critic_v1 --verify
.venv-pytorch/bin/python scripts/report_m3w_current_policy_critic.py complete
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_current_policy_reporting.py
```

The1,685,298B ZIP contains exactly72 completed aggregate metric JSON files,not
raw rows or weights. The manifest checks every byte against complete.json.
The complete reporter requires the separate full replay receipt,then reduces
the original full roster in canonical order. It performs no new training or
optimization. Training telemetry export and training artifact tests additionally
require all original matched checkpoints. Public-only aggregate checking and
full private-asset reproduction are distinct capabilities.

```sh
.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_current_policy_critic.py \
  tests/test_m3w_current_policy_readout.py \
  tests/test_m3w_current_policy_reporting.py \
  tests/test_m3w_current_policy_artifacts.py
```

The current-target loss retains a comparable original-set probe separately
from its moving masked term. Unknown/empty query losses do not become observed
safe-zero targets. No scientific threshold or checkpoint is selected from
partial outcomes. Final step2,000 is fixed. Exact agreement establishes
engineering reproducibility,not independent efficacy or calibrated safety.

Obs8/pred12/rawstride12,local-image detector-silver,not metric/seconds,human
gold,true3D,foundation,new trajectory dynamics or submission-ready evidence.
Independent calibration/confirmation roles remain closed. No private data,
per-row choices,large caches,checkpoint weights or third-party media in Git.
