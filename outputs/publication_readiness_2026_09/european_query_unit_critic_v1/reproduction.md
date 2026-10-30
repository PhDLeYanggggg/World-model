# Reproducing the query-unit experiment

Use native arm64 `.venv-pytorch/bin/python`,CPU4/interop1/workers0. The source
runner rejects Intel/Rosetta before Torch import. Required private source,
feature and checkpoint assets are not in Git. This is not a verified data-free
cold-start reproduction package. Do not overwrite completed evidence.

## Original workflow

```sh
.venv-pytorch/bin/python scripts/run_m3w_query_unit_critic.py register
# Commit the scientific code/config/protocol/registration before execution.
.venv-pytorch/bin/python scripts/run_m3w_query_unit_critic.py pilot
.venv-pytorch/bin/python scripts/run_m3w_query_unit_critic.py train
.venv-pytorch/bin/python scripts/report_m3w_query_unit_critic.py training
.venv-pytorch/bin/python scripts/run_m3w_query_unit_critic.py eval
.venv-pytorch/bin/python scripts/run_m3w_query_unit_critic.py verify
.venv-pytorch/bin/python scripts/report_m3w_query_unit_critic.py complete
```

The current pilot/training/readout are already complete and immutable. To
recover an actually interrupted unfinished phase,inspect its PID,heartbeat,
log and last checkpoint first;then use the SAME phase with`--resume`. Do not
duplicate a healthy process or change data/model/budget under resume. Checkpoint
bindings include source inputs,labels,normalizers,initial weights,query keys,
settings,random generators and draws. Healthy runs can continue up to12h.

## Audit public aggregate evidence

```sh
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_query_unit_critic_v1 --verify
.venv-pytorch/bin/python scripts/report_m3w_query_unit_critic.py complete
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_query_unit_reporting.py
```

The1,718,880B archive contains only completed aggregate JSON metrics. Its manifest
verifies the exact original group bytes. The complete reporter re-reduces the
fixed72-group roster in its original order;it does not retrain or reoptimize.
Re-exporting training telemetry and the private-asset artifact tests requires
the original matched checkpoints. An exact replay is not independent science.

## Read-only score diagnosis

```sh
.venv-pytorch/bin/python scripts/probe_m3w_selected_scores.py run
.venv-pytorch/bin/python scripts/probe_m3w_selected_scores.py verify
.venv-pytorch/bin/python scripts/report_m3w_selected_scores.py
```

The current diagnostic is also complete and refuses replacement. Its own
separate registration/execution hashes bind the full policy outputs. It
re-infers old/new critics and checks frozen choices,not reoptimizes them.
All288 original-set MSE controls agree,including preserved not-estimable cases.
Its conditional support differences are not causal shift estimates or safety
certificates. The read-only capacity diagnostic has an explicit historical-
event execution adapter;see its execution_amendment.md rather than editing its
old creation-time registration flag.

Current-query supervision is a new registered experiment,not a resume or a
renamed result here. Independent roles,Stage5C execution and SMC stay closed.
No raw data,caches,per-row arrays,checkpoints,third-party images/videos in Git.
