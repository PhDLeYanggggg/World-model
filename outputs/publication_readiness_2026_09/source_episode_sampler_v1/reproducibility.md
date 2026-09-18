# Reproduction

Registration SHA256: `d9f289f42d522289295ab1d8d584f3fbe49282397f36bf8b9439f187173e6cd5`.
Private hash-matching source features, episode audit and parent fits are required.
Missing assets do not constitute a completed reproduction.

```bash
.venv-pytorch/bin/python scripts/run_m3w_source_episode_sampler.py --registration configs/m3w_source_episode_sampler_v1.json
.venv-pytorch/bin/python scripts/run_m3w_source_episode_sampler.py --registration configs/m3w_source_episode_sampler_v1.json --replay
.venv-pytorch/bin/python scripts/analyze_m3w_source_episode_sampler.py --registration configs/m3w_source_episode_sampler_v1.json
.venv-pytorch/bin/python scripts/diagnose_m3w_source_episode_sampler.py
.venv-pytorch/bin/python scripts/verify_m3w_source_episode_sampler.py --registration configs/m3w_source_episode_sampler_v1.json
.venv-pytorch/bin/python scripts/report_m3w_source_episode_sampler.py --registration configs/m3w_source_episode_sampler_v1.json
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_source_episode_sampler.py tests/test_m3w_source_event_support.py
```

Atomic checkpoints every 200 steps retain optimizer, Torch RNG, sampler RNG and
draw counts. Restart the same command to resume; completed receipts verify hashes
and skip fitting. The 100-update named pilot is included in the fixed budget.
PID/step/loss heartbeat and the private training log support long-run monitoring.
No final-test checkpoint selection. The full non-hermetic legacy test suite was
not rerun for this scoped intervention. This is local reproduction, not new data.
