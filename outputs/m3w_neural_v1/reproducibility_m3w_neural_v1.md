# M3W-Neural v1 Reproducibility

Use arm64 PyTorch for training/evaluation commands on Apple Silicon.

```bash
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_build_seq2seq_dataset.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_build_all_agent_dataset.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_fresh_self_gated_endpoint_candidate.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_endpoint_geometry_audit.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_gates.py
python -m pytest tests
```

- frozen git commit at package time: `58119ad`
- package input hash: `d4fa0cd6545da4e5e0acafd4b6f314a678775553ddf1f35f0ce246724dee148e`
- Do not commit caches/checkpoints/raw data when reproducing.
