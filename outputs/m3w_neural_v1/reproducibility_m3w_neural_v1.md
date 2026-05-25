# M3W-Neural v1 Reproducibility

Use arm64 PyTorch for training/evaluation commands on Apple Silicon.

```bash
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_build_seq2seq_dataset.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_build_all_agent_dataset.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_bounded_neural_blend_dynamics.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_composite_tail_evidence.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_composite_tail_multiseed.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_pure_ucy_source_validation.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_pure_ucy_neural_retrain.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_endpoint_geometry_audit.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_gates.py
/usr/bin/arch -arm64 .venv-pytorch/bin/python run_stage41_freeze_m3w_neural_v1.py
python -m pytest tests
```

- frozen git commit at package time: `e767c49`
- package input hash: `e80e02aa1c7e89cb2892dfa9cc82fef8565a6525e9000a71adbde53aa96361d5`
- Do not commit caches/checkpoints/raw data when reproducing.
