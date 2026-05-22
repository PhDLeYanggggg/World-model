# BPSG-MA World Model v1

This is the final deliverable for the current scaffold: a CPU-runnable, baseline-preserving 2.5D multi-agent trajectory world-state model with failure diagnostics and fallback.

Run:

```bash
python run_train_final_world_model.py --quick
python run_evaluate_final_world_model.py --quick
python run_select_final_model.py
python run_infer_world_model.py --demo
python run_visualize_final_world_model.py --demo
```

Do not treat this as true 3D, a foundation world model, latent generative modeling, or SMC.
