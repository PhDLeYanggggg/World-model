# World Lab

World Lab is a bounded 2D virtual world for world-model training experiments.

It deliberately targets a realistic subset of physics instead of claiming to simulate all reality:

- Level 1: Newtonian rigid bodies with gravity, collisions, friction, restitution, and constraints.
- Level 2: sensor observations with camera rays, velocity, contact events, and occlusion flags.
- Level 3: interaction primitives for push, pull, drag, stacking, ramps, containers, ropes, and hinges.
- Level 4: material approximations through particles, soft constraints, liquid-like droplets, and breakable fragments.
- Level 5: domain randomization for seed, gravity, mass/material jitter, light angle, color/texture cues, and shape variation.
- Level 6: JSONL episode export with state, action, observation, reward, and causal events.

The app is designed as a training-data source and debugger, not a universal physics engine.

## Stage 2: Physical World Model 2.5D

Stage 2 adds a verifiable learned-residual crowd world model:

```bash
pip install -r requirements.txt
python run_stage2_demo.py
```

The demo runs a CPU-sized quick configuration from `configs/stage2.yaml`:

- generates `SyntheticPhysicalCrowd2.5D` train/val/test episodes under `data/synthetic/`;
- trains deterministic and stochastic neural residual transition models;
- evaluates t+1, t+10, t+25, t+50, and verified synthetic t+100;
- compares constant velocity, hand physics, learned residual, and three SMC proposal modes;
- writes metrics to `outputs/reports/metrics_stage2.json`, `.csv`, and `.md`;
- writes the Stage 2 report to `outputs/reports/report_stage2.md`;
- writes figures to `outputs/figures/stage2/`.

AerialMPT bauma3 remains explicitly limited: the current selected slice has only 16 frames, so t+100 is qualitative free-run only, not an accuracy metric.

Run the expert self-audit after any experiment:

```bash
python run_world_model_audit.py
```

The audit writes `outputs/reports/world_model_expert_self_audit.md` and fails the model loudly when long-horizon accuracy, physical consistency, coverage, semantic diversity, or real-data verification do not meet the bar.

## Stage 3 Data Expansion

Generate the Stage 3 data-source catalog and variable schema:

```bash
python run_stage3_data_catalog.py
```

Preview a TGSIM Foggy Bottom CSV, either from a local file or directly from the public CSV endpoint:

```bash
python run_tgsim_preview.py /path/to/TGSIM-Foggy-Bottom-Data.csv 50000
```

The preview normalizes columns into world-model fields such as `x_m`, `y_m`, `speed_mps`, `acceleration_mps2`, `body_radius_m`, and `agent_type`.
