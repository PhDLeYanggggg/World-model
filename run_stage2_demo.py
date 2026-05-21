from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
IN_PROJECT_VENV = Path(sys.prefix).resolve() == (PROJECT_ROOT / ".venv").resolve()
if VENV_PYTHON.exists() and not IN_PROJECT_VENV and os.environ.get("STAGE2_NO_VENV_REEXEC") != "1":
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

from src.data.synthetic_dataset import generate_dataset
from src.evaluation.world_model_self_audit import run_world_model_self_audit
from src.training.evaluate_stage2 import evaluate_stage2
from src.training.train_stage2 import train_stage2_models
from src.utils.config import ensure_dirs, load_config


def main() -> None:
    cfg = load_config("configs/stage2.yaml")
    ensure_dirs(
        Path(cfg["output_dir"]) / "reports",
        Path(cfg["output_dir"]) / "figures" / "stage2",
        Path(cfg["output_dir"]) / "models" / "stage2",
        cfg["synthetic_dir"],
    )
    write_model_audit(cfg)
    write_data_sources_stub(cfg)
    print("[stage2] generating SyntheticPhysicalCrowd2.5D quick dataset", flush=True)
    episodes = generate_dataset(cfg, quick=True, force=True)
    print(f"[stage2] episodes={len(episodes)}", flush=True)
    print("[stage2] training deterministic and stochastic neural residual models", flush=True)
    model_bundle = train_stage2_models(episodes, cfg, quick=True)
    print("[stage2] evaluating t+100 baselines, neural rollouts, and SMC proposals", flush=True)
    payload = evaluate_stage2(episodes, model_bundle, cfg, quick=True)
    audit = run_world_model_self_audit()
    summary_path = Path(cfg["output_dir"]) / "stage2_demo_summary.json"
    summary = console_summary(payload)
    summary["self_audit"] = {"score": audit["score"], "verdict": audit["verdict"]}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def write_model_audit(cfg: dict) -> None:
    path = Path(cfg["reports"]["audit_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Model Audit Stage 2

## One-Sentence Status

当前系统是 pseudo-3D physics-informed SMC state-space world-model scaffold。
它不是 true 3D world model。
它不是 learned neural world model in the full real-data sense。
它的 t+100 在 AerialMPT bauma3 上只是 free-run，不是 verified forecast。

## 1. Real Observed Variables

- AerialMPT image-space detections / track positions: `u`, `v` when available.
- Short observed trajectory window in bauma3: selected slice has 16 frames.
- SyntheticPhysicalCrowd2.5D true states: generated `X`, `Y`, velocity, acceleration, goal, radius, active/reached flags, collision/obstacle/boundary diagnostics.
- Synthetic scene geometry: walkable bounds, obstacle rectangles, exits, spawn regions.

## 2. Derived From Observation

- Weak ground-plane `X/Y` from image `u/v` using weak GSD / homography assumptions in the previous pseudo3D stage.
- Velocity and acceleration estimated by finite differences.
- Heading from velocity direction.
- Local density, nearest-neighbor distance, min gap, obstacle distance, boundary distance.
- Pixel/world projection error in the previous AerialMPT scaffold.

## 3. Latent Inferred Variables

- Per-person intent / goal when real goals are not annotated.
- Desired speed and interaction strength for real AerialMPT.
- Body footprint uncertainty when no calibrated body scale is available.
- SMC particle weight, latent branch identity, and terminal semantic mode.

## 4. Human Assumptions

- Ground is modeled as `Z=0`; this is 2.5D / pseudo-3D, not recovered metric 3D.
- Pedestrians are vertical cylinders/capsules with approximate radius.
- Weak homography / GSD replaces real intrinsics/extrinsics unless camera calibration exists.
- Synthetic scenes are simplified rectangles, exits, walls, and obstacles.
- Quick demo uses fewer episodes and particles than the full config to run on CPU.

## 5. Hand-Written Physics Rules

- Social-force-like acceleration toward goals.
- Neighbor repulsion and comfort margin.
- Obstacle repulsion and boundary pushback.
- Collision projection in world coordinates.
- Obstacle/boundary projection.
- Speed and acceleration clipping.

## 6. Learned From Data

- Stage 2 trains deterministic and stochastic neural residual transition models on synthetic trajectories.
- The learned component predicts residual acceleration:

```text
A_residual = A_true_next - A_hand_physics
```

- It is not a fully learned dynamics model because hand-coded physics, constraints, and projection remain central.

## 7. Why This Is Not a Full Learned World Model

- Real AerialMPT data does not provide long t+100 supervision in the selected bauma3 slice.
- Camera calibration is weak; no real metric 3D is recovered.
- Dynamics still depend on hand-coded social force and projection.
- Scene geometry is manually/synthetically specified, not inferred from pixels.
- Goal and intent are latent in real data rather than directly observed.

## 8. Why AerialMPT bauma3 t+100 Cannot Be Evaluated

- The selected sequence has only 16 frames.
- From start frame 4, the maximum ground-truth future is t+12.
- t+20, t+50, and t+100 have no ground truth in this slice.
- Therefore any AerialMPT t+100 output is qualitative free-run only and must not be reported as ADE@100/FDE@100.

## 9. Why Previous Terminal Clusters Lacked Semantic Diversity

- The previous clusters were mostly endpoint and local event based.
- SMC branches were dominated by the same jammed / east-shifted mode.
- Without long-horizon ground truth, goal labels, and diverse scene events, terminal modes collapsed into similar congestion-risk narratives.
- Stage 2 uses semantic event features, but if clusters still collapse, the report must say so explicitly.

## 10. Why Pseudo-3D Is Not True 3D

- `Z=0` ground plane is assumed.
- Human bodies are vertical cylinders/capsules, not measured meshes.
- Without real `K`, `R`, `t` or control-point homography, depth and metric scale remain uncertain.
- The model is useful for ground-plane crowd dynamics, not for full 3D physical reconstruction.
""",
        encoding="utf-8",
    )


def write_data_sources_stub(cfg: dict) -> None:
    path = Path(cfg["reports"]["data_sources_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Stage 2 Real Data Sources

This file records candidate long-trajectory data sources for moving beyond the short AerialMPT bauma3 slice.

| Dataset | Downloadable | Trajectories | Frames | Ground-plane coordinates | Homography / scene map | t+100 evaluation | Current decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Stanford Drone Dataset | Yes | Yes | Yes | Pixel annotations with scene context; can be mapped if calibrated | Scene images; homography must be supplied/estimated | Likely, depending on frame rate and track length | Best next real-data target |
| TrajNet++ | Yes | Yes | Usually trajectory tables, not raw images | Trajectory coordinates | Homography varies by source | Often yes | Good for trajectory-only training/eval |
| ETH/UCY | Yes | Yes | Fixed-camera pedestrian scenes | World/pixel annotations depending on version | Homographies often available in common releases | Often yes after horizon alignment | Good compact benchmark |
| AerialMPT | Partially in current project | Yes for selected slice, but bauma3 slice is short | Yes | Weak homography currently used | No real calibration in current scaffold | No for selected bauma3 16-frame slice | Keep only t+12 verified, t+100 qualitative |

## Source Notes

- Stanford Drone Dataset official page: https://cvgl.stanford.edu/projects/uav_data/ . It describes eight unique scenes and multiple agent types including pedestrians.
- TrajNet++ official EPFL/VITA dataset page: https://www.epfl.ch/labs/vita/datasets/ . It describes TrajNet++ as an interaction-centric human trajectory forecasting benchmark.
- ETH BIWI Walking Pedestrians official ETH dataset page: https://vision.ee.ethz.ch/datsets.html . It lists manually annotated walking pedestrians in busy bird-eye scenarios with annotations/videos.
- UCY crowd data official URL is commonly cited as https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data . The page was not reachable during this Stage 2 run, so it is not yet wired into the loader.

The current Stage 2 demo does not automatically download these datasets. Next step is to add one real long-trajectory loader and a calibrated scene file.
""",
        encoding="utf-8",
    )


def console_summary(payload: dict) -> dict:
    rows = payload["metrics_rows"]
    return {
        "report": payload["evaluation_meta"],
        "metrics_stage2_json": "outputs/reports/metrics_stage2.json",
        "report_stage2_md": "outputs/reports/report_stage2.md",
        "figures": "outputs/figures/stage2",
        "synthetic_t100": {
            row["model"]: {
                "ADE@100": row["ADE@100"],
                "FDE@100": row["FDE@100"],
                "coverage@64": row["coverage@64"],
                "event_acc": row["semantic_event_accuracy"],
            }
            for row in rows
        },
        "aerialmpt_t100": "qualitative free-run only; not accuracy-evaluable on current bauma3 slice",
    }


if __name__ == "__main__":
    main()
