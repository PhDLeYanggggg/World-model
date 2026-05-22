from __future__ import annotations

import argparse

from src.final_model_pipeline import CHECKPOINT_PATH, run_inference_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final world model inference.")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--episode", default=None)
    parser.add_argument("--scene-pack", default=None)
    parser.add_argument("--horizons", default="10,25,50")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    horizons = [int(item) for item in args.horizons.split(",") if item.strip()]
    result = run_inference_demo(
        checkpoint_path=args.checkpoint,
        episode_path=None if args.demo else args.episode,
        scene_pack_path=args.scene_pack,
        horizons=horizons,
    )
    print({"output": "outputs/final_model/inference_demo_output.json", "horizons": horizons, "agents": len(next(iter(result["predicted_trajectories"].values())))})


if __name__ == "__main__":
    main()
