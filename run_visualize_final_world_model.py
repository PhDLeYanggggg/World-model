from __future__ import annotations

import argparse

from src.final_model_pipeline import run_inference_demo, visualize_final_model_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize final world model demo outputs.")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        run_inference_demo()
    print(visualize_final_model_demo())


if __name__ == "__main__":
    main()
