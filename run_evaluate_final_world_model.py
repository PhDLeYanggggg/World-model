from __future__ import annotations

import argparse

from src.final_model_pipeline import evaluate_final_model, write_final_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BPSG-MA World Model v1.")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    metrics = evaluate_final_model(quick=args.quick)
    write_final_reports()
    print(metrics)


if __name__ == "__main__":
    main()
