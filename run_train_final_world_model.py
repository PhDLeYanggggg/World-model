from __future__ import annotations

import argparse

from src.final_model_pipeline import train_final_model, write_final_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPSG-MA World Model v1.")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = train_final_model(quick=args.quick)
    write_final_reports()
    print(result)


if __name__ == "__main__":
    main()
