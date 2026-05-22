from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stage14_pipeline import multimodal_data_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 14 legal multimodal pedestrian/drone data dry-run and local verifier.")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--verify-local", default=None)
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()
    result = multimodal_data_audit(verify_local=args.verify_local, dataset=args.dataset)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

