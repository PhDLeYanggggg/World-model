from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajnet-root", default="")
    parser.add_argument("--eth-ucy-root", default="")
    parser.add_argument("--sdd-root", default="")
    parser.add_argument("--opendd-root", default="")
    parser.add_argument("--ngsim-root", default="")
    args = parser.parse_args()
    paths = {key[:-5] if key.endswith("_root") else key: value for key, value in vars(args).items() if value}
    results = {}
    for name, path in paths.items():
        root = Path(path)
        results[name] = {
            "path": str(root),
            "exists": root.exists(),
            "file_count": sum(1 for item in root.rglob("*") if item.is_file()) if root.exists() else 0,
        }
    out = Path("outputs/reports/stage5b_path_verification.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0 if all(row["exists"] for row in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
