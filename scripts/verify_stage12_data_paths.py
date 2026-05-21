from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify optional Stage 12 local pedestrian/drone dataset paths.")
    parser.add_argument("--sdd-root", default="")
    parser.add_argument("--opentraj-root", default="")
    parser.add_argument("--trajnet-root", default="data/stage5b_raw/trajnetplusplusdataset")
    parser.add_argument("--aerialmpt-zip", default="data/aerialmpt/DLR_AerialMPT_Dataset.zip")
    args = parser.parse_args()
    result = {k: {"path": v, "exists": bool(v and Path(v).exists())} for k, v in vars(args).items()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
