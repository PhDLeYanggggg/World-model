from __future__ import annotations

import argparse
import json

from src.scene.stage14_multimodal_scene_pack_builder import build


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 14 multimodal scene packs.")
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(build(limit=args.limit), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

