#!/usr/bin/env python
from __future__ import annotations

import json

from src.scene.stage8_scene_gold_builder import build_scene_gold_packs, write_report


if __name__ == "__main__":
    payload = write_report(build_scene_gold_packs())
    print(json.dumps({k: v for k, v in payload.items() if k != "scene_packs"}, indent=2))

