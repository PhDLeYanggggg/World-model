#!/usr/bin/env python
from __future__ import annotations

import json

from src.scene.scene_pack_builder import build_scene_packs, write_report


if __name__ == "__main__":
    payload = write_report(build_scene_packs())
    print(json.dumps({"total_scene_packs": payload["total_scene_packs"]}, indent=2))

