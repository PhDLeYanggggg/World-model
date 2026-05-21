from __future__ import annotations

from src.scene.stage8p5_scene_gold_builder import build_stage8p5_scene_gold_packs, write_scene_gold_pack_report


if __name__ == "__main__":
    payload = build_stage8p5_scene_gold_packs()
    write_scene_gold_pack_report(payload)
    print({k: v for k, v in payload.items() if k != "scene_packs"})
