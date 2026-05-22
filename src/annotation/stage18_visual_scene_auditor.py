from __future__ import annotations


def audit_visual_scene(annotation):
    return {
        "visual_audit_available": bool(annotation.get("scene_image_path")),
        "quality_ceiling": "self_audited_silver",
        "gold_human": False,
    }

