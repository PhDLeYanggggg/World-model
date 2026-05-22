from pathlib import Path


def verify_aerialmpt_long_path(path: str) -> dict:
    root = Path(path)
    return {"path": str(root), "exists": root.exists(), "metric_status": "pixel_or_unknown_until_homography"}
