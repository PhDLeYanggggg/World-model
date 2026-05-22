from pathlib import Path


def verify_full_trajnet_path(path: str) -> dict:
    root = Path(path)
    return {"path": str(root), "exists": root.exists()}
