from pathlib import Path


def verify_opentraj_path(path: str) -> dict:
    root = Path(path)
    return {"path": str(root), "exists": root.exists(), "license_note": "Verify each OpenTraj source license."}
