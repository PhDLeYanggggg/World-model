from pathlib import Path


def verify_sdd_path(path: str) -> dict:
    root = Path(path)
    return {"path": str(root), "exists": root.exists(), "license_note": "SDD is non-commercial; user must accept terms."}
