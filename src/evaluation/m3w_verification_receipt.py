"""Retain original execution receipts while rechecking scientific identities."""
import json
from pathlib import Path


def retain_verified_receipt(path, value, *, fields, flag, immutable_write):
    path = Path(path)
    if flag not in fields or value.get(flag) is not True or any(k not in value for k in fields):
        raise ValueError('Complete successful scientific verification identity required')
    if not path.exists():
        return immutable_write(path, value)
    saved = json.loads(path.read_text())
    if any(saved.get(key) != value[key] for key in fields):
        raise ValueError('Scientific verification differs; runtime variance cannot conceal it')
    return saved
