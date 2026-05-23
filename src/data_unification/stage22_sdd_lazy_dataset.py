from __future__ import annotations

import json
from pathlib import Path


class Stage22SddLazyDataset:
    def __init__(self, jsonl_path: str):
        self.path = Path(jsonl_path)
        self.rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


__all__ = ["Stage22SddLazyDataset"]

