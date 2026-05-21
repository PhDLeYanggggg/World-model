from __future__ import annotations

import json
from pathlib import Path

from src.data.tgsim_adapter import TGSIM_FOGGY_BOTTOM_CSV_URL, summarize_tgsim_file


def main() -> None:
    # Pass a local CSV path as the first CLI arg if you have already downloaded it.
    import sys

    path_or_url = sys.argv[1] if len(sys.argv) > 1 else TGSIM_FOGGY_BOTTOM_CSV_URL
    max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
    summary = summarize_tgsim_file(path_or_url, max_rows=max_rows)
    out = Path("outputs/reports/tgsim_preview_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
