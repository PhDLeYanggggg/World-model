from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stage20_pipeline import prepare_user_download_instructions


if __name__ == "__main__":
    prepare_user_download_instructions()
