from __future__ import annotations

import sys

from src.stage23_pipeline import main_eval_selector


if __name__ == "__main__":
    main_eval_selector(sys.argv[1:])
