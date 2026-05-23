from __future__ import annotations

import sys

from src.stage23_pipeline import main_train_selector


if __name__ == "__main__":
    main_train_selector(sys.argv[1:])
