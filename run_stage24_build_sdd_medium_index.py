from __future__ import annotations

import sys

from src.stage24_pipeline import main_build_index


if __name__ == "__main__":
    main_build_index(sys.argv[1:])
