from __future__ import annotations

import sys

from src.stage24_pipeline import main_correction


if __name__ == "__main__":
    main_correction(sys.argv[1:])
