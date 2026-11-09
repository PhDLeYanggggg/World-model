"""Preserve the frozen MOT20 score computation and serialize its NumPy count."""

from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import score_m3w_mot20_motion_diagnostic as scorer


def json_counts(result):
    if "ols4_lower_fde_windows" in result:
        return {**result, "ols4_lower_fde_windows": int(result["ols4_lower_fde_windows"])}
    return result


if __name__ == "__main__":
    summarize = scorer.summarize
    with patch.object(scorer, "summarize", lambda queries: json_counts(summarize(queries))):
        scorer.main()
