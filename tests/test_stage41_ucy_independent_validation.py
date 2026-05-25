import numpy as np

from src import stage41_ucy_independent_validation as uiv


def test_positive_safe_requires_core_positive_and_easy_preserved():
    assert uiv._positive_safe(
        {
            "all_improvement": 0.1,
            "t50_improvement": 0.1,
            "hard_failure_improvement": 0.1,
            "easy_degradation": 0.01,
        }
    )
    assert not uiv._positive_safe(
        {
            "all_improvement": 0.1,
            "t50_improvement": -0.1,
            "hard_failure_improvement": 0.1,
            "easy_degradation": 0.01,
        }
    )
    assert not uiv._positive_safe(
        {
            "all_improvement": 0.1,
            "t50_improvement": 0.1,
            "hard_failure_improvement": 0.1,
            "easy_degradation": 0.03,
        }
    )


def test_temporal_blocks_cover_ucy_rows_once():
    frame_id = np.asarray([0, 1, 2, 3, 4, 5], dtype=float)
    mask = np.asarray([True, True, False, True, True, True])
    blocks = uiv._temporal_blocks(frame_id, mask)
    total = sum(int(v.sum()) for v in blocks.values())
    assert total == int(mask.sum())
    assert all(not np.any(blocks[a] & blocks[b]) for a in blocks for b in blocks if a != b)
