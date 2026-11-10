import json

import numpy as np
import pytest

from reproducibility.opendd_context_supplement.verify import replay
from scripts.build_m3w_context_review_supplement import REPORT, export, prior_builder


def test_context_supplement_replays_and_checks_same_view_budget():
    own = json.loads((REPORT/'readout.json').read_text())
    matched = json.loads((REPORT/'matched_swap_readout.json').read_text())
    rows, expected = export(own, matched)
    draws = np.random.default_rng(817).integers(7, size=(3000, 7)).tolist()
    assert prior_builder.check(replay(rows['locations'], draws), expected)
    rows['locations'][0][0]['matched']['aligned']['dual_positive'] += 1
    with pytest.raises(ValueError, match='Common swap mask changed'):
        replay(rows['locations'], draws)
