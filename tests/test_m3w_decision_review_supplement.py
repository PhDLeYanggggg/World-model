import json

import numpy as np
import pytest

from reproducibility.opendd_allocation_supplement.verify import interval
from scripts import build_m3w_decision_review_supplement as build


def test_anonymous_decision_replay_preserves_inherited_budget():
    public = build.archive.PUBLIC
    pair, conditional = (json.loads((public / name / 'readout.json').read_text()) for name in (build.PAIR, build.CONDITIONAL))
    groups = build.export(pair, conditional)['locations']
    inherited = [[{
        'views': row['views'], 'edits': row['scores']['frozen']['edits'], 'matched': {'frozen': row['scores']['frozen']}}
        for row in cells] for cells in groups]
    draws = np.random.default_rng(817).integers(7, size=(3000, 7)).tolist()
    replay = build.verify.replay(groups, inherited, draws, interval)
    build.check(replay, pair, conditional)
    assert 'rdb' not in json.dumps(groups)
    groups[0][0]['scores']['conditional_utility']['edits'] += 1
    with pytest.raises(ValueError, match='changed budget'):
        build.verify.replay(groups, inherited, draws, interval)
