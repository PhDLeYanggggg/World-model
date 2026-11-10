import json

import pytest

from scripts import build_m3w_frontier_review_supplement as build


def test_frontier_replay_checks_the_ranges_and_inherited_count():
    raw = json.loads((build.archive.PUBLIC / 'opendd_exchange_frontier_v1/readout.json').read_text())
    data = build.export(raw)
    replay = build.verify.replay(data, 60)
    build.check(replay, raw['summary'])
    assert 'rdb' not in json.dumps(data)
    with pytest.raises(ValueError, match='Inherited cohort'):
        build.verify.replay(data, 59)
    cell = data['locations'][0][0]
    cell['bounds']['one']['positive'][1] = -1
    with pytest.raises(ValueError, match='nonnested'):
        build.verify.replay(data, 60)
