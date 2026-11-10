import json

import pytest

from reproducibility.opendd_allocation_supplement.verify import interval as bootstrap
from scripts import build_m3w_exact_exchange_review_supplement as build


def test_exact_exchange_replay_matches_frontier_and_chance_controls():
    public = build.archive.PUBLIC
    frontier = json.loads((public / 'opendd_exchange_frontier_v1/readout.json').read_text())
    ranking = json.loads((public / 'opendd_one_exchange_ranker_v1/ranking_diagnostic.json').read_text())
    groups = build.export(ranking)['locations']
    draws = build.previous.export(frontier)['draws']

    def interval(x, d):
        r = bootstrap(x, d)
        return dict(point_pp=r['point'], ci95_pp=r['interval95'], locations_positive=r['positive_locations'])

    assert build.check(build.verify.replay(groups, frontier['summary'], draws, interval), ranking['summary'])
    assert 'rdb' not in json.dumps(groups)
    groups[0][0]['exact_frontier']['positive'] += 100
    with pytest.raises(ValueError, match='contradicts the archived frontier'):
        build.verify.replay(groups, frontier['summary'], draws, interval)
