import json

import pytest

from scripts import build_m3w_allocation_review_supplement as build


def test_anonymous_scalar_replay_matches_frozen_tables_and_rejects_changed_budget():
    public = build.archive.PUBLIC
    a = json.loads((public / 'opendd_count_motion_v1/readout.json').read_text())
    b = json.loads((public / 'opendd_prefix_value_v1/readout.json').read_text())
    scalars = build.payload(a, b)
    assert 'rdb' not in json.dumps(scalars)
    result = build.verify.replay(scalars)
    build.check_replay(result, a, b)
    assert result['motion']['count_motion'] == 210
    assert result['matched']['joint_value']['dual_positive'] == 42
    scalars['locations'][0][0]['matched']['joint_value']['edits'] += 1
    with pytest.raises(ValueError, match='changed the number of edits'):
        build.verify.replay(scalars)
