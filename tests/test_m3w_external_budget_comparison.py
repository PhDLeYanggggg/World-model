import json

import pytest

from scripts.audit_m3w_external_budget_comparison import PUBLIC, SOURCES, replay


@pytest.fixture
def receipts():
    return {k: (PUBLIC/p).read_bytes() for k, p in SOURCES.items()}


def test_source_pairing_and_release_are_separate(receipts):
    result = replay(receipts)
    assert not result['pooling'] and not result['new_outcomes_read']
    rows = result['cohorts']
    assert [rows[k]['units_resampled'] for k in ('wep', 'songdo', 'athens')] == [10, 14, 4]
    assert rows['wep']['joint_fixed_dual'] == [0, 45]
    assert rows['wep']['joint_released_edits'] == 0
    assert rows['songdo']['joint_fixed_dual'] == [53, 480]
    assert rows['songdo']['joint_released_views'] == 0
    assert rows['athens']['joint_fixed_dual'] == [1, 18]
    assert rows['athens']['release'] == 'not_run'
    assert rows['athens']['equal_steps_minus_original_observed_pp'] == pytest.approx(-.00094444836079)
    assert rows['athens']['equal_steps']['envelope_pp'][0] < 0 < rows['athens']['equal_steps']['envelope_pp'][1]


def test_different_source_support_is_rejected(receipts):
    obj = json.loads(receipts['wep_steps'])
    obj['source_values']['fixed_independent']['matched_steps']['upper'][0] = None
    receipts['wep_steps'] = json.dumps(obj).encode()
    with pytest.raises(ValueError, match='Incomplete paired source support'):
        replay(receipts)


def test_changed_release_role_is_rejected(receipts):
    obj = json.loads(receipts['athens_steps'])
    obj['release_recomputed'] = True
    receipts['athens_steps'] = json.dumps(obj).encode()
    with pytest.raises(ValueError, match='retain their original roles'):
        replay(receipts)
