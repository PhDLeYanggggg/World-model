from collections import Counter
import json

import pytest

from scripts.audit_m3w_pneuma_date_robustness import FILES, SOURCE, envelope, replay


@pytest.fixture
def readouts():
    return {k: (SOURCE/f'{k}.json').read_bytes() for k in FILES}


def test_frozen_athens_date_robustness(readouts):
    d = replay(readouts)
    assert sorted(d['date_counts'].values()) == [1, 1, 1, 3]
    c = d['contrasts']
    for key in ('joint_vs_cv', 'same_family_vs_cv'):
        assert c[key]['dates_definitely_positive'] == 4
        assert c[key]['all_leave_one_out_positive']
        assert c[key]['any_date_mixture_envelope_pp'][0] > 0
    for key in ('joint_vs_independent', 'joint_vs_additive',
                'joint_vs_independent_equal_step_movement'):
        assert c[key]['dates_including_zero'] == 4
        assert c[key]['all_leave_one_out_include_zero']
    assert c['joint_vs_cv']['equal_date_pp'][0] == pytest.approx(.02383170191695845)
    assert c['joint_vs_same_family']['dates_definitely_negative'] == 3
    assert c['joint_vs_same_family']['dates_definitely_positive'] == 1


def test_mismatched_secondary_binding_refused(readouts):
    steps = json.loads(readouts['stepwise_budget_sensitivity'])
    steps['attribution_receipt'] = 'invalid'
    readouts['stepwise_budget_sensitivity'] = json.dumps(steps).encode()
    with pytest.raises(ValueError, match='Secondary analyses'):
        replay(readouts)


def test_missing_acquisition_date_refused(readouts):
    d = json.loads(readouts['confirmation'])
    m = d['matched']['joint_minus']['independent']
    m['lower_pp']['by_date'].pop(next(iter(d['dates'].values())))
    with pytest.raises(ValueError, match='Mismatched acquisition-date aggregation'):
        envelope(m, Counter(d['dates'].values()))
