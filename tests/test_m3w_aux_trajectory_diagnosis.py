import json
import numpy as np
from scripts.diagnose_m3w_european_aux_trajectory import diagnose, BINS


def test_zero_and_tail_contributions_add_to_total_with_equal_localities():
    rows = []
    for seed in (17, 29, 43):
        for outside in ('a', 'b', 'c', 'd'):
            for arm in ('cost_only', 'cap_aux', 'shuffled_aux'):
                value = 1. if arm == 'cap_aux' else 2.
                scores = {k: dict(rows=1, easy_harm_SSE=value) for k in BINS}
                scores['envelope_positive'] = dict(rows=5, easy_harm=value)
                scores['true_event'] = dict(BCE=.6)
                rows.append(dict(identity=dict(pair='full', producer=1, controller=2, arm=arm,
                    seed=seed, excluded_locality=outside), step=200,
                    localities={s:scores for s in ('a', 'b', 'c', 'd') if s != outside}))
    result = diagnose(rows); json.dumps(result, allow_nan=False)
    assert len(result['severity_contributions']) == 2
    for r in result['severity_contributions']:
        assert r['total_reduction_percent'] == 50.
        np.testing.assert_allclose(list(r['additive_reduction_percentage_points'].values()), 10.)
    assert all(r['fitting_localities'] == 4 for r in result['true_event_BCE'])
