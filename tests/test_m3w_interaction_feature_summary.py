from copy import deepcopy

import pytest

from scripts.run_m3w_interaction_feature_cost import summarize


def fixture():
    roster = [str(i) for i in range(12)]
    view = dict(supported=True, known_all_risk=.01, known_easy_risk=.01,
                completion_all_risk_upper=.01, completion_easy_risk_upper=.01,
                known_easy_net_degradation=0.)
    policy = dict(known_ADE_gain_floor_percent=1., completion_lower_gain_floor_percent=-2.,
                  switch_rate=.2, subsets=dict(easy=dict(gain_percent=0.), hard=dict(gain_percent=2.)),
                  recordings={'record': view})
    delta = dict(known_pp=0., completion_lower_pp=0., completion_upper_pp=0.)
    groups = [dict(trial=f'p{producer}s{seed}', seed=seed, site=site,
                   policies={'interaction64_bounded': deepcopy(policy)},
                   primary=deepcopy(delta), interaction_minus_direct=deepcopy(delta), zero_minus_direct=deepcopy(delta))
              for site in roster for producer in (0, 1) for seed in (17, 29, 43)]
    cfg = dict(bootstrap_resamples=30, bootstrap_seed=11, primary='synthetic_fixed_primary')
    return groups, cfg, roster


def test_zero_effect_or_individual_bound_is_not_promoted_to_primary_success():
    rows, cfg, roster = fixture()
    result = summarize(rows, cfg, roster)
    assert result['contrasts']['primary']['completion_lower_pp']['CI95'] == [0., 0.]
    assert not result['primary_supported']
    assert result['original_risk_pass']
    assert not result['submission_ready'] and not result['independent_confirmation']
    for row in rows:
        row['primary']['completion_lower_pp'] = .5
    result = summarize(rows, cfg, roster)
    assert result['primary_supported']


def test_incomplete_seed_roster_and_undefined_support_are_not_silently_dropped():
    rows, cfg, roster = fixture()
    rows[0]['seed'] = 99
    with pytest.raises(ValueError):
        summarize(rows, cfg, roster)
    rows, cfg, roster = fixture()
    row = rows[0]['policies']['interaction64_bounded']['recordings']['record']
    row.update(supported=False, completion_easy_risk_upper=None, known_easy_net_degradation=None)
    result = summarize(rows, cfg, roster)
    assert not result['original_risk_pass']
    assert result['safety']['interaction64_bounded']['undefined_easy_views'] == 1
