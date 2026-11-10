import numpy as np
import pytest

from src.world_model import m3w_onpolicy_donors as model


def test_nested_rosters_exclude_outer_donor_and_inner_target():
    sites = tuple('abcdefg')
    assert model.fit_roster(sites, ('a', 'b')) == list('cdefg')
    for inner in 'cdefg':
        fit = model.fit_roster(sites, ('a', 'b', inner))
        assert len(fit) == 4 and not set(fit) & {'a', 'b', inner}
        model.verify_role(dict(omitted=sorted(('a', 'b', inner)), fit=fit), sites, ('a', 'b', inner))
    with pytest.raises(ValueError, match='exposed'):
        model.verify_role(dict(omitted=['a', 'b'], fit=list('bcdefg')), sites, ('a', 'b'))
    with pytest.raises(ValueError, match='roster'):
        model.fit_roster(sites, ('a', 'a'))


def test_label_blind_score_anchor_matches_budget_and_can_keep_zero():
    legal = np.ones((10, 3), dtype=bool)
    movement = np.tile([0., 1., 2.], (10, 1))
    scores = np.tile([0., 1., -1.], (10, 1))
    count, cap, status = model.score_anchor(scores, legal, movement, ['r']*10, np.arange(10))
    assert count.sum() == 2 and cap == 2. and status == 'score_optimal'
    scores[:, 0] = 2.
    zero, cap, status = model.score_anchor(scores, legal, movement, ['r']*10, np.arange(10))
    assert not zero.any() and cap == 0. and status == 'zero_realized'
    with pytest.raises(ValueError, match='one recording'):
        model.score_anchor(scores, legal, movement, ['r']*5+['s']*5, np.arange(10))


def test_unresolved_allocation_is_not_silently_replaced(monkeypatch):
    monkeypatch.setattr(model.capacity, 'allocate', lambda *args, **kwargs: (None, 'unresolved'))
    count, _, status = model.score_anchor(np.tile([0., 1., -1.], (10, 1)),
        np.ones((10, 3), bool), np.tile([0., 1., 2.], (10, 1)), ['r']*10, np.arange(10))
    assert count is None and status == 'unresolved'


def test_aggregation_preserves_zeros_and_negative_differences():
    assert model.total_fields([{'a': -2, 'b': 0}, {'a': 1}], ['a', 'b', 'c']) == {
        'a': -1, 'b': 0, 'c': 0}


def test_summary_keeps_empty_counts_and_requires_previous_policy_gain():
    from scripts import run_m3w_opendd_onpolicy_donors as runner

    fields = (*runner.previous.FIELDS, 'rescued_full_grid', 'destroyed_full_grid')
    def row():
        arms = {k: dict.fromkeys(fields, 0.) for k in runner.previous.ARMS}
        for k in ('pair_optional', 'pair_forced', 'point_optional'):
            arms[k]['full_grid'] = 1.
        return dict(seed=17, counts=dict(views=10, edits=4, eligible=2),
                    switches=dict.fromkeys(runner.previous.POLICIES, 1), arms=arms)
    groups = {str(i): [row()] for i in range(7)}
    reference = dict(summary=dict(counts=dict(views=70, edits=28, eligible=14),
        arms={k: dict.fromkeys(fields, 0.) for k in runner.previous.CONTROLS}), by_location=groups)
    result = runner.summarize(groups, reference)
    assert result['arms']['incumbent']['material_full_grid'] == 0
    assert result['contrasts']['incumbent']['ci95_pp'] == [10., 10.]
    assert result['contrasts']['old_pair_optional']['ci95_pp'] == [0., 0.]
    assert not result['primary_supported']
