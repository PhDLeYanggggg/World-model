import copy
from scripts.diagnose_m3w_european_cap_auxiliary_cost import diagnose, percent_gain


def test_zero_reference_is_unknown_not_infinite_gain():
    assert percent_gain(1, 0) is None
    assert percent_gain(None, 1) is None
    assert percent_gain(float('nan'), 1) is None
    assert percent_gain(1, 2) == 50


def test_fit_gain_cannot_be_reported_as_held_gain():
    def metrics(value):
        return {'envelope_positive': {'harm_MSE': value}}
    fitting = {arm: {'training': metrics(value)} for arm, value in
               [('cost_only', 2), ('cap_aux', 1), ('shuffled_aux', 2), ('original', 2)]}
    held = {arm: metrics(value) for arm, value in
            [('cost_only', 2), ('cap_aux', 3), ('shuffled_aux', 2), ('original', 2)]}
    rows = [{'group': 'fixture', 'pair': pair, 'folds': [{'held': 'site', 'training': fitting, 'metrics': held}]}
            for pair in ('full', 'motion_only')]
    before = copy.deepcopy(rows)
    result = diagnose(rows, [])
    item = result['comparisons']['full']['aux_vs_control']
    assert item['fitting']['median'] == 50 and item['held']['median'] == -50
    assert item['fit_positive_held_negative'] == 1
    assert not result['gates_changed'] and not result['causal_mechanism_identified']
    assert before == rows


def test_legacy_all_rows_cannot_substitute_for_positive_envelope():
    sub = {'envelope_positive': {'harm_MSE': 1}}
    training = {arm: {'training': sub} for arm in ('cost_only', 'cap_aux', 'shuffled_aux')}
    training['original'] = {'training': {'harm_MSE': .001, 'rows': 100}}
    row = {'group': 'fixture', 'pair': 'full', 'folds': [{'held': 'site',
           'training': training, 'metrics': {arm: sub for arm in training}}]}
    item = diagnose([row], [])['comparisons']['full']['aux_vs_original']
    assert item['fitting']['missing'] == 1 and item['fitting']['median'] is None
    assert item['held']['median'] == 0
    assert item['missing_fit_reason'] == 'legacy_original_has_all_rows_only_not_same_subset'
