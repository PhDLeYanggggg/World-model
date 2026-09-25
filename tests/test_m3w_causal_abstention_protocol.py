import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import run_m3w_european_causal_abstention as run


def test_frozen_matrix_and_roles():
    cfg = json.loads((run.ROOT/run.CONFIG).read_text()); run.validate(cfg)
    assert 2*3*3*2*2*10 == cfg['views']
    for key, change in [('views', 36), ('minimum_sources', 1), ('threshold_search', True), ('reserved_roles_opened', True)]:
        bad = copy.deepcopy(cfg); bad[key] = change
        with pytest.raises(ValueError): run.validate(bad)


def test_causal_inputs_ignore_future_arrays_and_only_fit_train():
    cfg = json.loads((run.ROOT/run.CONFIG).read_text()); n = 160
    rng = np.random.default_rng(1)
    data = dict(history=rng.normal(size=(n, 8, 2)), sites=np.array(['a']*40+['b']*40+['held']*80),
                recordings=np.array(['r']*n), frames=np.arange(n), baseline_ade=np.ones((n, 6)))
    design = dict(train_ids=np.arange(80), held_ids=np.arange(80, 160))
    a = dict(p=rng.normal(size=(n, 12, 2))); floor = np.zeros((n, 12, 2))
    original = run.inputs(cfg, data, design, a, floor)
    data['baseline_ade'][80:] = np.nan
    data['target_eval'] = 'deliberately forbidden future values'
    data['valid'] = 'future mask must not gate inference'
    again = run.inputs(cfg, data, design, a, floor)
    assert original[2] == again[2] and original[2]['sources'] == ['a', 'b']
    for i in (0, 1, 3): np.testing.assert_array_equal(original[i], again[i])


def test_decision_function_has_no_outcome_readout():
    tree = ast.parse(Path(run.__file__).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'decide')
    text = ast.unparse(function)
    for forbidden in ('target_eval', 'native_errors', 'paired_scene_metrics', 'selected_error'):
        assert forbidden not in text
    evaluate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
    text = ast.unparse(evaluate)
    assert text.index('ensure_both_frozen()') < text.index('native_errors(')
    assert text.index('verify_variants(') < text.index('native_errors(')


def test_removed_harm_lost_benefit_accounting():
    use = np.array([0, 0, 1, 0], bool); original = np.array([1, 1, 1, 1], bool)
    neural = np.array([8., 1., 4., np.nan]); floor = np.array([2., 3., 3., np.nan])
    r = run.removal_ledger(use, original, neural, floor, np.array(['a']*4), {'all': np.ones(4, bool)})['all']['a']
    assert r['removed_indexed'] == 3 and r['removed_known'] == 2
    assert r['avoided_harm'] == 6 and r['lost_benefit'] == 2
    assert r['gain_change_pp'] == 50
