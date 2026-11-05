import ast
from pathlib import Path

import numpy as np

from scripts import run_m3w_moving_public_deferral as run


def test_complete_budget_and_original_rules():
    spec = run.source.read(run.CONFIG)
    assert spec['new_heads'] == 18*3 and spec['primary_updates'] == 108000
    assert spec['training'] == run.source.read(run.old.CONFIG)['training']
    assert spec['risk_budget'] == spec['easy_net_budget'] == .02
    assert not any(spec[k] for k in ('new_H_readout', 'identity_certification', 'threshold_search', 'deployment_changed'))
    tree = ast.parse(Path(run.__file__).read_text())
    evaluate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
    body = ast.unparse(evaluate)
    assert body.index('causal_choices(') < body.index("data['target_eval']")


def test_shared_veto_and_matched_action_counts():
    point = np.zeros((4, 7, 5)); point[..., 0] = 1; point[..., 2:4] = 1
    point[:, 6, 0] = 2
    critic = point.copy(); critic[0, :, 1] = 1
    logits = np.tile(np.arange(7), (4, 1)).astype(float)
    allowed = np.ones((4, 7), bool); allowed[:, 0] = False
    c = dict(allowed=allowed, groups=[np.arange(2), np.arange(2, 4)], ids=np.arange(4))
    choices = run.causal_choices(point, logits, critic, c)
    assert np.all(choices['physical_floor'] == -1)
    np.testing.assert_array_equal(choices['public_veto'], choices['point_veto'])
    assert choices['public_veto'][0] == -1
    for arm in ('point', 'public'):
        for group in c['groups']:
            a, b = choices[arm+'_veto'][group], choices[arm+'_matched'][group]
            np.testing.assert_array_equal(np.bincount(a+1, minlength=8), np.bincount(b+1, minlength=8))
