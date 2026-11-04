import ast
from pathlib import Path
import numpy as np
from scripts import run_m3w_moving_joint_harm as run


def test_original_roles_budgets_and_complete_pairing():
    spec = run.source.read(run.CONFIG)
    assert spec['new_heads'] == 18*3*2 and spec['primary_updates'] == 216000
    assert spec['training'] == run.source.read(run.old.CONFIG)['training']
    assert spec['risk_budget'] == spec['easy_net_budget'] == .02
    assert not any(spec[k] for k in ('new_H_readout', 'identity_certification', 'threshold_search', 'deployment_changed'))
    tree = ast.parse(Path(run.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    assert 'held' not in ast.unparse(main)


def test_veto_only_and_query_amplitude_matched():
    p = np.zeros((4, 7, 5)); p[..., 0] = 1; p[..., 2:4] = 1
    p[:, 6, 0] = 2
    critic = dict(moments=p.copy())
    critic['moments'][0, :, 1] = 1
    allowed = np.ones((4, 7), bool); allowed[:, 0] = False
    c = dict(allowed=allowed, groups=[np.arange(2), np.arange(2, 4)], ids=np.arange(4))
    choices = run.causal_choices(p, {a: critic for a in run.api.ARMS}, c)
    for arm in run.api.ARMS:
        a, b = choices[arm+'_veto'], choices[arm+'_matched']
        assert a[0] == -1 and np.all(a[1:] == choices['frozen_proposer'][1:])
        for group in c['groups']:
            np.testing.assert_array_equal(np.bincount(a[group]+1, minlength=8), np.bincount(b[group]+1, minlength=8))
