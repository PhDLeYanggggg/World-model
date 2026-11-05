import ast
from pathlib import Path

from scripts import run_m3w_moving_proposal_risk as run


def test_complete_budget_and_no_changed_rules():
    spec = run.source.read(run.CONFIG)
    assert spec['new_heads'] == 18*3*2 and spec['primary_updates'] == 216000
    assert spec['training'] == run.source.read(run.parent.CONFIG)['training']
    assert spec['risk_budget'] == spec['easy_net_budget'] == .02
    assert not any(spec[k] for k in ('new_H_readout', 'identity_certification', 'threshold_search', 'deployment_changed'))


def test_decisions_before_labels_and_full_freeze():
    tree = ast.parse(Path(run.__file__).read_text())
    functions = {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert functions['training_packet'].index('router.choose(') < functions['training_packet'].index("data['target_eval']")
    assert functions['evaluate'].index('api.causal_choices(') < functions['evaluate'].index("data['target_eval']")
    assert "freeze['complete_heads'] != 108" in functions['evaluate']
    assert 'parent_control_replay_exact=True' in functions['evaluate']
    assert "'initial_model', 'preprocess', 'sampler', 'draw_hash'" in functions['train']
