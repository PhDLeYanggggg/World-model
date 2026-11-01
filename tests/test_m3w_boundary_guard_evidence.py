import copy
import json
import numpy as np
import pytest
from scripts import run_m3w_boundary_guard as run
from tests.test_m3w_active_bank_readout import fixture


def test_complete_original_roster_and_negative_safety_are_not_hidden():
    contexts = {f'c{i}': [f's{4*(i%3)+j}' for j in range(4)] for i in range(6)}
    rows = []
    for context, sources in contexts.items():
        for source in sources:
            p = fixture(source, missing=True)[0]['new_neural']
            choices = {a: np.full(2, 6) for a in run.evaluation.NAMES if a != 'floor'}
            predictions = {a: np.ones_like(p['y']) for a in run.api.ARMS}
            ranks = {a: np.ones(2) for a in choices}
            result = run.evaluation.readout(p, choices, ranks, predictions, 1.5, np.zeros((2, 2)))
            for seed in (17, 29, 43):
                rows.append(dict(context=context, source=source, seed=seed, result=copy.deepcopy(result),
                    matched_training=True, original_control_exact=True, guarded_predicted_feasible=True, minimum_gradient_query_visits=4))
    cfg = dict(seeds=[17, 29, 43], bootstrap_seed=7, bootstrap_resamples=32)
    s = run.evaluation.summarize(rows, contexts, cfg)
    assert s['groups'] == 72
    assert not s['fitting_advancement'] and not s['independent_confirmation']
    assert s['safety']['boundary_guarded']['selected_unknown'] > 0
    with pytest.raises(ValueError): run.evaluation.summarize(rows[:-1], contexts, cfg)
    with pytest.raises(ValueError): run.evaluation.summarize(rows[:-1]+rows[:1], contexts, cfg)


def test_registered_bindings_when_present():
    if not run.REG.exists(): pytest.skip('Definition not yet registered')
    reg = json.loads(run.REG.read_text())
    for rel, digest in dict(reg['bindings'], **reg['dependencies']).items():
        assert run.sha(run.ROOT/rel) == digest
    assert reg['fits'] == 36 and reg['parameters_each'] == 31119
    assert not reg['independent_roles_opened']


def test_actual_complete_readout_when_present():
    if not (run.PUBLIC/'verification.json').exists(): pytest.skip('Full experiment not yet complete')
    c = json.loads((run.PUBLIC/'complete.json').read_text()); v = json.loads((run.PUBLIC/'verification.json').read_text())
    assert len(c['groups']) == 72 and v['all72_new_cost_inference_and_action_scores_exact']
    assert v['summary_sha256'] == c['summary_sha256'] == run.sha(run.PUBLIC/'summary.json')
    assert v['complete_sha256'] == run.sha(run.PUBLIC/'complete.json')
